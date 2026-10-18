from synthetic_history import SyntheticHistoryTestCase
"""Portfolio integrity, denominators, filters, charts, exports and absence of fake AUC."""
import hashlib
import io
import runpy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import pandas as pd
from streamlit.testing.v1 import AppTest
from utils.client_dashboard_data import load_history,export_history_csv,distribution_figure
from utils.global_dashboard_data import (filter_portfolio,portfolio_kpis,daily_summary,daily_figures,
    recorded_scores,score_figure,category_figure,threshold_figure)
from utils.prediction_logs import append_prediction_record,COLUMNS
from test_reports import ROOT,LOG,BASELINE_HASH,row,write_csv
from test_client_dashboard import fixtures


def page(path):
    script=('import runpy\n'+f"ns=runpy.run_path({str(ROOT/'app/global_dashboard.py')!r},run_name='global_test')\n"
        +f"ns['main'].__globals__['LOG_PATH']={str(path)!r}\nns['main']()\n")
    return AppTest.from_string(script,default_timeout=90)


class PortfolioDataTests(SyntheticHistoryTestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.path=Path(self.temp.name)/'history.csv'
        self.df=fixtures(self.path)
    def tearDown(self):
        self.temp.cleanup()

    def test_4_records_dates_and_bytes_unchanged(self):
        before=LOG.read_bytes()
        df=load_history(LOG)
        self.assertEqual(len(df),4);self.assertEqual(df['_timestamp'].isna().sum(),0)
        self.assertEqual(hashlib.sha256(before).hexdigest(),BASELINE_HASH)
        self.assertEqual(LOG.read_bytes(),before)

    def test_kpi_formulas_and_unknown_decision_denominator(self):
        k=portfolio_kpis(self.df)
        self.assertEqual(k['total'],6);self.assertEqual(k['unique_clients'],2)
        self.assertEqual(k['valid_pd'],4);self.assertAlmostEqual(k['mean_pd'],.29)
        self.assertEqual(k['recorded_decisions'],4);self.assertEqual(k['missing_decisions'],2)
        self.assertEqual(k['recognized_decisions'],2);self.assertEqual(k['accepted'],1)
        self.assertEqual(k['unrecognized_decisions'],2);self.assertEqual(k['acceptance_rate'],.5)
        self.assertAlmostEqual(k['historical_above_rate'],2/3)
        self.assertNotIn('auc',k);self.assertNotIn('observed_default_rate',k)

    def test_unavailable_metrics_are_not_zero(self):
        invalid=self.df[self.df['_pd'].isna()].copy()
        invalid['decision']=None
        k=portfolio_kpis(invalid,.4)
        for name in ('mean_pd','acceptance_rate','historical_above_rate','simulation_rate'):
            self.assertIsNone(k[name])
        empty=portfolio_kpis(self.df.iloc[:0],.4)
        self.assertEqual(empty['total'],0);self.assertIsNone(empty['acceptance_rate'])

    def test_simulation_never_changes_decisions_or_historical_metrics(self):
        before=self.df.copy(deep=True)
        low=portfolio_kpis(self.df,.05);high=portfolio_kpis(self.df,.95)
        self.assertEqual(low['simulation_rate'],.75);self.assertEqual(high['simulation_rate'],0)
        for key in ('acceptance_rate','historical_above_rate','decision_counts','mean_pd','total'):
            self.assertEqual(low[key],high[key])
        pd.testing.assert_frame_equal(before,self.df)

    def test_filters_clients_rating_decision_model_and_missing_values(self):
        selected=filter_portfolio(self.df,clients=['A'],ratings=['BB'],decisions=['ACCEPT'],models=['pipeline_v1'])
        self.assertEqual(selected['_record_number'].tolist(),[2])
        selected=filter_portfolio(self.df,clients=[None])
        self.assertEqual(selected['_record_number'].tolist(),[4])
        selected=filter_portfolio(self.df,decisions=[None])
        self.assertEqual(set(selected['_record_number']),{4,5})
        df=self.df.copy()
        df.loc[df['_record_number']==3,'rating']=None
        self.assertEqual(filter_portfolio(df,ratings=[None])['_record_number'].tolist(),[3])
        self.assertTrue(filter_portfolio(self.df,models=['unknown']).empty)

    def test_dates_mix_timezone_whole_last_day_undated_and_recent_window(self):
        df=filter_portfolio(self.df,start='2026-02-01',end='2026-02-01',include_undated=True)
        self.assertEqual(set(df['_record_number']),{2,3})
        df=filter_portfolio(self.df,start='2026-02-01',end='2026-02-01',include_undated=False)
        self.assertEqual(df['_record_number'].tolist(),[2])
        df=filter_portfolio(self.df,last_30_days=True,include_undated=False,now='2026-02-10T12:00:00Z')
        self.assertEqual(df['_record_number'].tolist(),[2]) # future March row excluded
        df=filter_portfolio(self.df,last_30_days=True,include_undated=True,now='2026-02-10T12:00:00Z')
        self.assertEqual(set(df['_record_number']),{2,3})

    def test_daily_volume_counts_missing_ids_and_pd_mean_uses_only_valid_pd(self):
        df=self.df.copy()
        df['_timestamp']=pd.to_datetime(df['_record_number'].map(lambda value: '2026-01-01' if value != 6 else None),utc=True)
        daily=daily_summary(df)
        self.assertEqual(daily.iloc[0]['volume'],5) # includes missing client and invalid PD
        self.assertEqual(daily.iloc[0]['valid_pd'],3)
        self.assertAlmostEqual(daily.iloc[0]['mean_pd'],(.06+.9+.2)/3)
        mean,volume=daily_figures(df)
        self.assertEqual(sum(volume.data[0].y),5)
        self.assertEqual(mean.layout.yaxis.tickformat,'.0%')
        self.assertEqual(mean.data[0].customdata[0][0],3)
        self.assertEqual(mean.data[0].customdata[0][1],5)

    def test_daily_all_invalid_pd_has_gap_not_zero(self):
        df=self.df[self.df['_pd'].isna()].copy()
        mean,volume=daily_figures(df)
        self.assertTrue(all(value is None for value in mean.data[0].y))
        self.assertEqual(sum(volume.data[0].y),1) # other invalid row has no date

    def test_distribution_only_valid_pd_and_thresholds(self):
        pd_fig=distribution_figure(self.df,.4)
        threshold=threshold_figure(self.df)
        self.assertEqual(sorted(pd_fig.data[0].x),[0,.06,.2,.9])
        self.assertEqual(sorted(threshold.data[0].x),[0,.1,.3,.5,.8])
        self.assertEqual(threshold.layout.xaxis.tickformat,'.0%')
        self.assertFalse(threshold.layout.shapes)
        self.assertEqual(pd_fig.layout.shapes[0].x0,.4)

    def test_scores_are_archived_only_not_reconstructed_and_invalid_excluded(self):
        df=self.df.copy()
        df.loc[df['_record_number']==1,'score_1000']=None
        df.loc[df['_record_number']==2,'score_1000']=1001
        self.assertEqual(recorded_scores(df).notna().sum(),4)
        fig=score_figure(df)
        self.assertEqual(sorted(fig.data[0].x),[10,777,800,1000])
        self.assertEqual(fig.layout.xaxis.range,(0,1000))
        legacy=self.df.drop(columns=['score_1000'])
        self.assertEqual(recorded_scores(legacy).notna().sum(),0)
        self.assertEqual(len(score_figure(legacy).data[0].x),0)

    def test_category_charts_include_missing_keep_archived_labels(self):
        df=self.df.copy()
        df.loc[df['_record_number']==3,'rating']=None
        for column in ('rating','decision','_client'):
            fig=category_figure(df,column,column)
            self.assertEqual(sum(fig.data[0].y),len(df))
            self.assertIn('Non renseigne',fig.data[0].x)
        fig=category_figure(df,'decision','Decisions')
        self.assertIn('REFUS-HIST',fig.data[0].x)

    def test_csv_jsonl_export_matches_filtered_archived_result(self):
        selected=filter_portfolio(self.df,clients=['A'],decisions=['ACCEPT'])
        result=pd.read_csv(io.BytesIO(export_history_csv(selected)))
        self.assertEqual(len(result),1)
        self.assertEqual(result.iloc[0]['score_1000'],456)
        self.assertEqual(result.iloc[0]['rating'],'BB')
        self.assertEqual(result.iloc[0]['decision'],'ACCEPT')
        self.assertEqual(result.iloc[0]['risk_level'],'SOURCE-RISK')
        self.assertEqual(result.iloc[0]['model_id'],'MODEL-ID')
        self.assertEqual(result.iloc[0]['threshold'],.8)

    def test_mixed_7_9_columns_and_rerun_detects_append_without_cache(self):
        path=Path(self.temp.name)/'mixed.csv'
        first=row('OLD');second=row('NEW')
        write_csv(path,COLUMNS[:7],[[first[k] for k in COLUMNS[:7]],[second[k] for k in COLUMNS]])
        before=path.read_bytes()
        module=runpy.run_path(str(ROOT/'app/global_dashboard.py'),run_name='global_data_test')
        df=module['load_logs'](path)
        self.assertEqual(len(df),2);self.assertEqual(path.read_bytes(),before)
        self.assertIsNone(df[df['_client']=='OLD'].iloc[0]['rating'])
        append_prediction_record(path,row('THIRD'),dict(score_1000=937,risk_level='API-RISK'))
        df=module['load_logs'](path)
        self.assertEqual(len(df),3)
        self.assertEqual(df[df['_client']=='THIRD'].iloc[0]['score_1000'],937)


class GlobalUITests(SyntheticHistoryTestCase):
    def test_navigation_4_rows_and_no_auc_performance_metric(self):
        before=LOG.read_bytes()
        app=AppTest.from_file(str(ROOT/'streamlit_app.py'),default_timeout=90).run()
        self.assertEqual(len(app.exception),0,str(app.exception))
        home_auc=[metric for metric in app.metric if 'AUC' in metric.label]
        self.assertEqual(len(home_auc),1);self.assertEqual(home_auc[0].value,'Indisponible')
        app.switch_page('app/global_dashboard.py').run()
        self.assertEqual(len(app.exception),0,str(app.exception));self.assertEqual(len(app.error),0,str(app.error))
        self.assertEqual(app.metric[0].value,'4');self.assertEqual(app.metric[1].value,'2')
        self.assertEqual(len(app.get('plotly_chart')),8)
        self.assertFalse(any('auc' in metric.label.lower() for metric in app.metric))
        self.assertTrue(any('vraie classe' in item.value for item in app.info))
        self.assertEqual(LOG.read_bytes(),before)

    def test_filters_kpis_figures_and_csv_use_exact_same_perimeter(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'history.csv';fixtures(path)
            app=page(path).run()
            self.assertEqual(len(app.exception),0,str(app.exception))
            app.multiselect[0].set_value(['A'])
            app.multiselect[2].set_value(['ACCEPT'])
            captured=[]
            def export(df):
                captured.append(df.copy());return export_history_csv(df)
            with patch('utils.client_dashboard_data.export_history_csv',side_effect=export):
                app.run()
            self.assertEqual(len(app.exception),0,str(app.exception))
            self.assertEqual(app.metric[0].value,'1');self.assertEqual(app.metric[2].value,'90.00%')
            self.assertEqual(app.metric[4].value,'100.00%')
            self.assertEqual(captured[0]['_record_number'].tolist(),[2])
            self.assertEqual(len(app.dataframe[0].value),1)
            app.checkbox[2].set_value(True).run()
            app.slider[0].set_value(.95).run()
            self.assertEqual(app.metric[4].value,'100.00%');self.assertEqual(app.metric[5].value,'100.00%')
            self.assertEqual(app.metric[6].value,'0.00%')
            self.assertEqual(app.dataframe[0].value.iloc[0]['decision'],'ACCEPT')

    def test_missing_values_filter_and_no_matches_show_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'history.csv';fixtures(path)
            app=page(path).run()
            app.multiselect[2].set_value([None]).run()
            self.assertEqual(len(app.exception),0,str(app.exception))
            self.assertEqual(app.metric[0].value,'2');self.assertEqual(app.metric[4].value,'Indisponible')
            app.multiselect[0].set_value(['A']).run()
            self.assertEqual(len(app.exception),0,str(app.exception))
            self.assertEqual(app.metric[0].value,'0');self.assertEqual(app.metric[2].value,'Indisponible')
            self.assertEqual(len(app.get('plotly_chart')),0)

    def test_dates_range_in_progress_and_undated_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'history.csv';fixtures(path)
            app=page(path).run()
            app.checkbox[0].set_value(True).run()
            app.checkbox[2].set_value(False).run() # explicitly exclude undated
            self.assertEqual(len(app.exception),0,str(app.exception))
            self.assertEqual(app.metric[0].value,'5')
            app.date_input[0].set_value(pd.Timestamp('2026-03-01').date())
            app.date_input[1].set_value(pd.Timestamp('2026-02-01').date()).run()
            self.assertTrue(any('date de' in item.value for item in app.warning))
            self.assertEqual(len(app.exception),0,str(app.exception))

    def test_empty_or_corrupt_history_is_explicit_and_never_creates_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'absent.csv'
            app=page(path).run()
            self.assertEqual(len(app.exception),0,str(app.exception));self.assertFalse(path.exists())
            self.assertTrue(any('Aucun dossier' in item.value for item in app.info))
            path.write_text(','.join(COLUMNS)+'\n"broken',encoding='utf-8')
            app=page(path).run()
            self.assertEqual(len(app.exception),0,str(app.exception))
            self.assertTrue(any('Lecture du journal impossible' in item.value for item in app.error))

    def test_fixed_auc_removed_from_home_code_and_return_values(self):
        source=(ROOT/'app/home.py').read_text(encoding='utf-8')
        self.assertNotIn('auc_proxy',source)
        module=runpy.run_path(str(ROOT/'app/home.py'),run_name='home_auc_test')
        result=module['kpis_from_df'](pd.DataFrame())
        self.assertFalse(any('auc' in name.lower() for name in result))
