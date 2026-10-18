from pathlib import Path
import csv, hashlib, tempfile, unittest
from unittest.mock import patch
from utils.prediction_logs import COLUMNS

# Invented records only: no source data or private-history fingerprint.
_TEMP = tempfile.TemporaryDirectory(prefix='credit-synthetic-')
LOG = Path(_TEMP.name) / 'history.csv'
with LOG.open('w', encoding='utf-8', newline='') as stream:
    writer = csv.writer(stream)
    writer.writerow(COLUMNS)
    for i in range(4):
        writer.writerow([f'2026-01-0{i+1}T12:00:00', f'SYNTHETIC-{i%2}',
                         (i+1)/10, 0, .1, '{"HasMortgage": true}',
                         'synthetic-model', 'BB', 'ACCEPT' if i%2 else 'REVIEW/REJECT'])
BASELINE_HASH = hashlib.sha256(LOG.read_bytes()).hexdigest()

class SyntheticHistoryTestCase(unittest.TestCase):
    """Redirect only the application's default CSV; explicit test paths stay real."""
    def setUp(self):
        super().setUp()
        from utils import prediction_logs
        original = prediction_logs._raw_records
        default = (Path.cwd() / 'data/predictions_log.csv').resolve()
        def read(path):
            return original(LOG if Path(path).resolve() == default else path)
        self._history_patch = patch.object(prediction_logs, '_raw_records', side_effect=read)
        self._history_patch.start()
        self.addCleanup(self._history_patch.stop)
