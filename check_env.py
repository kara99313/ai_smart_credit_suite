import sys
import torch
import numpy as np
import pandas as pd
import sklearn
import transformers
import tokenizers
import sentence_transformers
import faiss

print("Python", sys.version)
print("torch", torch.__version__)
print("numpy", np.__version__, "| pandas", pd.__version__, "| sklearn", sklearn.__version__)
print("transformers", transformers.__version__, "| tokenizers", tokenizers.__version__)
print("sentence-transformers", sentence_transformers.__version__)
print("faiss OK:", hasattr(faiss, "IndexFlatL2"))
