import os
os.chdir('/Users/gavin.wu/Python/Project/Marketing/FromGavin/Python_Scorer')

import json
import codecs
import numpy as np
import pandas as pd
import xgboost as xgb
from scorefunc import score

test = pd.read_csv('/Users/gavin.wu/Python/Project/Marketing/marketing_test.csv', dtype=object)
scorer = score()
tt = test
ypred = scorer.get_score(tt)
print(ypred)
