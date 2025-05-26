import pandas as pd
import matplotlib.pyplot as plt
from forecasting import prep_testdata, predictor, train_model

params = {'n_estimators': 100,
          'max_depth': 12,
          'min_samples_split': 32,
          'learning_rate': 0.05,
          'loss': 'ls',
          }


gdata = pd.read_csv('generated_data/4456/tgan/tgan_4456_wk5.csv', index_col='datetime', parse_dates=True)
reg = train_model(gdata, params)
X_test, y_test, test_t0 = prep_testdata(4456)
mae, y_p = predictor(X_test, test_t0, y_test, reg)
print(-1)
