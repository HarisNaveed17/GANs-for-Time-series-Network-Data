import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def persistence_model(x):
    return x


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


data = pd.read_csv('raw_data/4259/4259_wk5.csv',
                   usecols=[1], names=['t0'], header=0)
data['t+1'] = data.t0.shift(periods=-2)
data.dropna(inplace=True)

train_size = int(len(data) * 0.85)
train_dat = data.iloc[:train_size].reset_index(drop=True)
test_dat = data.iloc[train_size:].reset_index(drop=True)
train_X, train_y = train_dat['t0'], train_dat['t+1']
test_X, test_y = test_dat['t0'], test_dat['t+1']

pred_y = list()
for i in test_X:
    pred_y.append(persistence_model(i))
mae = mean_absolute_percentage_error(test_y, pred_y)
print(f'Mean absolute percentage error on persistence model is: {mae}')

plt.plot(test_y, label='True values')
plt.plot(pred_y, label='Predicted values (previous timestep values)')
plt.legend()
plt.title(f'Persistence model, 4259, 5 weeks (MAE={mae})')
plt.show()
