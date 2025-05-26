import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor


def augment_data(area, model, partition):
    gen = pd.read_csv(f'generated_data/{area}/{model}/{model}_{area}_wk5.csv', parse_dates=['datetime'])
    real = pd.read_csv(f'raw_data/{area}/{area}_wk5.csv', parse_dates=['datetime'])
    if partition == '14':
        gen_p = gen.iloc[168:672]
        raw_p = real.iloc[:168]
        aug_set = raw_p.append(gen_p).set_index('datetime', drop=True)
    if partition == '41':
        gen_p = gen.iloc[672:]
        raw_p = real.iloc[:672]
        aug_set = raw_p.append(gen_p).set_index('datetime', drop=True)
    if partition == '23':
        gen_p = gen.iloc[336:]
        raw_p = real.iloc[:336]
        aug_set = raw_p.append(gen_p).set_index('datetime', drop=True)
    if partition == '32':
        gen_p = gen.iloc[:336]
        raw_p = real.iloc[336:]
        aug_set = raw_p.append(gen_p).set_index('datetime', drop=True)

    return aug_set


def mean_absolute_percentage_error(y_true, y_pred):
    y_true = np.array(y_true).reshape(-1, 1)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def plot_lag_corr(y_test, y_pred, num_lags):
    """Calculates & plots Lag Correlation
    Code is same as tsextract except for plotting"""
    lag_coffs = []
    for c in range(num_lags):
        lagged = pd.Series(y_pred).shift(c)
        lag_coffs.append(spearmanr(lagged, y_test, nan_policy='omit')[0])
    plt.figure()
    plt.plot(list(range(num_lags)), lag_coffs), plt.title('Lag Correlation Plot')
    plt.xlabel('No. of lags'), plt.ylabel('Correlation')
    plt.show()


def residual_plots(test, pred, zeroline):
    residuals = test - pred
    plt.scatter(pred, residuals)
    plt.plot(pred, zeroline)
    plt.xlabel('predicted values'), plt.ylabel('residuals'), plt.title('Residual Plot')
    plt.show()


def real_vs_pred(y_true, y_pred, time_idx):
    """Plot actual vs predicted line plots
    """
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    datum = pd.DataFrame({
        "Actual": y_true,
        "Prediction": y_pred
    }, index=time_idx)
    plt.figure()
    plt.plot(datum.index, datum.Actual, label='Real')
    plt.plot(datum.index, datum.Prediction, label='Predicted')
    plt.xlabel('Dates'), plt.ylabel('Number of connections')
    plt.legend()
    plt.show()


def data_prep(data):
    data['t-1'] = data.internet.shift(1)
    data['t+1'] = data.internet.shift(-1)
    data['f_diff'] = data.internet.diff()
    data['s_diff'] = data.f_diff.diff()
    data['Hours'] = data.index.hour
    data.rename({'internet': 't0'}, inplace=True, axis=1)
    data['target_diff'] = data['t+1'] - data['t0']
    data.dropna(inplace=True)
    pres_val = data['t0']
    data.drop(['t0', 't+1'], axis=1, inplace=True)

    return data, pres_val


def prep_testdata(area):
    test_series = pd.read_csv(f'raw_data/full/testdata_{area}.csv',
                              index_col='datetime', parse_dates=True)
    test_data, test_t0 = data_prep(test_series)
    X_test = np.array(test_data[test_data.columns.values[:-1]])
    y_test = np.array(test_data[test_data.columns.values[-1]]
                      ).reshape(-1, 1) + np.array(test_t0).reshape(-1, 1)
    return X_test, y_test, test_t0


def train_model(train, params):
    train_data, _ = data_prep(train)
    X_train = np.array(train_data[train_data.columns.values[:-1]])
    y_train = np.array(train_data[train_data.columns.values[-1]]).reshape(-1, 1)
    reg = GradientBoostingRegressor(**params)
    reg.fit(X_train, y_train)
    return reg


def predictor(X_test, test_t0, y_test, model):
    y_pred = model.predict(X_test).reshape(-1, 1)
    y_pred = y_pred + np.array(test_t0).reshape(-1, 1)
    mae = mean_absolute_percentage_error(y_test, y_pred)
    print(f'Mean absolute percentage error: {mae}')
    return mae, y_pred


def run_tstr(model_list, wk_list, area, model_params):
    results = dict()
    X_test, y_test, test_t0 = prep_testdata(area)
    for j in model_list:
        mae_list = list()
        for k in wk_list:
            train_series = pd.read_csv(f'generated_data/{area}/{j}/{j}_{area}_{k}.csv',
                                       index_col='datetime', parse_dates=True)
            reg = train_model(train_series, model_params)
            mae, _ = predictor(X_test, test_t0, y_test, reg)
            mae_list.append(mae)
        results[j] = mae_list
    return results


def run_trtr(wk_list, area, model_params):
    X_test, y_test, test_t0 = prep_testdata(area)
    mae_list = list()
    for k in wk_list:
        train_series = pd.read_csv(f'raw_data/{area}/{area}_{k}.csv',
                                   index_col='datetime', parse_dates=True)
        reg = train_model(train_series, model_params)
        mae, _ = predictor(X_test, test_t0, y_test, reg)
        mae_list.append(mae)
    return mae_list


def run_aug(model_list, area, model_params, partition_list):
    results = dict()
    X_test, y_test, test_t0 = prep_testdata(area)
    for j in model_list:
        mae_list = list()
        for partition in partition_list:
            aug_series = augment_data(area, j, partition)
            reg = train_model(aug_series, model_params)
            mae, _ = predictor(X_test, test_t0, y_test, reg)
            mae_list.append(mae)
        results[j] = mae_list
    return results

# train_series = pd.read_3csv('generated_data/4456/DG/DG_4456_wk1.csv', index_col='datetime', parse_dates=True)
# test_series = pd.read_csv('raw_data/full/testdata_4456.csv', index_col='datetime', parse_dates=True)


if __name__ == '__main__':
    params = {'n_estimators': 100,
              'max_depth': 12,
              'min_samples_split': 32,
              'learning_rate': 0.05,
              'loss': 'squared_error'
              }
    models = ['DG', 'tgan', 'par']
    wks = ['wk1', 'wk3', 'wk5']
    partitions = ['14', '41', '23', '32']
    cell = 5060
    res = run_tstr(models, wks, cell, params)
    res2 = run_trtr(wks, cell, params)
    res['raw'] = res2
    result_file = pd.DataFrame(res, index=['1 Week', '3 Weeks', '5 Weeks'])
    result_file.to_csv(f'results/tstr/tstr_{cell}.csv')

    # aug1 = run_aug(models, 4259, params, partitions)
    # result_file1 = pd.DataFrame(aug1, index=['1+4', '4+1', '2+3', '3+2'])
    # result_file1.to_csv('results/augmented/aug_4259.csv')
    #
    # aug2 = run_aug(models, 4456, params, partitions)
    # result_file2 = pd.DataFrame(aug2, index=['1+4', '4+1', '2+3', '3+2'])
    # result_file2.to_csv('results/augmented/aug_4456.csv')
    #
    # aug3 = run_aug(models, 5060, params, partitions)
    # result_file2 = pd.DataFrame(aug3, index=['1+4', '4+1', '2+3', '3+2'])
    # result_file2.to_csv('results/augmented/aug_5060.csv')
    print(-1)
