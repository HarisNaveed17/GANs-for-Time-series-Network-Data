import pandas as pd
from random import sample
import random
from forecasting import train_model, predictor, prep_testdata


def add_missing(data, missing_rate):
    random.seed(10)
    num_drop = int(missing_rate * len(data))
    data_index = data.index.values.tolist()
    samples_drop = sample(data_index, num_drop)
    samples = [pd.to_datetime(i) for i in samples_drop]
    data_miss = data.drop(samples, axis=0)
    return data_miss


def impute(data, gen_data):
    date_range = pd.date_range('2013-11-4', '2013-12-08 23:00:00', freq='H')
    filled_data = data.reindex(date_range)
    filldat = filled_data.copy()
    filldat[filldat.isnull()] = gen_data
    filldat.dropna(inplace=True)
    return filldat


def interpolator(data):
    data1 = data.copy()
    # data1['datetime'] = pd.to_datetime(data1['datetime'])
    # data1.set_index('datetime', inplace=True, drop=True)
    date_range = pd.date_range('2013-11-4', '2013-12-08 23:00:00', freq='H')
    interp = data1.reindex(date_range)
    interp = interp.interpolate(method='quadratic')
    return interp


def run_imputation_utility(missing_rates, model_list, area, model_params):
    ori_data = pd.read_csv(f'raw_data/{area}/{area}_wk5.csv', index_col='datetime',
                           parse_dates=True)
    X_test, y_test, test_t0 = prep_testdata(area)
    results = {}
    for miss_rate in missing_rates:
        missing_data = add_missing(ori_data, miss_rate)
        temp_res = []
        for model in model_list:
            gen_data = pd.read_csv(f'generated_data/{area}/{model}/{model}_{area}_wk5.csv',
                                   index_col='datetime', parse_dates=True)
            filled_data = impute(missing_data, gen_data)
            filled_model = train_model(filled_data, model_params)
            fill_mae, _ = predictor(X_test, test_t0, y_test, filled_model)
            temp_res.append(fill_mae)
        interp_data = interpolator(missing_data)
        miss_model = train_model(missing_data, model_params)
        interp_model = train_model(interp_data, model_params)

        miss_mae, _ = predictor(X_test, test_t0, y_test, miss_model)
        inter_mae, _ = predictor(X_test, test_t0, y_test, interp_model)

        temp_res.extend([miss_mae, inter_mae])
        results[miss_rate] = temp_res

    return results


if __name__ == '__main__':
    miss_rates = [0.2, 0.4, 0.6, 0.8, 0.9]
    models = ['DG', 'tgan', 'par']
    params = {'n_estimators': 100,
              'max_depth': 12,
              'min_samples_split': 32,
              'learning_rate': 0.05,
              'loss': 'squared_error',
              }
    res = run_imputation_utility(miss_rates, models, 5060, params)
    result_file = pd.DataFrame(res, index=['DG', 'tgan', 'par', 'missing', 'interpolated']).T
    result_file.to_csv('results/test/imputation/imputation_5060.csv')
    print(-1)
