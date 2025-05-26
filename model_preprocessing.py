import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Some utility functions to prepare initially preprocessed data for use with various generative models
# especially those used in our research work (TimeGAN, DoppelGANger, PAR)


def MinMaxScaler(data):
    min_val = min(data)
    max_val = max(data)
    numerator = data - min_val
    denominator = max_val - min_val
    norm_data = numerator / (denominator + 1e-7)
    return norm_data, min_val, max_val


def timegan_preprocessing(file, partitions, start_date=None, end_date=None):
    data = pd.read_csv(file, parse_dates=['datetime'])
    data.datetime = data.datetime.dt.floor('H')
    grped_data = data.groupby('datetime').mean()
    if start_date is not None and end_date is not None:
        grped_data = grped_data[start_date:end_date]

    grped_data['Hours'] = grped_data.index.hour
    grped_data['dayofyear'] = grped_data.index.dayofyear

    start = grped_data.index.values[0]
    filename = file.split('.')[0]
    for partition in partitions:
        end = pd.Timedelta(partition) - pd.Timedelta('1 hour')
        end = start + end
        cut_data = grped_data[start:end]
        cut_data.reset_index(drop=True, inplace=True)
        cut_data.to_csv(f'{filename}_{partition}.csv', index=False)

    return grped_data


def timegan_postprocessing(file, cell, time):
    df = pd.read_csv(file, index_col=False)
    df[['Hours', 'dayofyear']] = df[['Hours', 'dayofyear']].apply(np.rint).astype(int)
    df.reset_index(drop=True, inplace=True)
    df['datetime'] = [pd.to_datetime(f"2013{j[0]} {j[1]}:00:00", format="%Y%j %H:%M:%S") for j in
                      zip(df['dayofyear'], df['Hours'])]
    df = df.sort_values('datetime').drop(['Hours', 'dayofyear', 'Unnamed: 0'], axis=1)
    data_t = df.groupby('datetime').mean()
    new_name = f'tgan_{cell}_{time}.csv'
    data_t.to_csv(f'generated_data/{cell}/tgan/{new_name}')
    return data_t


def dg_preprocessing(file, weeks, start, end, cell):
    internet = pd.read_csv(file, parse_dates=['datetime'], index_col='datetime')
    internet = internet[start:end]
    internet['week'] = internet.index.week
    internet.internet, _, _ = MinMaxScaler(internet.internet)
    internet.week, _, _ = MinMaxScaler(internet.week)
    internet.reset_index(drop=True, inplace=True)
    data_attribute = np.unique(internet[['week']].to_numpy(), axis=0)
    data_feature = internet.internet.to_numpy().reshape((weeks, 168, 1))
    data_gen_flag = np.ones((weeks, 168))
    np.savez(f'Data/{cell}_{weeks}.npz', data_feature=data_feature,
             data_attribute=data_attribute, data_gen_flag=data_gen_flag)

    return data_attribute, data_feature, data_gen_flag


def dg_postprocessing(file, weeks, start, end, cell):
    internet = pd.read_csv(file, parse_dates=['datetime'], index_col='datetime')
    internet = internet[start:end]
    internet['week'] = internet.index.week
    min_i = min(internet.internet)
    max_i = max(internet.internet)
    min_w = min(internet.week)
    max_w = max(internet.week)
    gen_data = np.load(f'{cell}_{weeks}_s12_2000e_2.npz')
    gen_attributes = gen_data['data_attribute']
    gen_features = gen_data['data_feature']
    final = np.repeat(gen_attributes, repeats=168, axis=0)
    final = pd.DataFrame(final, columns=['week'])
    final['week'] = ((final['week'] * (max_w - min_w)) + min_w).apply(np.rint).astype(int)
    hours = np.arange(0, 24)
    reps = len(final) // 24
    hrs = np.tile(hours, reps=reps)
    final['Hours'] = hrs
    final['dayofyear'] = 0

    for i in range(0, len(final), 168):
        day = (final.week.iloc[i] - 1) * 7
        for j in range(i, i + 168, 24):
            final.dayofyear.iloc[j:j + 24] = day
            day += 1
    gen_feats = gen_features.reshape((-1, 1))
    final['internet'] = gen_feats
    final['internet'] = abs(final['internet'])
    final['internet'] = (final['internet'] * (max_i - min_i)) + min_i
    final['datetime'] = [pd.to_datetime(f"2013{j[0]} {j[1]}:00:00", format='%Y%j %H:%M:%S') for j in
                         zip(final['dayofyear'], final['Hours'])]
    final = final.groupby('datetime').mean()
    final.drop(['week', 'dayofyear', 'Hours'], axis=1, inplace=True)
    final.to_csv(f'dg_{cell}_{weeks}_s12_e2000_2.csv')
    return final


def visual_check(true_file, gen_file, start, end):
    real_dat = pd.read_csv(true_file, index_col=['datetime'], parse_dates=['datetime'])
    gen_dat = pd.read_csv(gen_file, index_col=['datetime'], parse_dates=['datetime'])

    real_dat = real_dat[start:end]

    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    ax1.plot(gen_dat.index, gen_dat['internet'])
    ax2.plot(real_dat.index, real_dat['internet'])
    fig.autofmt_xdate(ha='center', rotation=30)
    fig.suptitle('Generated vs Real (Navigli), 7 Weeks', fontsize=18)
    plt.xlabel('Date time', fontsize=16)
    ax1.set_title('Generated', fontsize=13)
    ax2.set_title('Real', fontsize=13)
    ax1.set_ylabel('Number of connections', fontsize=14)
    ax2.set_ylabel('Number of connections', fontsize=14)
    fig.text(0.02, 0.5, 'Number of connections', va='center', rotation='vertical', fontsize=14)
    # plt.savefig('44565wksdatefeats.png', transparent=False, orientation='landscape')
    plt.show()


def visualize_one(file, name):
    if not isinstance(file, pd.DataFrame):
        file = pd.read_csv(file, parse_dates=['datetime'], usecols=['datetime', 'internet'])
    fig = plt.figure()
    ax = plt.axes()
    ax.plot(file.datetime, file.internet)
    fig.autofmt_xdate(ha='center', rotation=30)
    ax.set_xlabel('Date time', fontsize=12)
    ax.set_ylabel('Number of connections')
    plt.savefig(f'{name}.png')
    plt.show()


if __name__ == "__main__":
    # gen_final = dg_postprocessing('5060_wk3.csv', 3, '2013-11-04', '2013-11-24', 5060)
    # visualize_one('2000e_s4_5060/dg_5060_3_s4_norm_T_e2000.csv')
    tdata = timegan_postprocessing('generated_data/5060/tgan/tgan_5060_wk7.csv', 5060, 'wk7')
    print(-1)

# As we increase the amount of data, the shape of the data converges to the model's underlying trend (for 5060's
# timegan data)
