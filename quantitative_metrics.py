import matplotlib.pyplot as plt
from dtw import *
import pandas as pd
import numpy as np
from scipy.stats import entropy
from random import sample


def calculate_dtw(parts, cellid, model):
    dtw_vals = list()
    for i in parts:
        real_data = pd.read_csv(f'raw_data/{cellid}/{cellid}_{i}.csv')
        gen_data = pd.read_csv(f'generated_data/{cellid}/{model}/{model}_{cellid}_{i}.csv')
        alignment = dtw(gen_data.internet, real_data.internet, keep_internals=True)
        # alignment.plot(type='threeway')
        dist = alignment.normalizedDistance
        dtw_vals.append(dist)
    return dtw_vals


def MinMaxScalar(sequence):
    minimum = np.min(sequence.values, 0)
    maximum = np.max(sequence.values, 0)
    norm_seq = (sequence - minimum)/(maximum - minimum)
    return norm_seq


def calculate_kl(parts, cellid, model):
    sample_size = [200, 400, 600]
    kl_vals = list()
    for i, s in zip(parts, sample_size):
        real_data = pd.read_csv(f'raw_data/{cellid}/{cellid}_{i}.csv', index_col='datetime',
                                parse_dates=['datetime'])
        gen_data = pd.read_csv(f'generated_data/{cellid}/{model}/{model}_{cellid}_{i}.csv', index_col='datetime',
                               parse_dates=['datetime'])
        rdata = real_data.diff().dropna().reset_index(drop=True)
        gdata = gen_data.diff().dropna().reset_index(drop=True)
        g_samples = sample(list(gdata.index.values), s)
        r_samples = sample(list(rdata.index.values), s)
        gen_samples = gdata.iloc[g_samples].internet
        norm_gen = MinMaxScalar(gen_samples)
        real_samples = rdata.iloc[r_samples].internet
        norm_real = MinMaxScalar(real_samples)

        KL = entropy(norm_real.values, norm_gen.values)
        kl_vals.append(KL)

    return kl_vals


def cal_dtw(area, wk_list, model_list):
    results = {}
    for i in model_list:
        wk_res = []
        for j in wk_list:
            real_data = pd.read_csv(f'raw_data/{area}/{area}_{j}.csv', index_col='datetime',
                                    parse_dates=True)
            gen_data = pd.read_csv(f'generated_data/{area}/{i}/{i}_{area}_{j}.csv', index_col='datetime',
                                   parse_dates=True)

            alignment = dtw(gen_data.internet, real_data.internet, keep_internals=True)
            # alignment.plot(type='threeway')
            dist = alignment.normalizedDistance
            wk_res.append(dist)
        results[i] = wk_res
    return results


def cal_kl(area, wk_list, model_list):
    results = {}
    for i in model_list:
        wk_res = []
        for j in wk_list:
            real_data = pd.read_csv(f'raw_data/{area}/{area}_{j}.csv', index_col='datetime',
                                    parse_dates=True)
            gen_data = pd.read_csv(f'generated_data/{area}/{i}/{i}_{area}_{j}.csv', index_col='datetime',
                                   parse_dates=True)
            # stat_real = real_data.diff().dropna()
            # stat_gen = gen_data.diff().dropna()
            stat_bin_real = pd.cut(real_data.internet, bins=10).value_counts()
            stat_bin_gen = pd.cut(gen_data.internet, bins=10).value_counts()
            kl = entropy(stat_bin_real.values, stat_bin_gen.values)
            wk_res.append(kl)
        results[i] = wk_res
    return results


def plot_stackedbars(y1, y2, y3, title, ylabel, p_label):
    x = np.arange(len(p_label))  # the label locations
    width = 0.2

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width, y1, width, label='DG')
    rects2 = ax.bar(x + width, y2, width, label='TimeGAN')
    rects3 = ax.bar(x, y3, width, label='PAR')

    ax.set_ylabel(ylabel, size=14)
    ax.set_title(title, size=16)
    ax.set_xticks(x)
    ax.set_xticklabels(p_label, size=14)

    ax.legend()
    plt.show()


if __name__ == "__main__":
    p = ['wk3', 'wk5', 'wk7']
    m = ['DG', 'tgan', 'par']
    cell = 5060
    p_label = ['3 weeks', '5 weeks', '7 weeks']
    res_dtw = cal_dtw(cell, p, m)
    res_kl = cal_kl(cell, p, m)
    result_dtw = pd.DataFrame(res_dtw, index=p_label)
    result_kl = pd.DataFrame(res_kl, index=p_label)
    result_dtw.to_csv(f'results/quantitative_metrics/dtw_{cell}.csv')
    result_kl.to_csv(f'results/quantitative_metrics/kl_{cell}.csv')
    print(-1)





