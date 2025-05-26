import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import pandas as pd
import matplotlib.patches as mpatches


px = 1/plt.rcParams['figure.dpi']


def plot_acf_subplots(parts, cellid, models, figtitle, lagnum):
    fig, figaxes = plt.subplots(3, 3, figsize=(1200*px, 1100*px))
    fig.suptitle(figtitle, size=16)
    col = 0
    corr_r = []
    corr_g = []
    lags = [i for i in range(lagnum)]
    wks = ['(a)\n 3 Weeks', '(b)\n 5 Weeks', '(c)\n 7 Weeks']
    col_labels = [model for model in models]
    for model in models:
        for fignum, i in enumerate(parts):
            real_data = pd.read_csv(f'raw_data/{cellid}/{cellid}_{i}.csv')
            gen_data = pd.read_csv(f'generated_data/{cellid}/{model}/{model}_{cellid}_{i}.csv')
            xr = real_data.internet.values
            xg = gen_data.internet.values
            for j in lags:
                yr = real_data.internet.shift(j).values
                yg = gen_data.internet.shift(j).values
                corr_r.append(spearmanr(xr, yr, nan_policy='omit')[0])
                corr_g.append(spearmanr(xg, yg, nan_policy='omit')[0])
            figaxes[fignum][col].plot(lags, corr_r, label='Real')
            figaxes[fignum][col].plot(lags, corr_g, label='Generated')
            corr_r.clear()
            corr_g.clear()
        col = col + 1
    handles, labels = plt.gca().get_legend_handles_labels()
    for ax, col in zip(figaxes[0], col_labels):
        ax.set_title(col, size=16, fontstyle='normal')

    for ax, row in zip(figaxes[:, 0], wks):
        ax.set_ylabel(row, rotation=0, size=14, labelpad=40, fontstyle='normal')
    fig.legend(handles, labels, prop={'size': 12})
    # plt.savefig(f'results/qualitative_metrics/acf_{cellid}.png', transparent=True)
    plt.show()


def plot_acf(parts, cellid, model, lag_num):
    corr_r = []
    corr_g = []
    lags = [i for i in range(lag_num)]
    for i in parts:
        real_data = pd.read_csv(f'raw_data/{cellid}/{cellid}_{i}.csv')
        gen_data = pd.read_csv(f'generated_data/{cellid}/{model}/{model}_{cellid}_{i}.csv')
        xr = real_data.internet.values
        xg = gen_data.internet.values
        for j in lags:
            yr = real_data.internet.shift(j).values
            yg = gen_data.internet.shift(j).values
            corr_r.append(spearmanr(xr, yr, nan_policy='omit')[0])
            corr_g.append(spearmanr(xg, yg, nan_policy='omit')[0])
        plt.plot(lags, corr_r, label='Real')
        plt.plot(lags, corr_g, label='Generated')
        plt.show()
        corr_r.clear()
        corr_g.clear()


def plot_histogram_subplots(parts, cellid, models, figtitle):
    fig, figaxes = plt.subplots(3, 3, figsize=(1200*px, 900*px))
    fig.suptitle(figtitle, size=16)
    col = 0
    wks = ['(a)\n 3 Weeks', '(b)\n 5 Weeks', '(c)\n 7 Weeks']
    col_labels = [model for model in models]
    for model in models:
        for fignum, i in enumerate(parts):
            real_data = pd.read_csv(f'raw_data/{cellid}/{cellid}_{i}.csv')
            gen_data = pd.read_csv(f'generated_data/{cellid}/{model}/{model}_{cellid}_{i}.csv')
            figaxes[fignum][col].hist(real_data.internet, bins=50, label='Real', alpha=0.6)
            figaxes[fignum][col].hist(gen_data.internet, bins=50, label='Generated', alpha=0.6)
        col = col + 1
    handles, labels = plt.gca().get_legend_handles_labels()
    for ax, col in zip(figaxes[0], col_labels):
        ax.set_title(col, size=16, fontstyle='normal')

    for ax, row in zip(figaxes[:, 0], wks):
        ax.set_ylabel(row, rotation=0, size=14, labelpad=40, fontstyle='normal')
    fig.legend(handles, labels, prop={'size': 12})
    # plt.savefig(f'results/qualitative_metrics/histograms_{cellid}.png', transparent=True)


def plot_histograms(parts, cellid, model):
    for i in parts:
        real_data = pd.read_csv(f'raw_data/{cellid}/{cellid}_{i}.csv')
        gen_data = pd.read_csv(f'generated_data/{cellid}/{model}/{model}_{cellid}_{i}.csv')
        plt.hist(real_data.internet, bins=50, label='Real')
        plt.hist(gen_data.internet, bins=50, label='Generated')
        blue_patch = mpatches.Patch(color='tab:blue', label='Real')
        orange_patch = mpatches.Patch(color='tab:orange', label='Generated')
        plt.title(f'{model}, {cellid}, {i}', size=16)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.xlabel('No. of connections', size=14)
        plt.ylabel('Frequency', size=14)
        plt.legend(handles=[blue_patch, orange_patch], prop={'size': 12})
        plt.show()


if __name__ == "__main__":
    p = ['wk3', 'wk5', 'wk7']
    # p_label = ['3 weeks', '5 weeks', '7 weeks']
    # Bocconi University, Navigli District, Duomo Cathedral
    plot_histogram_subplots(p, 5060, ['DG', 'par', 'tgan'], 'Duomo Cathedral')
    plot_acf_subplots(p, 5060, ['DG', 'par', 'tgan'], 'Duomo Cathedral', 168)
    plt.show()
