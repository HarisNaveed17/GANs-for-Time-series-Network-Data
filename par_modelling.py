from sdv.sequential import PARSynthesizer
import pandas as pd
from model_preprocessing import visualize_one


def train_par(cell, weeks, epochs, sequence_index='datetime', segment_size=None):
    train_data = pd.read_csv(f'raw_data/{cell}/{cell}_wk{weeks}.csv', parse_dates=['datetime'])
    model = PARSynthesizer(sequence_index=sequence_index, epochs=epochs, verbose=True, segment_size=segment_size)
    model.fit(train_data)
    model.save(f'{cell}_{weeks}.pkl')
    return model


def generate_par(model, cell, weeks, samples, name):
    new_data = model.sample(samples)
    visualize_one(new_data, name=name)
    new_data.to_csv(f'generated_data/{cell}/par/par_{cell}_wkk{weeks}.csv')
    return new_data


if __name__ == '__main__':
    model = train_par(5060, 7, 1000)
    #model = PAR.load('5060_1.pkl')
    new_data = generate_par(model, 4456, 7, 1, name=f'generated_fig_wk7_2')