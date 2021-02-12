import numpy as np, pandas as pd
import os, argparse, json
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import savgol_filter
import tensorflow as tf
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
from collections import defaultdict

#python match_peaks.py -r results/Test/Test_ff_model_predictions

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, help='The json configuration file for the aedes model.')
    parser.add_argument('-t', '--threshold', type = float, help = 'The threshold for declaring a peak.')
    parser.add_argument('-w', '--width', type = int, default = 5, help = 'The width for declaring a peak.')
    parser.add_argument('-r', '--results', type = str, default = '', help = 'The file for results that have already been generated.')
    parser.add_argument('--county', action='store_true', help = 'Whether or not to run a city/county specific model.')
    return parser.parse_args()

def load_val_data():
    training_data = pd.read_pickle('~/Documents/Projects/aedes_model/Data/train_data.pd')
    validation = pd.read_pickle(os.path.expanduser("~/Documents/Projects/aedes_model/Data/val_data.pd"))
    scaler = MinMaxScaler()
    scaler.fit(training_data.iloc[:, -5:])
    validation.columns = range(0, len(validation.columns))
    groups = validation.groupby(by = 0)
    data = {}
    data_shape = [90, 4]
    for city, subset in groups:
        sub_data = []
        for i in range(0, len(subset) - (data_shape[0] + 1)):
            sub_data.append(scaler.transform(subset.iloc[i: i + data_shape[0], -(data_shape[1] + 1):].values))
        data[city+',2019'] = np.array(sub_data[:365 - data_shape[0]])
        data[city+',2020'] = np.array(sub_data[365 - data_shape[0]:])
    return data

def load_results_data(filename, cities = None):
    results = pd.read_csv(filename)
    groups = results.groupby(by = 'County')
    output = {}
    for city, subset in groups:
        if cities is None or city in cities:
            for year in range(2011, 2021):
                label = city + ',' + str(year)
                y_pred = subset[subset['Year'] == year]['Neural Network']
                y_true = subset[subset['Year'] == year]['MoLS']
                output[label] = (y_pred, y_true)
    return output

def r2_keras(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return (1 - SS_res/(SS_tot + K.epsilon()))

def load_model(mfile):
    model_file = os.path.expanduser(mfile)
    if os.path.exists(model_file):
        model = tf.keras.models.load_model(model_file, custom_objects = {'r2_keras': r2_keras})
        print('MODEL LOADED')
        return model
    else:
        raise FileNotFoundError(f"Model does not exist at {mfile}")

def run_model(model, validation):
    output = {}
    for city, data in validation.items():
        X, y_true = data[:, :, :-1], data[:, -1, -1]
        y_pred = model.predict(X).flatten()
        output[city] = (y_pred, y_true)
    return output

def smooth_data(data, rounds = 1, max_val = 1):
    for _ in range(rounds):
        data = savgol_filter(data, 11, 3)
        data[data < 0] = 0
    return (data / data.max()) * max_val

def peak_finder(array, threshold = 0.8, peak_width = 5):
    peaks = set()
    above_threshold = array > threshold
    peak_start = 0 
    peaking = False
    for i, h in enumerate(above_threshold):
        if h and not peaking:
            peak_start = i
            peaking = True
        elif not h and peaking:
            if i - peak_start > peak_width:
                peaks.add((peak_start, i))
            peaking = False
        else:
            continue
    return peaks

def min_offset(peaks_pred, peaks_true):
    num_peaks_pred = len(peaks_pred)
    num_peaks_true = len(peaks_true)
    offsets = []
    widths = []
    if num_peaks_true and num_peaks_pred:
        for peak in peaks_true:
            mind = np.inf
            mino = None
            for pred in peaks_pred:
                offset = ((pred[0] - peak[0]), (pred[1] - peak[1]))
                distance = abs(offset[0]) + abs(offset[1])
                if distance < mind:
                    mind = distance
                    mino = offset
            widths.append(peak[1] - peak[0])
            offsets.append(mino)
    return {'True Peaks': num_peaks_true, 'Predicted Peaks': num_peaks_pred, 'Offsets': offsets, 'Width': widths}

def compare_peaks(output, metric, threshold=0.8, peak_width = 5, scale_to_1 = False, smooth = True):
    results = {}
    for city, (y_pred, y_true) in output.items():
        y_pred = smooth_data(y_pred, rounds = 2 * smooth, max_val = 1 if scale_to_1 else y_pred.max())
        y_true = smooth_data(y_true, rounds = 0, max_val = 1 if scale_to_1 else y_true.max())
        threshold_scaled = threshold * (1 if scale_to_1 else y_true.max())
        peaks_pred = peak_finder(y_pred, threshold=threshold_scaled, peak_width=peak_width)
        peaks_true = peak_finder(y_true, threshold=threshold_scaled, peak_width=peak_width)
        results[city] = metric(peaks_pred, peaks_true)
    return results

def to_latex(mean_table, stddev_table, model_name):
    rstring = ''
    for i in range(2):
        if i == 0:
            rstring += '\\multirow{2}{*}{' + model_name + ' Model} & Mean $D_{on}$'
        elif i == 1:
            rstring += ' & Mean $D_{off}$'
        for j in range(4):
            rstring += ' & ' + str(round(mean_table.loc['All'].iloc[5 + j*2 + i], 3))
            rstring += ' $\\pm$ ' + str(round(stddev_table.loc['All'].iloc[5 + j*2 + i], 3))
        rstring += ' \\\\\n'
    return rstring + '\\hline\n'
 
def main():
    args = parse_args()

    if args.results:
        if args.county:
            #COUNTIES = ['Avondale,Arizona', 'Ventura,California', 'Butte,California', 'Waukesha,Wisconsin', 'Collier,Florida', 'Cameron,Texas']
            output = load_results_data(args.results)
        else:
            output = load_results_data(args.results)
        smooth = False
    else:
        val_data = load_val_data()
        model = load_model(args.config)

        output = run_model(model, val_data)
        smooth = True

    city_to_state = pd.read_csv('~/Documents/Projects/aedes_model/Data/All_counties.csv')
    
    if args.county:
        indices = sorted(set(map(lambda x: x.split(',')[0], output.keys()))) + ['All']
    else:
        indices = sorted(set(city_to_state['State'])) + ['All']
    
    #indices = indices + [s + '_s' for s in indices]
    table_data = pd.DataFrame(0.0, index = indices, columns=['n', '20% n', '40% n', '60% n', '80% n'] + [f'{i - i % 2}0% Threshold ({"end" if i % 2 else "start"})' for i in range(2, 10)])

    stddev_data = pd.DataFrame(0.0, index = indices, columns=['n', '20% n', '40% n', '60% n', '80% n'] + [f'{i - i % 2}0% Threshold ({"end" if i % 2 else "start"})_stddev' for i in range(2, 10)])

    for scale in [False]: #[True, False]:

        threshold_array = []
        peaks_array = []
        start_array = []
        end_array = []
        season_width = {}

        #Calculate season width
        widths = defaultdict(list)
        results = compare_peaks(output, min_offset, threshold=0.2, peak_width=args.width, scale_to_1=scale, smooth=smooth)
        for city, result in results.items():
            if args.county:
                index = city.split(',')[0] + '_s' * scale
            else:    
                index = city[:-5].split(',')[1] + '_s' * scale
            widths[index].extend(result['Width'])
        
        for idx in widths:
            season_width[idx] = np.mean(widths[idx])

        for threshold in sorted([0.2, 0.4, 0.6, 0.8] + list(np.linspace(0.01, 0.99, 200))):
            results = compare_peaks(output, min_offset, threshold=threshold, peak_width=args.width, scale_to_1=scale, smooth=smooth)

            start_offsets = defaultdict(list)
            end_offsets = defaultdict(list)
            peak_difference = 0
            for city, result in results.items():
                if args.county:
                    index = city.split(',')[0] + '_s' * scale
                else:    
                    index = city[:-5].split(',')[1] + '_s' * scale
                if threshold == 0.2:
                    table_data['n'][index] += 1
                    stddev_data['n'][index] += 1
                    table_data['n']['All' + '_s' * scale] += 1
                    stddev_data['n']['All' + '_s' * scale] += 1

                if result['Offsets']:
                    start, end = zip(*result['Offsets'])
                    start, end = list(map(lambda x: x / season_width[index], start)), list(map(lambda x: x / season_width[index], end))
                    start_offsets[index].extend(start)
                    end_offsets[index].extend(end)
                    start_offsets['all'].extend(start)
                    end_offsets['all'].extend(end)

                peak_difference += result['Predicted Peaks'] - result['True Peaks']

                if threshold in [0.2, 0.4, 0.6, 0.8]:
                    table_data[f'{int(threshold * 100)}% n'][index] += result['Predicted Peaks'] - result['True Peaks']
                    stddev_data[f'{int(threshold * 100)}% n'][index] += result['Predicted Peaks'] - result['True Peaks']
                    table_data[f'{int(threshold * 100)}% n']['All' + '_s' * scale] += result['Predicted Peaks'] - result['True Peaks']
                    stddev_data[f'{int(threshold * 100)}% n']['All' + '_s' * scale] += result['Predicted Peaks'] - result['True Peaks']
                    if ("Avondale" in city or "Collier" in city) and '2020' in city and args.county:
                        with open("tables/avondale_and_collier.txt", 'a') as fp:
                            terms = args.results.split('_')
                            modelname = '_'.join([terms[1]] + terms[3:-1])
                            if result['Offsets']:
                                on, off = zip(*result['Offsets'])
                                diffs = (np.mean(on) / season_width[index], np.mean(off) / season_width[index])
                            else:
                                diffs = (0, 0)
                            print(str(threshold * 100) + '%',  modelname, ':', city, file = fp)
                            print(diffs, file = fp)

            if threshold in [0.2, 0.4, 0.6, 0.8]:
                for index in start_offsets:
                    table_data[f'{int(threshold * 100)}% Threshold (start)'][index] = np.mean(start_offsets[index])
                    table_data[f'{int(threshold * 100)}% Threshold (end)'][index] = np.mean(end_offsets[index])
                    stddev_data[f'{int(threshold * 100)}% Threshold (start)_stddev'][index] = np.std(start_offsets[index])
                    stddev_data[f'{int(threshold * 100)}% Threshold (end)_stddev'][index] = np.std(end_offsets[index])
                table_data[f'{int(threshold * 100)}% Threshold (start)']['All' + '_s' * scale] = np.mean(start_offsets['all'])
                table_data[f'{int(threshold * 100)}% Threshold (end)']['All' + '_s' * scale] = np.mean(end_offsets['all'])
                stddev_data[f'{int(threshold * 100)}% Threshold (start)_stddev']['All' + '_s' * scale] = np.std(start_offsets['all'])
                stddev_data[f'{int(threshold * 100)}% Threshold (end)_stddev']['All' + '_s' * scale] = np.std(end_offsets['all'])
                #plt.hist(start_offsets['all'])
                #plt.show()
                #plt.hist(end_offsets['all'])
                #plt.show()
                


            threshold_array.append(threshold)
            peaks_array.append(peak_difference)
            start_array.append(np.mean(start_offsets['all']))
            end_array.append(np.mean(end_offsets['all']))

        
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=False, figsize = (8,10))
        ax1.plot(threshold_array, peaks_array, color = 'tab:blue', marker = '', linestyle = '-', label = 'Difference in Number of Peaks')
        ax2.plot(threshold_array, start_array, color = 'tab:green', marker = '', linestyle = '-', label = 'Start-of-Peak offset')
        ax2.plot(threshold_array, end_array, color = 'tab:red', marker = '', linestyle = '-', label = 'End-of-Peak offset')

        ax1.set_xticks(np.linspace(0, 1, 11))
        ax2.set_xticks(np.linspace(0, 1, 11))

        ax1.set_xlim(0, 1)
        ax2.set_xlim(0, 1)

        ax1.set_ylabel('Peaks')
        ax1.set_xlabel('Threshold')
        ax2.set_ylabel('Days')
        ax2.set_xlabel('Threshold')

        ax1.set_title('Peaks vs Threshold ' + '(scaled) ' * scale + f'for model with minimum peak width = {args.width}')

        ax1.legend(loc = 1, bbox_to_anchor = (1.05, 1))
        ax2.legend(loc = 1, bbox_to_anchor = (1.05, 1))

        #plt.show()

    print(table_data)
    print(stddev_data)

    table_data.to_csv('tables/' + args.results[8:-4] + ('_state' * (1 - args.county)) + '_threshold_table.csv')
    stddev_data.to_csv('tables/' + args.results[8:-4] + ('_state' * (1 - args.county)) + '_threshold_table_stddev.csv')
    with open('tables/' + args.results[8:-4] + '_threshold_table_latex.txt', 'w') as fp:
        terms = args.results.split('_')
        fp.write(to_latex(table_data, stddev_data, '_'.join([terms[1]] + terms[3:-1])))

if __name__ == '__main__':
    main()