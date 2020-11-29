import numpy as np, pandas as pd
import os, argparse, json
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import savgol_filter
import tensorflow as tf
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt

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
    #groups = results[results['Year'] >= 2019].groupby(by = 'County')
    groups = results.groupby(by = 'County')
    output = {}
    for city, subset in groups:
        if cities is not None and city in cities:
            for year in [2019, 2020]:
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
    if num_peaks_true and num_peaks_pred:
        for peak in peaks_true:
            peak_offsets = [(-(peak[0] - x[0]), -(peak[1] - x[1])) for x in peaks_pred]
            offsets.append(min(peak_offsets, key = lambda x: abs(x[0]) + abs(x[1])))
        mean_start_offset = sum(x[0] for x in offsets) / len(offsets)
        mean_end_offset = sum(x[1] for x in offsets) / len(offsets)
    else:
        mean_start_offset = mean_end_offset = 0
    return {'True Peaks': num_peaks_true, 'Predicted Peaks': num_peaks_pred, 'Mean Start Offset': mean_start_offset, 'Mean End Offset': mean_end_offset}

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
 
def main():
    args = parse_args()

    if args.results:
        if args.county:
            COUNTIES = ['Avondale,Arizona', 'Ventura,California', 'Butte,California', 'Waukesha,Wisconsin', 'Collier,Florida', 'Cameron,Texas']
            output = load_results_data(args.results, COUNTIES)
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
        indices = COUNTIES
    else:
        indices = sorted(set(city_to_state['State']))
    
    indices = indices + [s + '_s' for s in indices]
    table_data = pd.DataFrame(0.0, index = indices, columns=['n', '20% n', '40% n', '60% n', '80% n'] + [f'{i - i % 2}0% Threshold ({"end" if i % 2 else "start"})' for i in range(2, 10)])

    for scale in [True, False]:

        threshold_array = []
        matching_array = []
        less_array = []
        more_array = []
        start_array = []
        end_array = []

        for threshold in sorted([0.2, 0.4, 0.6, 0.8] + list(np.linspace(0.01, 0.99, 200))):
            results = compare_peaks(output, min_offset, threshold=threshold, peak_width=args.width, scale_to_1=scale, smooth=smooth)

            matching_peaks = 0
            offset_divisor = 0
            matching_start_offset = 0
            matching_end_offset = 0
            less_peaks = 0
            more_peaks = 0
            for city, result in results.items():
                if args.county:
                    index = city[:-5] + '_s' * scale
                else:    
                    index = city[:-5].split(',')[1] + '_s' * scale
                if threshold == 0.2:
                    table_data['n'][index] += 1

                if result['True Peaks'] == result['Predicted Peaks']:
                    matching_peaks += result['True Peaks']
                    matching_start_offset += result['Mean Start Offset'] * result['True Peaks']
                    matching_end_offset += result['Mean End Offset'] * result['True Peaks']
                    offset_divisor += 1

                    if threshold in [0.2, 0.4, 0.6, 0.8]:
                        table_data[f'{int(threshold * 100)}% Threshold (start)'][index] += result['Mean Start Offset']
                        table_data[f'{int(threshold * 100)}% Threshold (end)'][index] += result['Mean End Offset']
                        table_data[f'{int(threshold * 100)}% n'][index] += 1

                elif result['True Peaks'] < result['Predicted Peaks']:
                    more_peaks += result['Predicted Peaks'] - result['True Peaks']
                else:
                    less_peaks += result['True Peaks'] - result['Predicted Peaks']

            threshold_array.append(threshold)
            matching_array.append(matching_peaks)
            less_array.append(less_peaks)
            more_array.append(more_peaks)
            start_array.append(matching_start_offset / offset_divisor)
            end_array.append(matching_end_offset / offset_divisor)

        
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=False, figsize = (8,10))
        ax1.plot(threshold_array, matching_array, color = 'tab:green', marker = '', linestyle = '-', label = 'Matching Peaks')
        ax1.plot(threshold_array, less_array, color = 'tab:blue', marker = '', linestyle = '-', label = 'Missing Peaks')
        ax1.plot(threshold_array, more_array, color = 'tab:orange', marker = '', linestyle = '-', label = 'Extra Peaks')
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

    table_data['20% Threshold (start)'] /= table_data['20% n']
    table_data['20% Threshold (end)'] /= table_data['20% n']
    table_data['40% Threshold (start)'] /= table_data['40% n']
    table_data['40% Threshold (end)'] /= table_data['40% n']
    table_data['60% Threshold (start)'] /= table_data['60% n']
    table_data['60% Threshold (end)'] /= table_data['60% n']
    table_data['80% Threshold (start)'] /= table_data['80% n']
    table_data['80% Threshold (end)'] /= table_data['80% n']
    print(table_data)

    table_data.to_csv('tables/' + args.results[:-4].split('/')[-1] + '_state_threshold_table.csv')

if __name__ == '__main__':
    main()