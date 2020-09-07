import numpy as np, pandas as pd
import os, argparse, json
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import savgol_filter
import tensorflow as tf
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='The json configuration file for the aedes model.')
    parser.add_argument('-t', '--threshold', type = float, help = 'The threshold for declaring a peak.')
    parser.add_argument('-w', '--width', type = int, default = 5, help = 'The width for declaring a peak.')
    return parser.parse_args()

def load_val_data(config):
    training_data = pd.read_pickle(os.path.expanduser(config['files']['training']))
    validation = pd.read_pickle(os.path.expanduser(config['files']['validation']))
    scaler = MinMaxScaler()
    scaler.fit(training_data.iloc[:, -(config['data']['data_shape'][1] + 1):])
    validation.columns = range(0, len(validation.columns))
    groups = validation.groupby(by = 0)
    data = {}
    data_shape = config['data']['data_shape']
    for city, subset in groups:
        sub_data = []
        for i in range(0, len(subset) - (data_shape[0] + 1)):
            sub_data.append(scaler.transform(subset.iloc[i: i + data_shape[0], -(data_shape[1] + 1):].values))
        data[city+',2019'] = np.array(sub_data[:365 - data_shape[0]])
        data[city+',2020'] = np.array(sub_data[365 - data_shape[0]:])
    return data

def r2_keras(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return (1 - SS_res/(SS_tot + K.epsilon()))

def load_model(config):
    model_file = os.path.expanduser(config['files']['model'])
    if os.path.exists(model_file):
        model = tf.keras.models.load_model(model_file, custom_objects = {'r2_keras': r2_keras})
        print('MODEL LOADED')
        return model
    else:
        raise FileNotFoundError(f"Model does not exist at {config['files']['model']}")

def run_model(model, validation):
    output = {}
    for city, data in validation.items():
        X, y_true = data[:, :, :-1], data[:, -1, -1]
        y_pred = model.predict(X).flatten()
        output[city] = (y_pred, y_true)
    return output

def smooth_data(data, rounds = 1):
    for _ in range(rounds):
        data = savgol_filter(data, 11, 3)
        data[data < 0] = 0
    return data / data.max()

def peak_finder(array, threshold = 0.8, peak_width = 5):
    peaks = set()
    above_threshold = array > (threshold * array.max())
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

def compare_peaks(output, metric, threshold=0.8, peak_width = 5):
    results = {}
    for city, (y_pred, y_true) in output.items():
        y_pred_smoothed = smooth_data(y_pred, rounds = 2)
        y_true_scaled = smooth_data(y_true, rounds = 0)
        peaks_pred = peak_finder(y_pred_smoothed, threshold=threshold, peak_width=peak_width)
        peaks_true = peak_finder(y_true_scaled, threshold=threshold, peak_width=peak_width)
        results[city] = metric(peaks_pred, peaks_true)
    return results
 
def main():
    args = parse_args()
    with open(args.config) as fp:
        config = json.load(fp)

    for key, value in config['files'].items():
        config['files'][key] = value.replace('/', '\\')

    val_data = load_val_data(config)
    model = load_model(config)

    output = run_model(model, val_data)

    threshold_array = []
    matching_array = []
    less_array = []
    more_array = []
    start_array = []
    end_array = []

    for threshold in np.linspace(0.01, 0.99, 200):
        results = compare_peaks(output, min_offset, threshold=threshold, peak_width=args.width)

        matching_peaks = 0
        offset_divisor = 0
        matching_start_offset = 0
        matching_end_offset = 0
        less_peaks = 0
        more_peaks = 0
        for city, result in results.items():
            #print(f"{city}: {result}")
            if result['True Peaks'] == result['Predicted Peaks']:
                matching_peaks += result['True Peaks']
                matching_start_offset += result['Mean Start Offset'] * result['True Peaks']
                matching_end_offset += result['Mean End Offset'] * result['True Peaks']
                offset_divisor += 1
            elif result['True Peaks'] < result['Predicted Peaks']:
                more_peaks += result['Predicted Peaks'] - result['True Peaks']
            else:
                less_peaks += result['True Peaks'] - result['Predicted Peaks']

        '''
        print()
        print(f'Threshold: {threshold}')
        print(f'Matching Peaks - {matching_peaks}')
        print(f'Mean Matching Start Offset - {matching_start_offset / offset_divisor}    Mean Matching End Offset - {matching_end_offset / offset_divisor}')
        print(f'Less Peaks Predicted - {less_peaks}')
        print(f'More Peaks Predicted - {more_peaks}')
        '''

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

    ax1.set_title(f'Peaks vs Threshold for LSTM model with minimum peak width = {args.width}')

    ax1.legend(loc = 1, bbox_to_anchor = (1.05, 1))
    ax2.legend(loc = 1, bbox_to_anchor = (1.05, 1))

    plt.show()

if __name__ == '__main__':
    main()