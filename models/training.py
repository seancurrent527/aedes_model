import tensorflow as tf
import tensorflow.keras.backend as K
import argparse, os, json
import pandas as pd, numpy as np
from sklearn.preprocessing import MinMaxScaler
import models, visuals

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
np.random.seed(14)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='The json configuration file for the aedes model.')
    parser.add_argument('-t', '--testing', action='store_true', help='Whether or not to run the test set.')
    parser.add_argument('-l', '--load', action='store_true', help='Whether or not to load a previously defined model.')
    return parser.parse_args()

def format_data(data, data_shape, samples_per_city, scaler = None, fit_scaler = False):
    data.columns = range(0, len(data.columns))
    if fit_scaler:
        scaler.fit(data.iloc[:, -(data_shape[1] + 1):])
    groups = data.groupby(by = 0)
    data = []
    for _, subset in groups:
        random_indices = np.random.randint(0, len(subset) - (data_shape[0] + 1), size = samples_per_city)
        for i in range(samples_per_city):
            random_index= random_indices[i]
            data.append(scaler.transform(subset.iloc[random_index: random_index + data_shape[0], -(data_shape[1] + 1):].values))
    return np.array(data) if not fit_scaler else (np.array(data), scaler)

def split_and_shuffle(data):
    permutation = np.random.permutation(len(data))
    return data[permutation, :, :-1], data[permutation, -1, -1]

def r2_keras(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return (1 - SS_res/(SS_tot + K.epsilon()))

def main():
    args = parse_args()
    with open(args.config) as fp:
        config = json.load(fp)

    for key, value in config['files'].items():
        config['files'][key] = value.replace('/', '\\')

    # load the model as necessary
    if args.load and os.path.exists(config['files']['model']):
        model = tf.keras.models.load_model(config['files']['model'], custom_objects = r2_keras)
    else:
        model = getattr(models, config['model'])(config['data']['data_shape'])
    
    # get the data
    training = pd.read_pickle(os.path.expanduser(config['files']['training']))
    validation = pd.read_pickle(os.path.expanduser(config['files']['validation']))
    testing = pd.read_pickle(os.path.expanduser(config['files']['testing']))
    training, scaler = format_data(training, config['data']['data_shape'], config['data']['samples_per_city'],
                                   scaler=MinMaxScaler(), fit_scaler=True)
    validation = format_data(validation, config['data']['data_shape'], config['data']['samples_per_city'],
                             scaler=scaler)
    testing = format_data(testing, config['data']['data_shape'], config['data']['samples_per_city'],
                          scaler=scaler)
    X_train, y_train = split_and_shuffle(training)
    X_val, y_val = split_and_shuffle(validation)
    X_test, y_test = split_and_shuffle(testing)

    if args.testing:
        print('Running test set...')
        model.evaluate(X_test, y_test)

    else:
        model.compile(optimizer = getattr(tf.keras.optimizers, config['compile']['optimizer'])(lr = config['compile']['learning_rate']),
                      loss = config['compile']['loss'], metrics = [r2_keras])
        history = model.fit(X_train, y_train, validation_data = (X_val, y_val), **config['fit'],
                  callbacks = [tf.keras.callbacks.TensorBoard(), tf.keras.callbacks.EarlyStopping(patience = 30, restore_best_weights = True)])
        model.save(os.path.expanduser(config['files']['model']))

        visuals.plot_loss(history)
        visuals.plot_r2(history)


if __name__ == '__main__':
    main()