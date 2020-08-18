import tensorflow as tf

def conv_model(input_shape):
    xin = tf.keras.layers.Input(input_shape)
    
    c1 = tf.keras.layers.Conv1D(128, 3, activation = 'relu', data_format = 'channels_last')(xin)
    c2 = tf.keras.layers.Conv1D(128, 3, activation = 'relu', data_format = 'channels_last')(c1)
    c3 = tf.keras.layers.Conv1D(128, 3, activation = 'relu', data_format = 'channels_last')(c2)

    h1 = tf.keras.layers.BatchNormalization()(c3)
    h2 = tf.keras.layers.Flatten()(h1)
    h3 = tf.keras.layers.Dense(256, activation = 'relu')(h2)

    xout = tf.keras.layers.Dense(1, activation = 'relu')(h3)
    
    return tf.keras.models.Model(xin, xout)

def lstm_model(input_shape):
    xin = tf.keras.layers.Input(input_shape)
    
    c1 = tf.keras.layers.Conv1D(64, 3, activation = 'relu', data_format = 'channels_last')(xin)
    c2 = tf.keras.layers.Conv1D(64, 3, activation = 'relu', data_format = 'channels_last')(c1)
    
    h1 = tf.keras.layers.BatchNormalization()(c2)
    h2 = tf.keras.layers.Permute((2, 1))(h1)
    
    r1 = tf.keras.layers.LSTM(64, return_sequences = True)(h2)
    r2 = tf.keras.layers.LSTM(64)(r1)

    xout = tf.keras.layers.Dense(1, activation = 'relu')(r2)
    
    return tf.keras.models.Model(xin, xout)

def gru_model(input_shape):
    xin = tf.keras.layers.Input(input_shape)
    
    c1 = tf.keras.layers.Conv1D(64, 3, activation = 'relu', data_format = 'channels_last')(xin)
    c2 = tf.keras.layers.Conv1D(64, 3, activation = 'relu', data_format = 'channels_last')(c1)
    
    h1 = tf.keras.layers.BatchNormalization()(c2)
    h2 = tf.keras.layers.Permute((2, 1))(h1)
    
    r1 = tf.keras.layers.GRU(64, return_sequences = True)(h2)
    r2 = tf.keras.layers.GRU(64)(r1)

    xout = tf.keras.layers.Dense(1, activation = 'relu')(r2)
    
    return tf.keras.models.Model(xin, xout)