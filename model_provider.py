import tensorflow as tf

from tensorflow import keras


def own_mse_loss(y_true, y_pred):
    return tf.math.reduce_mean(tf.square(y_true - y_pred))


def own_sum_loss(y_true, y_pred):
    return tf.math.reduce_sum(tf.square(y_true - y_pred))


def get_lstm_model(layers, input_shape, model_file_path, sequence_to_sequence: bool,bi:bool=False):
    model = keras.models.Sequential()
    for i in range(len(layers)):
        if (i == 0):
            if (len(layers) == 1):
                model.add(keras.layers.LSTM(layers[i], return_sequences=False, input_shape=input_shape))
            else:
                model.add(keras.layers.LSTM(layers[i], return_sequences=True, input_shape=input_shape))
        else:
            if (i == len(layers) - 1):
                model.add(keras.layers.LSTM(layers[i], return_sequences=sequence_to_sequence))
            else:
                model.add(keras.layers.LSTM(layers[i], return_sequences=True))

    if (sequence_to_sequence):
        model.add(keras.layers.TimeDistributed(keras.layers.Dense(2)))
    else:
        model.add(keras.layers.Dense(256, activation="relu"))
        model.add(keras.layers.Dense(2))
    model.compile(loss=own_sum_loss, optimizer="adam")
    checkpoint = keras.callbacks.ModelCheckpoint(model_file_path,
                                                 monitor='val_loss',
                                                 save_best_only=True,
                                                 save_weights_only=True)
    return model, checkpoint
