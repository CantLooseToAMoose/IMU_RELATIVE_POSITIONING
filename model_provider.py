import tensorflow as tf
import numpy as np
from tensorflow import keras
import math


def own_mse_loss(y_true, y_pred):
    return tf.math.reduce_mean(tf.square(y_true - y_pred))


def own_sum_loss(y_true, y_pred):
    return tf.math.reduce_sum(tf.square(y_true - y_pred))


def own_angle_sum_loss(y_true, y_pred):
    tensor_pi = [0, 2 * math.pi]
    y_pred_1 = tf.add(y_pred, tensor_pi)
    y_pred_2 = tf.subtract(y_pred, tensor_pi)
    return tf.reduce_min(tf.stack(
        [tf.math.reduce_sum(tf.square(y_true - y_pred)), tf.math.reduce_sum(tf.square(y_true - y_pred_1)),
         tf.math.reduce_sum(tf.square(y_true - y_pred_2))]))


def get_td_dense_lstm_dense_model(td_dense_layers, lstm_layers, dense_layers, input_shape, model_file_path, n_out,
                                  sequence_to_sequence: bool):
    model = keras.models.Sequential()
    for i in range(len(td_dense_layers)):
        if i == 0:
            model.add(keras.layers.TimeDistributed(
                keras.layers.Dense(td_dense_layers[i], input_shape=input_shape, activation="relu")))
        else:
            model.add(keras.layers.TimeDistributed(keras.layers.Dense(td_dense_layers[i], activation="relu")))

    for i in range(len(lstm_layers)):
        if (i == 0 & len(td_dense_layers) == 0):
            if (len(lstm_layers) == 1):
                model.add(keras.layers.LSTM(lstm_layers[i], return_sequences=False, input_shape=input_shape))
            else:
                model.add(keras.layers.LSTM(lstm_layers[i], return_sequences=True, input_shape=input_shape))
        else:
            if (i == len(lstm_layers) - 1):
                model.add(keras.layers.LSTM(lstm_layers[i], return_sequences=sequence_to_sequence))
            else:
                model.add(keras.layers.LSTM(lstm_layers[i], return_sequences=True))

    if (sequence_to_sequence):
        model.add(keras.layers.TimeDistributed(keras.layers.Dense(2)))
    else:
        for i in range(len(dense_layers)):
            model.add(keras.layers.Dense(dense_layers[i], activation="relu"))
        model.add(keras.layers.Dense(n_out))
    model.compile(loss=own_angle_sum_loss, optimizer="adam")
    checkpoint = keras.callbacks.ModelCheckpoint(model_file_path,
                                                 monitor='val_loss',
                                                 save_best_only=True,
                                                 save_weights_only=True)
    return model, checkpoint


def get_bi_td_dense_lstm_dense_model(td_dense_layers, lstm_layers, dense_layers, input_shape, model_file_path, n_out,
                                     sequence_to_sequence: bool):
    model = keras.models.Sequential()
    for i in range(len(td_dense_layers)):
        if i == 0:
            model.add(keras.layers.Bidirectional(keras.layers.TimeDistributed(
                keras.layers.Dense(td_dense_layers[i], input_shape=input_shape, activation="relu"))))
        else:
            model.add(keras.layers.Bidirectional(
                keras.layers.TimeDistributed(keras.layers.Dense(td_dense_layers[i], activation="relu"))))

    for i in range(len(lstm_layers)):
        if (i == 0 & len(td_dense_layers) == 0):
            if (len(lstm_layers) == 1):
                model.add(keras.layers.Bidirectional(
                    keras.layers.LSTM(lstm_layers[i], return_sequences=False, input_shape=input_shape)))
            else:
                model.add(keras.layers.Bidirectional(
                    keras.layers.LSTM(lstm_layers[i], return_sequences=True, input_shape=input_shape)))
        else:
            if (i == len(lstm_layers) - 1):
                model.add(keras.layers.Bidirectional(
                    keras.layers.LSTM(lstm_layers[i], return_sequences=sequence_to_sequence)))
            else:
                model.add(keras.layers.Bidirectional(keras.layers.LSTM(lstm_layers[i], return_sequences=True)))

    if (sequence_to_sequence):
        model.add(keras.layers.TimeDistributed(keras.layers.Dense(2)))
    else:
        for i in range(dense_layers):
            model.add(keras.layers.Dense(dense_layers[i], activation="relu"))
        model.add(keras.layers.Dense(n_out))
    model.compile(loss=own_angle_sum_loss, optimizer="adam")
    checkpoint = keras.callbacks.ModelCheckpoint(model_file_path,
                                                 monitor='val_loss',
                                                 save_best_only=True,
                                                 save_weights_only=True)
    return model, checkpoint
