import data_provider
import model_provider
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import testing


def layers_toString(layers):
    ret = "["
    for i in range(len(layers)):
        ret += str(layers[i]) + ","
    ret = ret[:-1]
    ret += "]"
    return ret


def evaluate_polar_test(y_pred, Y_test, plot=False):
    errors_r = np.abs(y_pred[:, 0] - Y_test[:, 0])
    errors_r = np.sort(errors_r)
    errors_a = np.abs(y_pred[:, 1] - Y_test[:, 1])
    errors_a = np.sort(errors_a)
    p = 1. * np.arange(len(errors_r)) / (len(errors_r) - 1)
    mean_r = np.mean(errors_r)
    std_r = np.std(errors_r)
    minimum_r = np.min(errors_r)
    maximum_r = np.max(errors_r)
    print("mean_radius=", mean_r)
    print("std_radius=", std_r)
    print("min_radius=", minimum_r)
    print("max_radius=", maximum_r)
    mean_a = np.mean(errors_a)
    std_a = np.std(errors_a)
    minimum_a = np.min(errors_a)
    maximum_a = np.max(errors_a)
    print("mean_angle=", mean_a)
    print("std_angle=", std_a)
    print("min_ange=", minimum_a)
    print("max_angle=", maximum_a)
    if (plot):
        fig, axs = plt.subplots(2)
        axs[0].plot(errors_r, p)
        fig.suptitle("CDF")
        axs[0].set_xlabel("error in meters")
        axs[1].plot(errors_a, p)
        axs[1].set_xlabel("angle in radians")
        plt.show()
    return np.array([mean_r, std_r, minimum_r, maximum_r])


def naive_lstm_workflow_for_one_experiment(datapath, train, plot, seq_length, stride, number_of_features=6, lstm_layers=[300, 100, 100]):
    polar, imu = data_provider.load_imu_and_polar_vector_list_of_lists_for_one_experiment(
        datapath, window_size=seq_length, stride=stride)
    testing.test_if_preprocessing_is_working_for_every_file_in_experiment(
        datapath, window_size=seq_length, stride=stride, plot=True)
    polar, imu = data_provider.convert_list_of_lists_of_data_from_one_experiment_to_2D_numpy_array(polar, imu)
    # scaler_imu = StandardScaler()
    X_train, X_test, Y_train, Y_test = train_test_split(imu, polar, test_size=0.3, shuffle=True, random_state=42)
    X_train = X_train.reshape(-1, number_of_features)
    # X_train = scaler_imu.fit_transform(X_train)
    X_train = X_train.reshape(-1, seq_length, number_of_features)
    X_test = X_test.reshape(-1, number_of_features)
    # X_test = scaler_imu.transform(X_test)
    X_test = X_test.reshape(-1, seq_length, number_of_features)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.3, shuffle=True, random_state=42)
    input_shape = [seq_length, number_of_features]
    model_file_path = 'models\\models_lstm\\model_polar_with_stride_' + str(stride) + "_layers_" + layers_toString(
        lstm_layers) + "_seq_length_" + str(
        seq_length) + '.hdf5'
    model, checkpoint = model_provider.get_td_dense_lstm_dense_model(td_dense_layers=[128],lstm_layers=lstm_layers,dense_layers=[256], input_shape=input_shape, model_file_path=model_file_path,n_out=2,
                                                                     sequence_to_sequence=False)

    if (train):
        model.fit(X_train, Y_train, epochs=50, batch_size=64, validation_data=(X_val, Y_val), callbacks=[checkpoint])
    else:
        model.fit(X_train, Y_train, epochs=1, steps_per_epoch=1, batch_size=64, validation_split=0.1)

    model.load_weights(model_file_path)
    y_pred = model.predict(X_test)
    if (plot):
        fig, axs = plt.subplots(2)
        fig.suptitle("Test")
        axs[0].plot(np.arange(len(Y_test)), Y_test[:, 0], label="Truth")
        axs[0].plot(np.arange(len(Y_test)), y_pred[:, 0], label="Prediction")
        plt.legend()
        axs[0].set_title("Radius")
        axs[1].plot(np.arange(len(Y_test)), Y_test[:, 1], label="Truth")
        axs[1].plot(np.arange(len(Y_test)), y_pred[:, 1], label="Prediction")
        axs[1].set_title("Angle")
        plt.show()

    eval = evaluate_polar_test(y_pred, Y_test, plot)

    return y_pred, Y_test, eval,model
