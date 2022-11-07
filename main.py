import numpy as np

import data_provider
import lib
import matplotlib

import model_provider

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import testing
import workflow
import tensorflow as tf

# polar_list_of_lists, imu_list_of_lists = data_provider.load_imu_and_polar_vector_list_of_lists_for_one_experiment(
#     "Oxford Inertial Odometry Dataset/handheld", window_size=100,stride=10)

pred, test, eval, model = workflow.naive_lstm_workflow_for_one_experiment("Oxford Inertial Odometry Dataset/handheld",
                                                                          train=False, plot=True,
                                                                          seq_length=200, stride=10,
                                                                          lstm_layers=[128, 256])

# print(np.ptp(pred, axis=0))
# print(np.ptp(test, axis=0))
Y, X = data_provider.load_sensor_and_polar_vector_list_for_one_data_folder(
    "Oxford Inertial Odometry Dataset/handheld/data5", window_size=200)
X_one, Y_one = X[0], Y[0]
translation = data_provider.load_translation_x_y_list_from_one_data_folder(
    "Oxford Inertial Odometry Dataset/handheld/data5", window_size=200)
translation_one = translation[0]
Y_pred_one = model.predict(X_one)
translation_one_pred_backwards = lib.convert_polar_to_absolute(translation_one[0], 0, Y_pred_one[:, 0],
                                                               Y_pred_one[:, 1])
translation_one_backwards = lib.convert_polar_to_absolute(translation_one[0], 0, Y_one[:, 0], Y_one[:, 1])
fig, axs = plt.subplots(2)
fig.suptitle("Test")
axs[0].plot(np.arange(len(Y_one)), Y_one[:, 0], label="Truth")
axs[0].plot(np.arange(len(Y_one)), Y_pred_one[:, 0], label="Prediction")
axs[0].set_title("Radius")
axs[1].plot(np.arange(len(Y_one)), Y_one[:, 1], label="Truth")
axs[1].plot(np.arange(len(Y_one)), Y_pred_one[:, 1], label="Prediction")
axs[1].set_title("Angle")
plt.legend()
plt.show()

for i in range(len(translation_one)):
    plt.plot(translation_one[i:i + 2, 0], translation_one[i:i + 2, 1], "-ro", label="Truth")
    plt.plot(translation_one_backwards[i:i + 2, 0], translation_one_backwards[i:i + 2, 1], "-bo", label="backwards")
    plt.plot(translation_one_pred_backwards[i:i + 2, 0], translation_one_pred_backwards[i:i + 2, 1], "-go",
             label="backwards_pred")

plt.legend()
plt.show()
