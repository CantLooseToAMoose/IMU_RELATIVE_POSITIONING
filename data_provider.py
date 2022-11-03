import pandas as pd
import os
import lib
import numpy as np


def load_imu_and_polar_vector_list_of_lists_for_one_experiment(path, window_size,stride=0):
    polar_list_of_lists, imu_list_of_lists = list(), list()
    all_sub_dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    for directory in all_sub_dirs:
        data_path = os.path.join(path, directory)
        polar_list, imu_list = load_sensor_and_polar_vector_list_for_one_data_folder(data_path, window_size,stride)
        polar_list_of_lists.append(polar_list)
        imu_list_of_lists.append(imu_list)

    return polar_list_of_lists, imu_list_of_lists


def load_sensor_and_polar_vector_list_for_one_data_folder(path, window_size, stride=0):
    sensor_list, vico_list = lib.load_vico_and_imu_as_dataframes_from_inside_syn_folder(path + "/syn")
    polar_list, imu_list = list(), list()
    for vico in vico_list:
        if stride == 0:
            polar = load_polar_vector_array_for_one_data_file(vico, window_size)
            polar_list.append(polar)
        else:
            for i in range(int(window_size/stride)):
                polar = load_polar_vector_array_for_one_data_file(vico, window_size,offset=i*stride)
                polar_list.append(polar)
    for sensor_data in sensor_list:
        if stride == 0:
            imu = load_sensor_data_array_for_one_data_file(sensor_data, window_size)
            imu_list.append(imu)
        else:
            for i in range(int(window_size/stride)):
                imu = load_sensor_data_array_for_one_data_file(sensor_data, window_size,offset=i*stride)
                imu_list.append(imu)


    return polar_list, imu_list


def load_polar_vector_array_for_one_data_file(vico, window_size, offset=0):
    vico = vico.to_numpy()
    pos = vico[offset::window_size, [2, 3]]
    delta_vector = np.subtract(pos[1:], pos[:-1])
    delta_loc = np.linalg.norm(delta_vector, axis=1)
    headings = np.zeros((len(delta_vector)))
    headings[0] = np.arctan2(delta_vector[0, 1], delta_vector[0, 0])
    headings[1:] = lib.AngleToTheLastVector(delta_vector)
    return np.concatenate([delta_loc.reshape(-1, 1), headings.reshape(-1, 1)], axis=1)


def load_sensor_data_array_for_one_data_file(sensor_data, window_size, offset=0):
    sensor_data = sensor_data.to_numpy()
    sensor_data = sensor_data[offset:-1, 1:]
    cutoff = len(sensor_data) % window_size
    features = len(sensor_data[0])
    if cutoff != 0:
        sensor_data = sensor_data[:-cutoff]
    sensor_data = sensor_data.reshape(-1, window_size, features)

    return sensor_data


def convert_list_of_lists_of_data_from_one_experiment_to_2D_numpy_array(polar_list_of_lists, imu_list_of_lists):
    polar_array = list()
    imu_array = list()
    for polar_list in polar_list_of_lists:
        polar_array.append(np.concatenate(polar_list, axis=0))
    polar_array = np.concatenate(polar_array, axis=0)
    for imu_list in imu_list_of_lists:
        imu_array.append(np.concatenate(imu_list, axis=0))
    imu_array = np.concatenate(imu_array, axis=0)
    return polar_array, imu_array


def load_imu_and_polar_vector_array_for_one_experiment(path, window_size):
    polar_list_of_lists, imu_list_of_lists = load_imu_and_polar_vector_list_of_lists_for_one_experiment(path,
                                                                                                        window_size)
    return convert_list_of_lists_of_data_from_one_experiment_to_2D_numpy_array(polar_list_of_lists, imu_list_of_lists)


def load_translation_x_y_list_of_lists_from_one_experiment(path, window_size,stride=0):
    translation_list_of_lists = list()
    all_sub_dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    for directory in all_sub_dirs:
        data_path = os.path.join(path, directory)
        translation_list = load_translation_x_y_list_from_one_data_folder(data_path, window_size,stride=stride)
        translation_list_of_lists.append(translation_list)
    return translation_list_of_lists


def load_translation_x_y_list_from_one_data_folder(path, window_size,stride=0):
    sensor_list, vico_list = lib.load_vico_and_imu_as_dataframes_from_inside_syn_folder(path + "/syn")
    translation_list = list()
    for vico in vico_list:
        if stride==0:
            translation_list.append(load_translation_x_y_from_one_data_file(vico, window_size))
        else:
            for i in range(int(window_size/stride)):
                translation_list.append(load_translation_x_y_from_one_data_file(vico,window_size,offset=i*stride))
    return translation_list


def load_translation_x_y_from_one_data_file(vico, window_size,offset=0):
    vico = vico.to_numpy()
    return vico[offset::window_size, [2, 3]]
