import data_provider
import numpy as np
import matplotlib.pyplot as plt

import lib


def test_if_preprocessing_is_working_for_every_file_in_experiment(path, window_size, stride=0, plot: bool = False):
    polar_list_of_lists, imu_list_of_lists = data_provider.load_imu_and_polar_vector_list_of_lists_for_one_experiment(
        path, window_size=window_size, stride=stride)
    translation_list_of_lists = data_provider.load_translation_x_y_list_of_lists_from_one_experiment(path,
                                                                                                     window_size=window_size,
                                                                                                     stride=stride)
    # print("There are " + str(len(polar_list_of_lists)) + " data folders in the path \"" + str(path) + "\".")
    for i in range(len(polar_list_of_lists)):
        # print("In the " + str(i+1) + "-data folder there are " + str(len(
        #     polar_list_of_lists)) + " experiments which will be tested.")
        polar_list = polar_list_of_lists[i]
        translation_list = translation_list_of_lists[i]
        imu_list = imu_list_of_lists[i]
        for j in range(len(polar_list)):

            translation = translation_list[j][:-1]
            polar = polar_list[j]
            imu = imu_list[j]

            assert len(polar) == len(imu), "Problem with File [" + str(i + 1) + "," + str(
                j + 1) + "]: polar vector array with shape " + str(polar.shape) + " and imu array with shape " + str(
                imu.shape) + " is not matching"

            translation_backwards = lib.convert_polar_to_absolute(start_pos=translation[0], start_rot=0,
                                                                  delta_loc=polar[:, 0], headings=polar[:, 1])
            backward_test = np.allclose(translation, translation_backwards)
            assert backward_test, "Problem with File [" + str(i + 1) + "," + str(
                j + 1) + "]: backward_preprocessing not working"
            # print("File [" + str(i+1) + "," + str(j+1) + "]: " + str(test))
            if (plot):
                fig,axs=plt.subplots(2)
                axs[0].axis("equal")
                axs[1].axis("equal")
                for i in range(len(translation)):
                    axs[0].plot(translation[i:i+2, 0], translation[i:i+2, 1], "-ro", label="Truth")
                    axs[1].plot(translation_backwards[i:i+2, 0], translation_backwards[i:i+2, 1], "-bo", label="Backwards")
                    print(str(polar[i,0])+","+str(polar[i,1]*180/np.pi))
                    print()
                fig.suptitle("Data: " + str(i + 1) + "  File: " + str(j + 1))
                plt.show()
            if not backward_test:
                print("-------------------")
                print("Debug Info:")
                print("Translation_shape: " + str(translation.shape))
                print("Polar_shape: " + str(polar.shape))
                print("Translation_backwards_shape: " + str(translation_backwards.shape))
                print("-------------------")
