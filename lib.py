import math
import os
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R


def load_vico_and_imu_as_dataframes_from_inside_syn_folder(path):
    imu_list = list()
    vico_list = list()
    for file in os.listdir(path):
        if str(file).find("imu") != -1:
            imu_list.append(pd.read_csv(path + "/" + file))
        elif str(file).find("vi") != -1:
            vico_list.append(pd.read_csv(path + "/" + file))
    return imu_list, vico_list


def RotateVectorFromQuaternion(x, y, z, qx, qy, qz, qw):
    r = R.from_quat([qx, qy, qz, qw])
    return r.apply(np.array([x, y, z]))


def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)

    from https://automaticaddison.com/how-to-convert-a-quaternion-into-euler-angles-in-python/
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z  # in radians


def AngleToTheLastVector(x):
    angles = np.arctan2(x[:, 1], x[:, 0])
    subtracts = np.subtract(angles[1:], angles[:-1])
    subtracts[subtracts > np.pi] -= 2 * np.pi
    subtracts[subtracts < -np.pi] += 2 * np.pi
    return subtracts


def RotationMatrix(angle):
    return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])


def convert_polar_to_absolute(start_pos, start_rot, delta_loc, headings):
    start_vector = start_pos
    new_vector = list()
    new_vector.append(start_vector)
    rotation = start_rot

    for i in range(len(delta_loc) - 1):
        rotation += headings[i]
        start_vector = np.dot(RotationMatrix(rotation), np.array([delta_loc[i], 0])) + start_vector
        new_vector.append(start_vector)

    return np.array(new_vector)
