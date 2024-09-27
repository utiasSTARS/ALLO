import numpy as np
import random


#* all posssible camera positions and orientations
#* (camera index, x,y,z, roll, pitch, yaw)
CAMERA_DATA_TRAIN = np.array([[0, -37.69, -6.94, 4.93, 0.7887, 0, -0.0103],
                        [1, -32.058, -12.693, 0.985, 1.2991, 0, 0.412],
                        [2, -42.709, -11.351, -3.9103, 1.72, 0, -0.5548],
                        [3, -50.611, -8.0709, 2.6462, 1.1822, 0, -0.322],
                        [4, -38.208, -24.523, -3.015, 1.225, 0, 0.01935],
                        [5, -26.849, -20.747, -2.873, 1.1796, 0, 0.59967],
                        [6, -32.32, -20.491, -3.2598, 1.389, 0, -0.622],
                        [7, -14.746, -19.52, -0.9587, 1.2799, 0, 0.6895],
                        [8, -18.29, -12.02, 7.991, -0.0709, 0, 0.00975],
                        [9, -28.208, -4.4832, -19.123, 2.6727, 0, 1.552],
                        [10, -42.603, -1.9641, -19.939, 2.6867, 0, -1.561],
                        [11, -42.675, -9.8571, -9.5763, 1.6656, 0, -0.3358],
                        [12, -53.164, 6.536, -11.685, 1.595, 0, -1.4737],
                        [13, -47.231, 6.8322, -2.3479, 1.21976, 0, -1.8464],
                        [14, -38.236, 22.892, -9.2555, 1.38033, 0, -2.74],
                        [15, -26.651, 16.116, -8.1411, 1.3297, 0, -3.4573],
                        [16, -26.162, 14.556, -14.362, 2.0837, 0, -3.0524],
                        [17, -23.111, -3.4253, -11.53, 1.8672, 0, -5.991],
                        [18, -22.018, 0.3996, -14.101, 2.2442, 0, -5.5657],
                        [19, -27.629, -4.5704, -13.227, 1.83935, 0, -5.6494],
                        [20, -42.838, 9.8911, -3.066, 1.40484, 0, -3.5245],
                        [21, -49.696, -2.9966, -21.002, 2.6824, 0, -5.6887],
                        [22, -61.048, -4.4178, -13.375, 2.5707, 0, -6.7918],
                        [23, -45.757, 4.5353, 3.0974, 1.307, 0, -4.9208],
                        [24, -52.069, 9.041, 2.9575, 1.3559, 0, -3.8596],
                        [25, -69.677, -2.4612, 9.1964, 1.0557, 0, -1.6046],
                        [26, -18.072, 10.925, 0.2730, 1.1954, 0, -2.805],
                        [27, -21.737, -3.041, 2.2315, 1.369, 0, -1.472],
                        [28, -15.612, -3.7128, 10.227, 0.8262, 0, -1.3281],
                        [29, -4.074, -2.0981, 11.934, 0.2529, 0, 1.5945],
                        [30, 0.67647, 2.5488, -6.2168, 1.6526, 0, 2.077],
                        [31, -19.21, -7.9349, -0.7891, 0.5007, 0, 3.4532],
                        [32, -32.635, -20.179, -8.050, 1.953, 0, 5.705],
                        [33, -30.236, -13.446, 0.31413, 1.276, 0, 6.676],
                        [34, -48.26, 12.461, -11.389, 1.883, 0, 4.155],
                        [35, -43.892, 11.564, -2.455, 1.136, 0, 4.63],
                        [36, -34.366, 25.479, -15.159, 1.869, 0, 3.129],
                        [37, -12.044, -2.538, 16.697, 0.243, 0, 4.588],
                        [38, -56.953, 1.3727, -23.115, 3.021, 0, 2.9549],
                        [39, -24.11, 1.457, -17.8, 2.77, 0, 0.6721]])

#* Test camera positions and orientations
CAMERA_DATA_TEST = np.array([[0, -26.537, 27.247, -14.353, 1.73031, 0, -3.3945],
                        [1, -43.2772, 18.155, -14.919, 1.73903, 0, -2.69898],
                        [2, -32.684, -17.856, -14.368, 1.75117, 0, 0.030074],
                        [3, -49.273, -17.333, -6.388, 1.54634, 0, -0.646429],
                        [4, -70.522, -10.731, 5.139, 1.38908, 0, -1.18138],
                        [5, -30.004, -25.217, -2.630, 1.22848, 0, 0.413794],
                        [6, -12.492, 12.980, 8.232, 1.00619, 0, 3.26553],
                        [7, -6.200, -7.592, 5.144, 0.850549, -0.2, 0.850549],
                        [8, -21.709, -10.0781, -17.298, 2.06358, 0, 1.11829],
                        [9, -54.314, 13.020, -19.700, -0.966208, -3.1416, 0.022838]])


def get_cam_pos(cam_idx_list, anomalous):
    """Returns the camera positions and orientations at the given indices"""
    if anomalous:
        CAMERA_DATA = CAMERA_DATA_TEST
    else:
        CAMERA_DATA = CAMERA_DATA_TRAIN
    return CAMERA_DATA[cam_idx_list,:]

def get_camera_forward(cam_euler):
    ''' Calculates camera forward vector
    Args: 
        - cam_euler: euler rotation of camera in degrees
    Returns:
        - forward: forward unit vector of camera '''
    rotation_matrix = rotation_matrix_from_euler(cam_euler)
    forward = np.matmul(rotation_matrix,[0,0,-1])
    return forward

def rotation_matrix_from_euler(cam_ori):
    '''Calculates camera rotation matrix
    Args:
        - cam_ori: euler camera rotation in degrees
    Returns:
        - rotation_matrix: camera rotation matrix '''
    #? Convert angles to radians
    roll_rad = cam_ori[0]
    pitch_rad = cam_ori[1]
    yaw_rad = cam_ori[2]

    #? Calculate trigonometric values
    cos_roll = np.cos(roll_rad)
    sin_roll = np.sin(roll_rad)
    cos_pitch = np.cos(pitch_rad)
    sin_pitch = np.sin(pitch_rad)
    cos_yaw = np.cos(yaw_rad)
    sin_yaw = np.sin(yaw_rad)

    #? Calculate rotation matrix
    R_x = np.array([[1, 0, 0],
                    [0, cos_roll, -sin_roll],
                    [0, sin_roll, cos_roll]])
    R_y = np.array([[cos_pitch, 0, sin_pitch],
                    [0, 1, 0],
                    [-sin_pitch, 0, cos_pitch]])
    R_z = np.array([[cos_yaw, -sin_yaw, 0],
                    [sin_yaw, cos_yaw, 0],
                    [0, 0, 1]])

    #? Combine rotations
    rotation_matrix = np.dot(R_z, np.dot(R_y, R_x))

    return rotation_matrix