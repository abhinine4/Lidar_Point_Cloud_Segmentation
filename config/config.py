class cfg:
    PATH_TRAIN = r'/home/csgrad/mbhosale/Lidar_Segmentation/pandaset-devkit/data/PandaSet/subsampled/train/'
    PATH_VALID = r'/home/csgrad/mbhosale/Lidar_Segmentation/pandaset-devkit/data/PandaSet/subsampled/valid/'
    PATH = r'/home/csgrad/mbhosale/Lidar_Segmentation/pandaset-devkit/data/PandaSet/subsampled/'
    sub_grid_size = 0.04
    num_points = 81920  # Number of input points
    sub_grid_size = 0.04  # preprocess_parameter

    train_steps = 240  # Number of steps per epochs
    val_steps = 100    # Number of validation steps per epoch

    sampling_type = 'active_learning'
    data_name = 'pandaset'
    if data_name == 's3dis':
        class_weights = [1938651, 1242339, 608870, 1699694, 2794560, 195000, 115990, 549838, 531470, 292971, 196633, 59032, 209046, 39321]
    else:
        class_weights = [2.11006107e+04, 1.91488042e+06, 2.25280049e+04, 2.57896353e+04,
       4.41526212e+00, 1.48011476e-01, 7.86469909e-01, 1.12910593e-01,
       5.67728173e+00, 7.68341710e+01, 4.16576521e+00, 4.31592132e-01,
       4.18537087e+01, 1.85477670e-01, 2.51571407e+00, 7.79720472e+00,
       4.94162689e+04, 2.60085626e+02, 4.62735486e+01, 3.14271598e+01,
       5.42412945e+01, 1.06739526e+02, 7.67049048e+01, 3.52953506e+00,
       2.61105222e+02, 3.45579228e+01, 2.47722226e+01, 3.82976084e+03,
       7.65952167e+06, 2.61113589e+00, 4.01036775e+00, 6.53464189e+00,
       7.65952167e+06, 6.83092988e+02, 1.11459861e+03, 2.97226297e+03,
       9.54734730e+00, 1.39963850e+02, 1.99046846e+02, 1.48645162e+01,
       1.10895058e+02, 8.36408026e-02, 2.62485276e-01]