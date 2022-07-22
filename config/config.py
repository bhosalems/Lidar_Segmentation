class cfg:
    PATH_TRAIN = r'/home/csgrad/mbhosale/Lidar_Segmentation/pandaset-devkit/data/PandaSet/train/'
    PATH_VALID = r'/home/csgrad/mbhosale/Lidar_Segmentation/pandaset-devkit/data/PandaSet/valid/'
    sub_grid_size = 0.04
    num_points = 40960  # Number of input points
    sub_grid_size = 0.04  # preprocess_parameter

    train_steps = 200  # Number of steps per epochs
    val_steps = 100    # Number of validation steps per epoch

    sampling_type = 'active_learning'
    data_name = 'pandaset'
    if data_name == 's3dis':
        class_weights = [1938651, 1242339, 608870, 1699694, 2794560, 195000, 115990, 549838, 531470, 292971, 196633, 59032, 209046, 39321]
    else:
        class_weights = [19500] * 43