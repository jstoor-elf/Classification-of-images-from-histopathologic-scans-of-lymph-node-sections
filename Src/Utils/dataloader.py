import h5py
import pandas as pd
import numpy as np
import os

ABS_PATH = './dl_data/'
DIR_NAME = 'Camelyon'



def get_h5_data(path, file_name):

    ''' Reads in a given .h5 file and converts it to a numpy array '''

    print('\t {} ... '.format(file_name), end='')
    f_in = os.path.join(path, file_name)
    with h5py.File(f_in, 'r') as f:
        a_group_key = list(f.keys())[0] # List all groups
        images = list(f[a_group_key]) # Get and return the data
        print('Done')
        return np.asarray(images)


def get_metadata(path, file_name):

    ''' Reads in csv-metadata as a pandas dataframe '''

    return pd.read_csv(os.path.join(path, file_name))


def load_data(data_type):

    ''' Reads in train, valid (validation), or test data '''

    dirname = os.path.join(ABS_PATH, DIR_NAME)

    try:

        if data_type == 'train':
            x_name = 'camelyonpatch_level_2_split_train_x.h5'
            y_name = 'camelyonpatch_level_2_split_train_y.h5'
            m_name = 'camelyonpatch_level_2_split_train_meta.csv'
        elif data_type == 'valid':
            x_name = 'camelyonpatch_level_2_split_valid_x.h5'
            y_name = 'camelyonpatch_level_2_split_valid_y.h5'
            m_name = 'camelyonpatch_level_2_split_valid_meta.csv'
        elif data_type == 'test':
            x_name = 'camelyonpatch_level_2_split_test_x.h5'
            y_name = 'camelyonpatch_level_2_split_test_y.h5'
            m_name = 'camelyonpatch_level_2_split_test_meta.csv'
        else:
            raise NotImplementedError('Incorrect input type!')

        image_data = get_h5_data(dirname, x_name)
        label_data = get_h5_data(dirname, y_name)
        meta_data  = get_metadata(dirname, m_name)

    except OSError:
        raise NotImplementedError('Cannot read in the h5 file!')

    return (image_data.astype("float32"), label_data.squeeze(), meta_data)


def load_data_as_dict(data_types=None):

    ''' Reads in data as a dictionary. data_type is a list with string keys that
        can be of type "train", "valid", or "test". If None is specified, all types
        are read in.
    '''

    if data_types is None:
        data_types = ['train', 'valid', 'test']

    print("Reading in data: ")

    data = {}
    for data_type in data_types:
        (image_data, label_data, meta_data) = load_data(data_type)
        data[data_type] = {'images' : image_data, 'labels' : label_data, 'meta' : meta_data}

    return data
