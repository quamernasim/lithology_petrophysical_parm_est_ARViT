import numpy as np
import pandas as pd
from os.path import join as pjoin

import torch

from sklearn.model_selection import train_test_split

from datasets.utils import create_patches, scale_data

def prepare_data(config, test=False, scaler_save=True):
    """
    Prepare the data for training and validation
    It loads the data from the processed data folder and creates patches if the data is patch based
    Then it splits the data into train and validation
    It also scales the data

    Parameters
    ----------
    config : dict
        The config dictionary
    test : bool, optional
        If True, the data is not split into train and validation, by default False

    Returns
    -------
    x_train : torch.tensor
        The training data
    x_val : torch.tensor
        The validation data
    y_train : torch.tensor
        The training labels
    y_val : torch.tensor
        The validation labels
    num_classes : int
        The number of classes in the data
    """
    
    print("Preparing the data...")

    # Load the parameters from the config file
    data_config = config['data']
    root = config['root']

    data_config['x_file_name'] = data_config['x_file_name'] if not config['model']['use_lora'] else 'lora_' + data_config['x_file_name']
    data_config['y_file_name'] = data_config['y_file_name'] if not config['model']['use_lora'] else 'lora_' + data_config['y_file_name']

    # Load the path to the processed data
    x_path = pjoin(root, data_config['processed_data_path'], data_config['x_file_name'])
    y_path = pjoin(root, data_config['processed_data_path'], data_config['y_file_name'])

    # Load the data from hdf5 files
    X = pd.read_hdf(x_path)
    Y = pd.read_hdf(y_path)

    # Get the unique well names
    well_names = X.UWI.unique()

    # Drop the columns that are not needed in the training
    X.drop(data_config['drop_columns'], axis=1, inplace=True)

    # Scale the data
    X = scale_data(X, config, test=test, save=scaler_save)

    # check if the data is patch based, if yes, create patches else model will be trained on the point data
    if data_config['patch_based']:

        # Create patches from the data
        num_features = X.shape[1] - 1
        x_patches, y_patches = create_patches(X, Y, well_names, data_config, num_features)

        # Get the number of classes
        num_classes = len(np.unique(y_patches[:, :, 0]))

        if not test:
            # Split the data into train and validation
            (
                x_train, 
                x_val, 
                y_train, 
                y_val
            ) = train_test_split(
                x_patches, 
                y_patches, 
                test_size=data_config['split_size'], 
                random_state=config['random_state']
            )

            # Convert the data to PyTorch tensors
            x_train = torch.tensor(x_train).float()
            y_train = torch.tensor(y_train)
            x_val = torch.tensor(x_val).float()
            y_val = torch.tensor(y_val)
        else:
            x_train = torch.tensor(x_patches).float()
            y_train = torch.tensor(y_patches)
            x_val = None
            y_val = None
    else:
        # drop the well name column
        X = X.drop([data_config['well_name_column']], 
                   axis=1)
        
        # Get the number of classes
        num_classes = len(np.unique(Y))

        # Split the data into train and validation
        (
            x_train, 
            x_val, 
            y_train, 
            y_val
        ) = train_test_split(
            X, 
            Y, 
            test_size=data_config['split_size'], 
            random_state=config['random_state']
        )

    print(f"Number of classes: {num_classes} and shape of x_train: {x_train.shape}")
    return x_train, x_val, y_train, y_val, num_classes
