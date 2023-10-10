import numpy as np
from tqdm import tqdm
import patchify as pf
import joblib
from os.path import join as pjoin
from utils.misc import add_to_config
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def create_patches(X, Y, well_names, data_config, num_features):
    """
    Create patches from the well data
    
    Parameters
    ----------
    X : pandas.DataFrame
        The input data
    Y : pandas.DataFrame
        The target data
    well_names : list
        The list of well names
    data_config : dict
        The data configuration
    num_features : int
        The number of features in the data

    Returns
    -------
    x_patches : numpy.ndarray
        The input patches
    y_patches : numpy.ndarray
        The target patches
    """

    # get parameters of patch
    patch_size = data_config['patch']['patch_size']
    stride = data_config['patch']['stride']
    well_name_column = data_config['well_name_column']
    x_patches, y_patches = [], []

    # iterate over wells
    for well in tqdm(well_names, desc='Creating Patches'):

        # get data of each well
        well_x = X[X[well_name_column] == well]
        well_y = Y.loc[well_x.index]

        # drop well name column
        well_x = well_x.drop([well_name_column], axis=1)

        # create patches if well has data more than patch size
        if well_x.shape[0] >= patch_size:

            # create patches
            well_x_patches = pf.patchify(well_x.values, (patch_size, num_features), step=stride)
            well_x_patches = well_x_patches.squeeze()

            well_y_patches = pf.patchify(well_y.values, (patch_size, well_y.values.shape[-1]), step=stride)
            well_y_patches = well_y_patches.squeeze()

            if well_x_patches.ndim == 3:
                for i, j in zip(well_x_patches, well_y_patches):
                    x_patches.append(i)
                    y_patches.append(j)
                    
            #if well has only one patch
            else:
                x_patches.append(well_x_patches)
                y_patches.append(well_y_patches)

    return np.array(x_patches), np.array(y_patches)

def scale_data(X, config, test=False, save=True):
    """
    Scale the data using StandardScaler or MinMaxScaler based on the column name
    Also saves the scaler for each column if test is False, else loads the scaler

    Parameters
    ----------
    X : pandas.DataFrame
        The input data
    config : dict
        The configuration
    test : bool, optional
        Whether to test or not, by default False

    Returns
    -------
    pandas.DataFrame
        The scaled data
    """
    
    # if save:
    exp_path = config['trainer']['experiment_path']
    scaler_folder = pjoin(exp_path, config['trainer']['scaler_path'])

    data_config = config['data']
    cols_to_be_scaled = data_config['scaled_columns']

    for col in cols_to_be_scaled:
        if test:
            # load scaler
            print(f"Loading scaler for {col}...")
            scaler = joblib.load(pjoin(scaler_folder, f'scaler_{col}.bin'))
        else:
            # define scaler
            print(f"Creating scaler for {col}...")
            if col in ['lat', 'lng', 'DEPT']:
                scaler = MinMaxScaler(feature_range=(0, 1))
            else:
                scaler = StandardScaler()

            # fit scaler
            scaler.fit(X[col].values.reshape(-1, 1))

            # save scaler
            if save:
                joblib.dump(scaler, pjoin(scaler_folder, f'scaler_{col}.bin'), compress=True)
        
        # scale data
        X[col] = scaler.transform(X[col].values.reshape(-1, 1))

    return X