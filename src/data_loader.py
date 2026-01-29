import numpy as np
import h5py

#to load and preprocess Lyman-alpha forest data for machine learning applications.
def load_data(path0, path1):
    """
    Load and preprocess Lyman-alpha forest data from given file paths.

    Parameters:
    path0 (str): Path to the first data file.
    path1 (str): Path to the second data file.

    Returns:
    tuple: Preprocessed data arrays.
    """
    # Load data from the first file
    with h5py.File(path0, 'r') as f:
        data0 = f['/tau/H/1/1215'][:]
    f.close()

    # Load data from the second file
    with h5py.File(path1, 'r') as f:
        data1 = f['/tau/H/1/1215'][:]
    f.close()

    #Combine the data and create labels
    X = np.concatenate((data0, data1), axis=0)
    y = np.concatenate((np.zeros(data0.shape[0]), np.ones(data1.shape[0])))
    
    return X, y

if __name__ == "__main__":
    path0 = 'data/raw/snapshot_028_EX3.hdf5' #'data/raw/EX0_spectra.hdf5'
    path1 = 'data/raw/snapshot_028_EX1.hdf5' #'data/raw/EX1_spectra.hdf5'
    X, y = load_data(path0, path1)
    #Check the shapes of the loaded data
    print("Data shape:", X.shape)
    print("Labels shape:", y.shape)