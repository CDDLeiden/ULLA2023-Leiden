import os
import torch
import random

import numpy as np
import matplotlib.pyplot as plt

def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def create_dataloader(data, subset, label_col, feature_col, batch_size):

    """ 
    Create a PyTorch DataLoader object from a Pandas DataFrame.

    Parameters
    ----------
    data : pandas.DataFrame
        Dataframe containing the data to be split.
    subset : str
        Subset to be used for the DataLoader.
    label_col : str
        Name of the column containing the labels.
    feature_col : str
        Name of the column containing the features.
    batch_size : int
        Batch size.

    Returns
    -------
    dataloader : torch.utils.data.DataLoader
        DataLoader object.
    """

    # Subset data
    df = data[data['Subset'] == subset]


    # Convert to PyTorch tensors
    features = np.array(df[feature_col].tolist())
    features = torch.tensor(features, dtype=torch.float)
    labels = torch.tensor(df[label_col].values, dtype=torch.float)

    # Create dataset
    dataset = torch.utils.data.TensorDataset(features, labels)

    # Create dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return dataloader