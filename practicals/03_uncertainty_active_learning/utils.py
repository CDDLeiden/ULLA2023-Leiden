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
    
def variance_distribution_plot(variance, title):
    
        fig, ax = plt.subplots(figsize=(4, 4))
    
        ax.hist(variance, bins=50, density=True, ec='k', histtype='stepfilled'  )
        ax.axvline(variance.mean(), color='k', linestyle='--', linewidth=2)
        ax.set_xlabel('Variance')
        ax.set_ylabel('Density')
        ax.set_title(title)
        ax.grid(linestyle='--', alpha=0.5)
    
def correlation_plot(x, y, y_err, title):

    fig, ax = plt.subplots(figsize=(4, 4))

    xmax = max(x.max(), y.max()) +1
    xmin = min(x.min(), y.min()) -1
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(xmin, xmax)
    ax.grid(linestyle='--', alpha=0.5)
    ax.set_aspect('equal')

    ax.errorbar(x, y, yerr=y_err, fmt='o', mec='k', ecolor='k', elinewidth=1, capsize=0)
    ax.plot([xmin, xmax], [xmin, xmax], color='k', linestyle='--', linewidth=2)

    ax.set_xlabel('True value')
    ax.set_ylabel('Predicted value')
    ax.set_title(title)

    
