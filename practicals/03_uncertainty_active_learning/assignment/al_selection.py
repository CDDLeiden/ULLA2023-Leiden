import numpy as np
import pandas as pd

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

# Random selection #######################################################################################################

def random_selection(data, nmolecules = 20):
    
        """
        Function to select molecules randomly from the dataset.
    
        Parameters
        ----------
        data : pandas DataFrame
            DataFrame containing the molecules
        nmolecules : int
            Number of molecules to select
    
        Returns
        -------
        data : pandas DataFrame
            Updated DataFrame containing the molecules
        """

        data_learn = data[data.Subset == 'Learn']
    
        # Select molecules randomly
        if nmolecules < len(data_learn):
            idx = np.random.choice(len(data_learn), nmolecules, replace=False)
            selected_smiles_list = data_learn.iloc[idx]['SMILES'].values
        else:
            selected_smiles_list = data_learn['SMILES'].values
    
        # Update training set
        data.loc[data.SMILES.isin(selected_smiles_list), 'Subset'] = 'Train'
    
        return data



# Uncertainty selection - exploration ####################################################################################

def uncertainty_selection(data, y_uncertainty, nmolecules = 20):

    """
    Function to select molecules with the highest uncertainty from the dataset.

    Parameters
    ----------
    data : pandas DataFrame
        DataFrame containing the molecules
    y_uncertainty : numpy array
        Array containing the uncertainty for the learning set
    nmolecules : int
        Number of molecules to select

    Returns
    -------
    data : pandas DataFrame
        Updated DataFrame containing the molecules
    """

    data_learn = data[data.Subset == 'Learn']

    # Add uncertainty to DataFrame
    data_learn.loc[data_learn.Subset == 'Learn', 'Uncertainty'] = y_uncertainty

    # Select molecules with the highest uncertainty
    if nmolecules < len(data_learn):
        idx = np.argsort(data_learn['Uncertainty'].values)[::-1] # Sort in descending order
        selected_smiles_list = [data_learn.iloc[i]['SMILES'] for i in idx[:nmolecules]]
    else:
        selected_smiles_list = data_learn['SMILES'].values

    # Update training set
    data.loc[data.SMILES.isin(selected_smiles_list), 'Subset'] = 'Train'

    return data


# Activity selection - exploitation ######################################################################################

def activity_selection(data, y_predictions, nmolecules = 20):

    """
    Function to select molecules with the highest activity from the dataset.

    Parameters
    ----------
    data : pandas DataFrame
        DataFrame containing the molecules
    y_predictions : numpy array
        Array containing the predictions for the learning set
    nmolecules : int
        Number of molecules to select

    Returns
    -------
    data : pandas DataFrame
        Updated DataFrame containing the molecules
    """

    data_learn = data[data.Subset == 'Learn']

    # Add predictions to DataFrame
    data_learn.loc[data_learn.Subset == 'Learn', 'Activity'] = y_predictions

    # Select molecules with the highest activity
    if nmolecules < len(data):
        idx = np.argsort(data_learn['Activity'].values)[::-1] # Sort in descending order
        selected_smiles_list = [data_learn.iloc[i]['SMILES'] for i in idx[:nmolecules]]
    else:
        selected_smiles_list = data_learn['SMILES'].values

    # Update training set
    data.loc[data.SMILES.isin(selected_smiles_list), 'Subset'] = 'Train'

    return data


# Distance based selection ###############################################################################################


def minimum_interset_Tanimoto_distance(smiles_list, reference_smiles_list):

    """
    Function to calculate the minimum interset Tanimoto distance between a list of SMILES strings and a reference list of SMILES strings.
    
    Parameters
    ----------
    smiles_list : list
        List of SMILES strings.
    reference_smiles_list : list
        List of reference SMILES strings.
    
    Returns
    -------
    min_dist : float
        Minimum interset Tanimoto distance.
    """

    # Calculate Morgan fingerprints for both sets of SMILES strings
    fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), 2, nBits=2048) for smiles in smiles_list]
    reference_fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), 2, nBits=2048) for smiles in reference_smiles_list]

    min_dist = np.zeros(len(fps))

    # Calculate Tanimoto distance between each SMILES string and the reference set
    for i, fp in enumerate(fps):
        dist = DataStructs.BulkTanimotoSimilarity(fp, reference_fps)
        min_dist[i] = np.min(dist)

    return min_dist

def distance_selection(data, nmolecules = 20):

    """
    Function to update the training set by selecting the molecules with the highest minimum interset Tanimoto distance.

    Parameters
    ----------
    data : pandas DataFrame
        DataFrame containing the molecules
    nmolecules : int
        Number of molecules to select

    Returns
    -------
    data : pandas DataFrame
        Updated DataFrame containing the molecules
    """

    smiles_list = data[data.Subset == 'Learn']['SMILES'].values
    reference_smiles_list = data[data.Subset == 'Train']['SMILES'].values

    # Calculate minimum interset Tanimoto distance
    dist = minimum_interset_Tanimoto_distance(smiles_list, reference_smiles_list)

    # Select molecules with the highest minimum interset Tanimoto distance
    if nmolecules < len(smiles_list):
        idx = np.argsort(dist)[::-1]
        selected_smiles_list = [smiles_list[i] for i in idx[:nmolecules]]
    else:
        selected_smiles_list = smiles_list

    # Update training set
    data.loc[data.SMILES.isin(selected_smiles_list), 'Subset'] = 'Train'

    return data