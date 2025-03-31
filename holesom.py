import numpy as np
import matplotlib.pyplot as plt
from collections import Counter  
import random
from matplotlib import cm
import pandas as pd

import joblib
import sys
sys.modules['sklearn.externals.joblib'] = joblib
import sompy
import pickle 
from sompy.sompy import SOMFactory



def slopes(lum1, lum2, freq1, freq2): 
    return (np.log10(lum2/lum1)/np.log10(freq2/freq1))

"SOM size"
ms0 = 28
ms1 = 18

# Read trained SOM
file = open('SOM_MBHs_2025.pickle', 'rb')
sm = pickle.load(file)
file.close()

# Read cell median of Mbh and fedd
mbh_median = np.load("mbh_median.npy")
fedd_median = np.load("fedd_median.npy")

# Name of input feature columns
name_features = [r'Log($\nu$L$_{5GHz}$)', r'Log($\nu$L$_{50GHz}$)', r'log($\nu$L$_{230GHz}$)',r'5GHz-950GHz',r'950GHz-25.5$\mu$m',
r'25.5$\mu$m-8.6$\mu$m', r'8.6$\mu$m-F150W', r'F150W-0.4$\mu$m', r'0.4$\mu$m-0.17$\mu$m', r'L$_X$']

# Load data
features_df = [] # working on this

# Recover for missing observations
def new_sed_wmissing(t_id, N_mc, name_column):
    """
    Generate new SEDs by filling in missing color values using random selections 
    from the dataset.

    Parameters:
    ----------
    t_id : int or index label
        Identifier of the target object in `features_df`.
    N_mc : int
        Number of Monte Carlo samples to generate.
    name_column : list
        List of column names corresponding to missing color values.

    Returns:
    -------
    df_new : pd.DataFrame
        DataFrame containing N_mc samples of the object with missing values 
        replaced by randomly selected values from the dataset.

    Notes:
    ------
    - If only one color is missing, it selects values from that column.
    - If multiple colors are missing, it selects values from the corresponding 
      stacked columns while maintaining consistency across samples.
    """
    
    # Retrieve the object's features
    d_object = features_df.loc[t_id]

    # Create a new DataFrame with N_mc copies of the object
    df_new = pd.DataFrame([d_object] * N_mc)  
    np.random.seed(42)  # Set seed for reproducibility
    
    if len(name_column) == 1:
        # Single missing color: Randomly select values from the dataset
        name_column = name_column[0]
        new_color = np.random.choice(features_df[name_column], size=N_mc, replace=False)
    else:
        # Multiple missing colors: Select corresponding values from multiple columns
        stack_colors = np.column_stack([features_df[col] for col in name_column])
        indices = np.random.choice(len(stack_colors), size=N_mc, replace=False)
        new_color = stack_colors[indices]

    # Assign the selected values to the missing columns
    df_new[name_column] = new_color

    return df_new


# Project data onto the SOM map and get PDF

def project_and_estimate(t_id, N_mc, name_column, features_df, sm, ms0, ms1, fedd_median, mbh_median, output_file):
    """
    Projects an object onto a trained SOM, recovers missing data if necessary, 
    computes the likelihood distribution, and estimates the posterior distributions 
    for Eddington ratio and black hole mass.

    Parameters:
    ----------
    t_id : int or index label
        Identifier of the test object in `features_df`.
    N_mc : int
        Number of Monte Carlo samples for missing data recovery or for addressing photometric errors.
    name_column : list
        List of missing color column names. If empty, no missing data is assumed.
    features_df : pd.DataFrame
        DataFrame containing the full dataset with photometric features.
    sm : SOM object
        Trained Self-Organizing Map for projection.
    ms0, ms1 : int
        Dimensions of the SOM grid.
    fedd_median : np.ndarray
        2D array of median Eddington ratios mapped to the SOM.
    mbh_median : np.ndarray
        2D array of median black hole masses mapped to the SOM.
    output_file : str
        Path to save the posterior distribution.

    Returns:
    -------
    edd_ratio : list
        [median, lower bound, upper bound, standard deviation] of Eddington ratio.
    mass : list
        [median, lower bound, upper bound, standard deviation] of black hole mass.
    """

    np.random.seed(42)

    # Recover missing data if necessary
    if name_column:  # If there are missing values
        df_new = new_sed_wmissing(t_id, N_mc, name_column)
    else:
        d_object = features_df.loc[t_id]
        df_new = pd.DataFrame([d_object] * N_mc)

    # Project dataset onto the trained SOM
    projection_newset = sm.bmu_ind_to_xy(sm.project_data(df_new))
    xi, yj = projection_newset[:, 0], projection_newset[:, 1]

    # Compute likelihood distribution
    likelihood = np.zeros((ms0, ms1))
    for xcord, ycord in zip(xi, yj):
        likelihood[xcord, ycord] += 1
    likelihood_idx = likelihood / N_mc  # Normalize likelihood

    # Compute posterior distributions
    def compute_posterior(median_map, goodpoints, binning=8):
        """Computes the posterior distribution for a given parameter."""
        param_values = median_map[goodpoints]
        weights = np.ones_like(param_values) / len(param_values)
        _, _ = np.histogram(param_values, bins=binning, weights=weights)
        error = np.std(param_values)
        median_val = np.median(param_values)
        return [median_val, median_val - error, median_val + error, error]

    # Identify valid points in the likelihood map
    goodpoints_h = np.isfinite(np.log10(likelihood_idx))

    # Compute posteriors for Eddington ratio and mass
    edd_ratio = compute_posterior(fedd_median, goodpoints_h)
    mass = compute_posterior(mbh_median, goodpoints_h)

    # # Save posterior distributions
    # np.savez(output_file, likelihood=likelihood_idx, edd_ratio=edd_ratio, mass=mass)

    return edd_ratio, mass


# Compute BMU distance
def classify_sed(new_sed, codebook_vectors_weights):
    """
    Find the best matching unit (BMU) for a new SED.
    :param new_sed: A 10-point SED (1D numpy array)
    :param codebook_vectors_weights: Matrix of SOM neuron prototypes
    :return: 1) Index of the best matching neuron,
             2) Euclidean distances between MBH object and all cells
             3) distance at BMU cell 
    """
    distances = np.sqrt(np.sum((codebook_vectors_weights - new_sed) ** 2, axis=1)) #Euclidean distance
    bmu_index = np.argmin(distances)  # Best Matching Unit (BMU)
    return bmu_index, distances, np.min(distances)

