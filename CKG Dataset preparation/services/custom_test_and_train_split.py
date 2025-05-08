import pandas as pd
import numpy as np
import random
# Assuming df is your DataFrame and 'research_field_column' is the column with research fields
# Replace 'research_field_column' with the actual name of your column


def custom_split_df(df, test_size=150, random_seed=None):
    """
    Split a DataFrame into training and testing sets.

    Parameters:
    - df: The input DataFrame.
    - test_size: The number of samples to include in the test set (default is 100).
    - random_seed: Seed for the random number generator (optional).

    Returns:
    - train_data: The training set (DataFrame).
    - test_data: The testing set (DataFrame).
    """
    # Set the seed for reproducibility
    np.random.seed(random_seed)

    # Randomly select test_size indices for the test set
    test_indices = np.random.choice(df.index, size=test_size, replace=False)

    # Use the remaining indices for the training set
    train_indices = df.index.difference(test_indices)

    # Select the corresponding rows from the original DataFrame
    train_data = df.loc[train_indices]
    test_data = df.loc[test_indices]

    return train_data, test_data

# def test_and_train_split(df, research_field_label):
#     # Shuffle the DataFrame
#     df_shuffled = df.sample(frac=1, random_state=42)
#
#     # Get unique values in the research field column for splitting
#     unique_research_fields = df_shuffled[research_field_label].unique()
#
#     # Initialize empty lists to store train and test DataFrames
#     train_dfs = []
#     test_dfs = []
#
#     # Split the DataFrame based on unique research fields
#     for field in unique_research_fields:
#         # Select rows where the column has the specific research field
#         subset_df = df_shuffled[df_shuffled[research_field_label] == field]
#
#         # Split the subset into train (80%) and test (20%)
#         train_subset, test_subset = custion_split_df(subset_df)
#
#         # Append to the lists
#         train_dfs.append(train_subset)
#         test_dfs.append(test_subset)
#
#     # Concatenate the DataFrames to get the final train and test sets
#     train_set = pd.concat(train_dfs)
#     test_set = pd.concat(test_dfs)
#
#     # Resetting index for consistency
#     train_set.reset_index(drop=True, inplace=True)
#     test_set.reset_index(drop=True, inplace=True)
#
#     # Now, train_set and test_set contain the 80% and 20% splits, respectively, based on research fields
#     return train_set, test_set
