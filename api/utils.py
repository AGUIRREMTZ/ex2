"""
Utility functions for data processing and dataset handling.
Based on the NSL-KDD dataset processing notebooks.
"""
import arff
import pandas as pd
from sklearn.model_selection import train_test_split


def load_kdd_dataset(data_path):
    """
    Load NSL-KDD dataset from ARFF format.
    
    Args:
        data_path (str): Path to the ARFF file
        
    Returns:
        pd.DataFrame: Dataset as pandas DataFrame
    """
    with open(data_path, 'r') as train_set:
        dataset = arff.load(train_set)
        attributes = [attr[0] for attr in dataset["attributes"]]
    return pd.DataFrame(dataset["data"], columns=attributes)


def train_val_test_split(df, rstate=42, shuffle=True, stratify=None):
    """
    Split dataset into train, validation, and test sets.
    
    Args:
        df (pd.DataFrame): Input dataset
        rstate (int): Random state for reproducibility
        shuffle (bool): Whether to shuffle data before splitting
        stratify (str): Column name to use for stratified sampling
        
    Returns:
        tuple: (train_set, val_set, test_set) as pandas DataFrames
    """
    print("Dataset length:", len(df))
    strat = df[stratify] if stratify else None
    
    # Split 60% train, 40% temp (for val and test)
    train_set, test_set = train_test_split(
        df, test_size=0.4, random_state=rstate, shuffle=shuffle, stratify=strat
    )
    
    # Split temp into 50% val, 50% test (20% each of original)
    strat = test_set[stratify] if stratify else None
    val_set, test_set = train_test_split(
        test_set, test_size=0.5, random_state=rstate, shuffle=shuffle, stratify=strat
    )
    
    return (train_set, val_set, test_set)


def get_dataset_info(df):
    """
    Get basic information about the dataset.
    
    Args:
        df (pd.DataFrame): Input dataset
        
    Returns:
        dict: Dictionary with dataset information
    """
    return {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.astype(str).to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'memory_usage': df.memory_usage(deep=True).sum(),
    }
