from sklearn.model_selection import train_test_split
from typing import List, Tuple
import pandas as pd

def split_dataset(
    texts: List[str],
    labels: List[int],
    test_size: float = 0.2, Stratify = False
) -> Tuple[List[str], List[str], List[int], List[int]]:
    """
    Splits the dataset into stratified training and testing sets and saves them as CSV files.

    Args:
        texts (List[str]): A list of text inputs.
        labels (List[int]): A list of corresponding class labels.
        test_size (float): Proportion of the dataset to include in the test split.
        Stratify (bool): Whether to stratify the split based on labels.

    Returns:
        Tuple: A tuple (X_train, X_test, y_train, y_test) with stratified splits.
    """
    # Stratified split: maintain class distribution across train and test sets
    if Stratify:
        X_train, X_test, y_train, y_test = train_test_split(
            texts,
            labels,
            test_size=test_size,
            random_state=42,
            stratify=labels 
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            texts,
            labels,
            test_size=test_size,
            random_state=42
        )
    
    # Save train and test data to CSV files
    train_data = pd.DataFrame({'data': X_train, 'labels': y_train})
    test_data = pd.DataFrame({'data': X_test, 'labels': y_test})
    
    train_data.to_csv('train.csv', index=False)
    test_data.to_csv('test.csv', index=False)
    
    return X_train, X_test, y_train, y_test
