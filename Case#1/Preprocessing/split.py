from sklearn.model_selection import train_test_split

# Split dataset menjadi train dan test
def split_dataset(texts, labels, test_size=0.2):
    """
    Splits the dataset into training and testing sets.
    
    Args:
        texts (list): A list of text data.
        labels (list): A list of corresponding labels.
    
    Returns:
        tuple: A tuple containing the training and testing sets for texts and labels.
    """
    # Split data menjadi 80% untuk training dan 20% untuk testing
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test

from sklearn.model_selection import train_test_split
from typing import List, Tuple

def split_dataset_stratisfied(
    texts: List[str],
    labels: List[int],
    test_size: float = 0.2
) -> Tuple[List[str], List[str], List[int], List[int]]:
    """
    Splits the dataset into stratified training and testing sets.

    Args:
        texts (List[str]): A list of text inputs.
        labels (List[int]): A list of corresponding class labels.
        test_size (float): Proportion of the dataset to include in the test split.

    Returns:
        Tuple: A tuple (X_train, X_test, y_train, y_test) with stratified splits.
    """
    # Stratified split: maintain class distribution across train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=test_size,
        random_state=42,
        stratify=labels 
    )
    return X_train, X_test, y_train, y_test
