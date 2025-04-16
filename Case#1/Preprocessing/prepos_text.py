# from Sastrawi.Tokenizer import Tokenizer
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from util import yield_tokens, nltk_tokenizer, vocabularize

factory = StemmerFactory()
stemmer = factory.create_stemmer()

indonesia_stopwords = stopwords.words('indonesian')

def remove_stopwords(tokens):
    """
    Removes stopwords from the list of tokens using nltk stopword remover.
    
    Args:
        tokens (list): A list of tokens.
    
    Returns:
        list: A list of tokens with stopwords removed.
    """
    # Hapus stopwords dengan mengecek apakah token ada di daftar stopwords
    filtered_tokens = [token for token in tokens if token not in indonesia_stopwords]
    
    return filtered_tokens


def stem_tokens(tokens):
    """
    Stems the list of tokens using Sastrawi stemmer.
    
    Args:
        tokens (list): A list of tokens.
    
    Returns:
        list: A list of stemmed tokens.
    """
    # Gabungkan kembali tokens menjadi satu kalimat
    sentence = ' '.join(tokens)
    
    # Lakukan stemming pada kalimat
    stemmed_sentence = stemmer.stem(sentence)
    
    # Tokenize kembali kalimat hasil stemming
    stemmed_tokens = stemmed_sentence.split()
    
    return stemmed_tokens


def lowercase_text(text):
    """
    Converts all characters in the given text to lowercase.

        text (str): The original string.

        str: The string with all characters converted to lowercase.
    """
    return text.lower()

def clean_text(text, URL=True, numbers=True, symbols=True):
    """
    Cleans the text by removing URLs, numbers, symbols, and emojis (without converting to lowercase).

    Args:
        text (str): The original text.
        URL (bool): Whether to remove URLs.
        numbers (bool): Whether to remove numbers.
        symbols (bool): Whether to remove symbols.

    Returns:
        str: The cleaned text.
    """

    # Replace <USER_MENTION> and <PROVIDER_NAME> with appropriate placeholders
    text = re.sub(r'<user_mention>', 'otheruser', text)
    text = re.sub(r'<provider_name>', 'providername', text)
    
    # Remove URLs
    if URL:
        # Remove URLs using regex
        text = re.sub(r'http[s]?://\S+', '', text)
    
    # Remove mentions and hashtags
    text = re.sub(r'@\w+|#\w+', '', text)
    
    # Remove numbers
    if numbers:
        # Remove numbers using regex
        text = re.sub(r'\d+', '', text)

    # Remove symbols
    if symbols:
        # Remove symbols using regex
        text = re.sub(r'[^a-zA-Z\s]', '', text)
 
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def preprocess_dataframe(df, 
                         remove_special_char=True,
                         remove_stopword=True, 
                         apply_stemming=True,
                         URL=True,
                         numbers=False,
                         symbols=False):
    """
    Preprocess a DataFrame, CSV file path, or list of text with various preprocessing options.
    
    Args:
        df (Union[pd.DataFrame, str, list]): Input data. Can be a DataFrame, CSV file path, or list of strings.
        remove_special_char (bool): Remove special characters such as emojis, URLs, symbols.
        remove_stopword (bool): Remove Indonesian stopwords.
        apply_stemming (bool): Apply stemming using Sastrawi.
        URL (bool): Whether to remove URLs.
        numbers (bool): Whether to remove numbers.
        symbols (bool): Whether to remove symbols.

    Returns:
        pd.DataFrame: A new DataFrame with an additional column 'processed_text'.
    """

    processed_texts = []
    # Handle different input types
    if isinstance(df, str) and df.endswith('.csv'):
        df = pd.read_csv(df)
    elif isinstance(df, list) and all(isinstance(item, str) for item in df):
        df = pd.DataFrame({'Text Tweet': df})
    elif isinstance(df, str) and not df.endswith('.csv'):
        df = pd.DataFrame({'Text Tweet': [df]})
    elif not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a DataFrame, CSV file path, or a list of strings.")
    
    for text in df["Text Tweet"]:
        # 1. Convert text to lowercase
        text = lowercase_text(text)
        # 2. Optional: Clean special characters
        if remove_special_char:
            text = clean_text(text, URL=URL, numbers=numbers, symbols=symbols)
        # 3. Tokenize the text
        tokens = nltk_tokenizer(text)

        # 4. Optional: Remove stopwords
        if remove_stopword:
            tokens = remove_stopwords(tokens)

        # 5. Optional: Apply stemming
        if apply_stemming:
            tokens = stem_tokens(tokens)

        # Combine tokens back into a sentence
        processed = ' '.join(tokens)
        processed_texts.append(processed)

    # Add a new column to the DataFrame
    df = df.copy()
    df["processed_text"] = processed_texts

    return df
