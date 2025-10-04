import pandas as pd 


def load_data(file_path):
    """Load data from a CSV file."""
    return pd.read_csv(file_path)
def preprocess_data(df):
    """Preprocess the data."""