import pandas as pd 


def load_data(file_path):
    """Load data from a CSV file."""
    return pd.read_csv(file_path)
def preprocess_data(df):
    """Preprocess the data."""

    df = df.dropna()  # Drop rows with missing values
    df['date'] = pd.to_datetime(df['date'])  # Convert date column to datetime
    df = df.sort_values(by='date')  # Sort by date
    return df
def main():
    # Load and preprocess data
    data = load_data('aus_weather.csv')
    data = preprocess_data(data)
    
    # Display the first few rows of the preprocessed data
    print(data.head())
    print("Data preprocessing complete.")