# easy function to load the data
import pandas as pd


def load_dataset(path='../../data/Chapter_1_cleaned_data.csv'):
    return pd.read_csv(path)
