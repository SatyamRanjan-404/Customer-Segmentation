"""
Customer segmentation based on RFM method and K-Means clustering.
Model: RFM scoring + K-Means clustering
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

URL = r'Data\sales_data_sample.csv'


def read_data(path):
    '''Read and preprocess data'''
    df = pd.read_csv(path, encoding = 'unicode_escape')
    # Select relevant columns
    col =['CUSTOMERNAME', 'ORDERNUMBER', 'ORDERDATE', 'SALES']
    # Create a new DataFrame with selected columns
    data = df[col]
    # Convert 'ORDERDATE' to datetime
    data['ORDERDATE'] = pd.to_datetime(data['ORDERDATE'], errors='coerce')
    return data


def create_rfm_table(df, reference_date):
    '''Create the RFM Table: calculate recency, frequency, and monetary value'''
    rfm_df = df.groupby('CUSTOMERNAME').agg({'ORDERDATE': lambda x: (reference_date - x.max()).days,
                                             'ORDERNUMBER': 'count',
                                             'SALES': 'sum'})
    rfm_df.rename(columns={'ORDERDATE': 'Recency', 'ORDERNUMBER': 'Frequency', 'SALES': 'MonetaryValue'}, inplace=True)
    return rfm_df


def calculate_rfm_scores(df):
    '''Calculate RFM scores'''
    # Apply the percentiles of the distribution of the given variable to calculate RFM scores
    r = pd.qcut(df.Recency, 4, labels = list(range(0,4)))
    f = pd.qcut(df.Frequency, 4, labels = list(range(0,4)))
    m = pd.qcut(df.MonetaryValue, 4, labels = list(range(0,4)))
    rfm_df_cutted = pd.DataFrame({'Recency' : r, 'Frequency' : f, 'MonetaryValue' : m})
    return rfm_df_cutted

