import numpy as np
import pandas as pd

from old_scripts.csv_stats import read_csv
from feature_engine.creation import CyclicalFeatures

def dataframe_to_floats(df):
    df_float = df.copy()

    for column in df_float.columns:
        df_float[column] = pd.to_numeric(df_float[column], errors='coerce')

    return df_float
def remove_outliers_iqr(df, columns, boolean_print=False) -> pd.DataFrame:
    """
    Odstráni outliery pomocou IQR metódy pre vybrané stĺpce
    """
    df_clean = df.copy()

    for column in columns:
        Q1 = df_clean[column].quantile(0.25)
        Q3 = df_clean[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Odstráni outliery z dataframeu
        before = len(df_clean)
        df_clean = df_clean[(df_clean[column] >= lower_bound) & (df_clean[column] <= upper_bound)]
        after = len(df_clean)
        if boolean_print:
            print(f"{column}: Odstránených {before - after} outlierov")

    return df_clean
'''
Preprocess data for a tree, leaving the time data (day, month..) as is, because tree based models
wouldn't handle them if it was mapped cyclically or RBF encoded
'''
def preprocess_data_for_tree()-> pd.DataFrame:
    #Open csv and convert do DataFrame
    raw_data = read_csv('z2_data_1y.csv')
    df1 = pd.DataFrame(raw_data)

    #Remove instant and date columns, we do not need them
    df1 = df1.drop('instant', axis = 1)
    df1 = df1.drop('date', axis = 1)


    #Mapping booleans to int type + handling missing values from 'holiday' column
    #Only there are missing values
    df1['workingday'] = df1['workingday'].map({'True': True, 'False': False}).astype(int)
    df1 = df1.replace('', np.nan)
    df1['holiday'] = df1['holiday'].map({'1.00': 1, '0.00': 0, '': None})
    df1 = df1.dropna(subset=['holiday'])
    #Encode weather column with ordinal encoding
    weather_mapping = {
        'clear': 0,
        'cloudy' : 1,
        'light rain/snow' : 2,
        'heavy rain/snow' : 3
    }

    df1['weather'] = df1['weather'].map(weather_mapping)
    df1 = dataframe_to_floats(df1)

    #Removed outliers, about 4% of data deleted
    df1 = remove_outliers_iqr(df1, ['temperature','humidity','windspeed','count'], False)


    return df1

def preprocess_data_with_cyclical() -> pd.DataFrame:
    df1= preprocess_data_for_tree()
    #Cyclical encoding for month, weekday, hour
    cyclical_features = CyclicalFeatures(
        variables=['month', 'weekday', 'hour'],
        drop_original=True
    )
    df1 = cyclical_features.fit_transform(df1)
    return df1

if __name__ == "__main__":
    pass

