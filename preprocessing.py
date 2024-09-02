# preprocessing.py
import pandas as pd

def preprocess_data(df):
    """
    Preprocess the data by engineering features and dropping unnecessary columns.
    """
    try:
        df['X'] = df['X_Maximum'] - df['X_Minimum']
        df['Y'] = df['Y_Maximum'] - df['Y_Minimum']
        df['Luminosity'] = df['Maximum_of_Luminosity'] - df['Minimum_of_Luminosity']
        df['Area_Perimeter_Ratio'] = df['Pixels_Areas'] / (df['X_Perimeter'] + df['Y_Perimeter'])
        
        # Drop original columns
        df = df.drop([
            'X_Maximum', 'X_Minimum', 'Y_Maximum', 'Y_Minimum',
            'Maximum_of_Luminosity', 'Minimum_of_Luminosity',
            'Pixels_Areas', 'X_Perimeter', 'Y_Perimeter'
        ], axis=1)
    except KeyError as e:
        print(f"Missing column: {e}")
    
    return df

def load_data(train_path, test_path):
    """
    Load the training and testing data from specified paths.
    """
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    return df_train, df_test
