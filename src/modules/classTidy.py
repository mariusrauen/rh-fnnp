from dataclasses import dataclass
import pandas as pd

@dataclass
class Tidy:

    def df_info(self, df: pd.DataFrame):
        '''Function to print information about the processed DataFrames'''
        # Try to find the variable name of the DataFrame using the `globals()` dictionary
        df_name = None
        for name, obj in globals().items():
            if obj is df:
                df_name = name
                break
        # If no variable name is found, default to 'DataFrame'
        df_name = df_name if df_name else "DataFrame"
        
        # Display information about the DataFrame.
        print(f'                               ')
        print('================================')
        #print(df.dtypes)
        print(df.head(2))
        print(df.tail(2))
        print('--------------------------------')
        print(df.shape)
        #print(df.columns)
        print('================================')   
        print(f'                               ')
    
    def standardize_time(self, df: pd.DataFrame, date_column: str) -> pd.DataFrame:
        ''' Function to standardize the time stamps of each DataFrame'''
        # Try different date formats explicitly
        df[date_column] = pd.to_datetime(df[date_column], format='%d/%m/%Y', errors='coerce') \
                            .combine_first(pd.to_datetime(df[date_column], format='%Y-%m-%d', errors='coerce')) \
                            .combine_first(pd.to_datetime(df[date_column], format='%d-%b-%Y', errors='coerce')) \
                            .combine_first(pd.to_datetime(df[date_column], format='%Y-%m-%d %H:%M:%S%z', errors='coerce')) \
                            .combine_first(pd.to_datetime(df[date_column], format='%d.%m.%Y %H:%M', errors='coerce')) \
                            .combine_first(pd.to_datetime(df[date_column], format='%m/%d/%Y', errors='coerce')) \
                            .combine_first(pd.to_datetime(df[date_column], format='%Y/%m/%d', errors='coerce')) \
                            .combine_first(pd.to_datetime(df[date_column], format='%d-%m-%Y', errors='coerce')) \
                            .combine_first(pd.to_datetime(df[date_column], format='%Y-%m-%d %H:%M', errors='coerce')) \
                            .combine_first(pd.to_datetime(df[date_column], format='%d.%m.%Y', errors='coerce')) \
                            .combine_first(pd.to_datetime(df[date_column], format='%Y.%m.%d', errors='coerce')) \
                            .combine_first(pd.to_datetime(df[date_column], format='%b %d %Y', errors='coerce')) \
                            .combine_first(pd.to_datetime(df[date_column], format='%d %b %Y', errors='coerce')) \
                            .combine_first(pd.to_datetime(df[date_column], format='%Y-%m-%dT%H:%M:%S', errors='coerce')) \
                            .combine_first(pd.to_datetime(df[date_column], format='%Y-%m-%dT%H:%M:%S%z', errors='coerce'))
        inconsistent_rows = df[df[date_column].isna()]
        if not inconsistent_rows.empty:
            print(f'Inconsistencies found: \n{inconsistent_rows}')
          
        return df