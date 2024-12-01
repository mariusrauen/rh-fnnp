from dataclasses import dataclass
import logging
import pandas as pd
import numpy as np
import warnings
from typing import Dict
from sklearn. preprocessing import MinMaxScaler

from .classTidy import Tidy
timeformat = '%Y-%m-%d %H:%M'
ID = 'ID'
tidy = Tidy()

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning, message="The behavior of array concatenation with empty entries is deprecated.*")

@dataclass
class Transformer:
    #def __init__(self):
    #    pass


    @staticmethod
    def manipulate_esolog(df, date_col, period_col=None):
        '''
        Function to create one identifies column called LOG
        Parameters:
        df: DataFrame
        date_col: contains the date
        Period_col: contains the period form 1...48 to be combined with the date
        Returns:
        Column with combined datetime
        '''
        if period_col is not None and period_col in df.columns:
            base_date = pd.to_datetime(df[date_col])
            
            # Calculate the time offset from SETT_PERIOD
            hours = (df[period_col] - 1) // 2
            minutes = 30 * ((df[period_col] - 1) % 2)

            # Create a time offset column using pd.Timedelta
            time_offset = pd.to_timedelta(hours, unit='h') + pd.to_timedelta(minutes, unit='m')

            # Combine base_date with time_offset to create the LOG datetime column
            df[ID] = (base_date + time_offset).dt.strftime(timeformat)
            
            # Reorder columns to make 'LOG' the first column
            columns = [ID] + [col for col in df.columns if col != ID]
            df = df[columns]

            # Remove some columns
            df = df.drop([date_col, period_col, 'source_file'], axis=1)

            logging.info(f'Creation of ID column for ESO df with 30min LOGs successful')            
            return df
                
        else:
            base_date = pd.to_datetime(df[date_col])

            # Create LOG datetime column
            df[ID] = (base_date).dt.strftime(timeformat)
            
            # Reorder columns to make 'LOG' the first column
            columns = [ID] + [col for col in df.columns if col != ID]
            df = df[columns]

            # Remove some columns
            df = df.drop([date_col, 'source_file'], axis=1)
        
            logging.info(f'Creation of ID column for ESO df with 30min LOGs successful')
            return df


    @staticmethod
    def manipulate_smardlog(df, date_col):
        '''
        Function for data set from SMARD to remove rows with times in the LOG column that fall on quarter hours (XX:15:00 and XX:45:00).
        Parameters:
        df: DataFrame
        log_col: Name of the LOG column
        Returns:
        DataFrame with quarter-hour rows removed
        '''
        df = df[df[date_col].dt.minute.isin([15, 45]) == False]
        df = df.reset_index(drop=True)
        df = df.drop(['Datum bis'], axis=1)
        df[date_col] = df[date_col].dt.strftime(timeformat)
        df.rename(columns={date_col: ID}, inplace=True)

        logging.info(f'Creation of ID column for SMARD df with 30min LOGs successful')  
        return df


    @staticmethod
    def set_time_span(df, start_date='2017-04-01 00:00', end_date= '2024-08-31 23:30'):
        '''
        Function to assimilate and specifiy the investigated period of time of the data recorded.
        Parameters:
        df: DataFrame
        start_date: Beginnging of the time span
        end_date: End of the time span
        Returns:
        DataFrame only containing the time span of user defined interest
        '''
        df[ID] = pd.to_datetime(df[ID])

        # transform the start_date and end_date in pandas datimetime to furhter process it
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)

        df = df[(df[ID] >= start) & (df[ID] <= end)]
        
        logging.info(f'Setting time span from 2017-04-01 00:00 to 2024-08-31 23:30')
        return df


    @staticmethod
    def analyze_df_mismatches(df_dict):
        '''
        Function to identify missmatches of the shape from the dataframes.
        Parameters:
        df_dict: Dictionary with names and Dataframes
        Returns:
        '''  
        for df in df_dict.values():
            if df[ID].dtype != 'datetime64[ns]':
                df[ID] = pd.to_datetime(df[ID])
    
        # Find the common date range
        start_date = max(df[ID].min() for df in df_dict.values())
        end_date = min(df[ID].max() for df in df_dict.values())
        
        print(f'\nCommon date range: {start_date} to {end_date}')
        
        # Analyze each dataframe
        for name, df in df_dict.items():
            df_dates = set(df[ID])
            
            # Calculate the most common time difference
            time_diffs = df[ID].diff().value_counts()
            most_common_diff = time_diffs.index[0]
            
            
            print('\n================================')
            print(f'DataFrame: {name}')
            print(f'Total rows: {len(df)}')
            
            # Generate expected dates based on the most common difference
            expected_dates = pd.date_range(start=start_date, end=end_date, freq=most_common_diff)
            
            missing_dates = set(expected_dates) - df_dates
            #extra_dates = df_dates - set(expected_dates)
            
            print(f'Missing dates: {len(missing_dates)}')
            #print(f'Extra dates: {len(extra_dates)}')
                    
            if missing_dates:
                #print('Sample of missing dates:')
                #print(sorted(list(missing_dates))[:])
                pass

            # Check for any irregular time differences
            irregular_diffs = time_diffs[time_diffs.index != most_common_diff]
            if not irregular_diffs.empty:
                print('\nIrregular time differences detected:')
                print(irregular_diffs)
                print('================================')


    @staticmethod
    def align_dataframes(df_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        ''''
        Aligns multiple dataframes by their ID column, keeping only rows where IDs exist in all dataframes.
        Each DataFrame must have an 'ID' column with datetime values in the format 'YYYY-MM-DD HH:MM:SS'.
        
        Parameters:
        df_dict (Dict[str, pd.DataFrame]): Dictionary of DataFrames with 'ID' column
        
        Returns:
        Dict[str, pd.DataFrame]: Dictionary of aligned DataFrames with identical IDs
        '''
        # Convert all ID columns to datetime if they aren't already
        for name, df in df_dict.items():
            if df[ID].dtype != 'datetime64[ns]':
                df[ID] = pd.to_datetime(df[ID])
            # Remove any duplicates in ID column
            if df['ID'].duplicated().any():
                logging.warning(f'Found {df[ID].duplicated().sum()} duplicate IDs in {name}')
                df = df.drop_duplicates(subset=[ID], keep='first')
                df_dict[name] = df
        
        # Get set of IDs from each DataFrame and log the sizes
        id_sets = {}
        for name, df in df_dict.items():
            id_sets[name] = set(df[ID])
            logging.info(f'{name} has {len(id_sets[name])} unique IDs')
        
        # Find common IDs across all DataFrames using set intersection
        common_ids = set.intersection(*id_sets.values())
        logging.info(f'Found {len(common_ids)} common IDs across all DataFrames')
        
        # Debug: Check which IDs are missing from each DataFrame
        for name, id_set in id_sets.items():
            missing_ids = common_ids - id_set
            extra_ids = id_set - common_ids
            if missing_ids:
                logging.warning(f'{name} is missing {len(missing_ids)} IDs from common set')
                logging.debug(f'Sample of missing IDs in {name}: {sorted(list(missing_ids))[:5]}')
            if extra_ids:
                logging.warning(f'{name} has {len(extra_ids)} extra IDs not in common set')
                logging.debug(f'Sample of extra IDs in {name}: {sorted(list(extra_ids))[:5]}')
        
        # Create a sorted list of common IDs for consistent ordering
        common_ids_sorted = sorted(list(common_ids))
        
        # Filter each DataFrame to keep only common IDs
        aligned_dict = {}
        for name, df in df_dict.items():
            # Filter DataFrame to keep only common IDs
            aligned_df = df[df[ID].isin(common_ids_sorted)].copy()
            
            # Sort by ID to ensure consistent order
            aligned_df = aligned_df.sort_values('ID').reset_index(drop=True)
            
            # Convert ID back to string format for consistency
            aligned_df[ID] = aligned_df[ID].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Store in new dictionary
            aligned_dict[name] = aligned_df
            
            # Log the changes
            logging.info(f'Aligned DataFrame {name}: Original shape {df.shape} -> New shape {aligned_df.shape}')
        
        # Verify all DataFrames have the same size and IDs
        sizes = {name: len(df) for name, df in aligned_dict.items()}
        if len(set(sizes.values())) > 1:
            # If sizes don't match, do detailed comparison
            for name1, df1 in aligned_dict.items():
                for name2, df2 in aligned_dict.items():
                    if name1 < name2:  # Compare each pair only once
                        diff_ids = set(df1['ID']) ^ set(df2['ID'])  # Symmetric difference
                        if diff_ids:
                            logging.error(f'ID mismatch between {name1} and {name2}: {len(diff_ids)} different IDs')
                            logging.debug(f'Sample of different IDs: {sorted(list(diff_ids))[:5]}')
            raise ValueError(f'Error: DataFrames have different sizes after alignment: {sizes}')
        else:
            logging.info(f'All DataFrames successfully aligned to {next(iter(sizes.values()))} rows')
            
            # Final verification of ID consistency
            first_df_ids = set(aligned_dict[next(iter(aligned_dict))][ID])
            for name, df in aligned_dict.items():
                if set(df[ID]) != first_df_ids:
                    raise ValueError(f'ID mismatch found in {name} after alignment')
        
        return aligned_dict
    
    """
    @staticmethod
    def align_dataframes(df_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        ''''
        Aligns multiple dataframes by their ID column, filling missing rows with median values.
        Each DataFrame must have an 'ID' column with datetime values in the format 'YYYY-MM-DD HH:MM:SS'.
        
        Parameters:
        df_dict (Dict[str, pd.DataFrame]): Dictionary of DataFrames with 'ID' column
        
        Returns:
        Dict[str, pd.DataFrame]: Dictionary of aligned DataFrames with identical IDs
        '''
        # Convert all ID columns to datetime if they aren't already
        for name, df in df_dict.items():
            if df[ID].dtype != 'datetime64[ns]':
                df[ID] = pd.to_datetime(df[ID])
            # Remove any duplicates in ID column
            if df['ID'].duplicated().any():
                logging.warning(f'Found {df[ID].duplicated().sum()} duplicate IDs in {name}')
                df = df.drop_duplicates(subset=[ID], keep='first')
                df_dict[name] = df
        
        # Get set of IDs from each DataFrame and log the sizes
        id_sets = {}
        for name, df in df_dict.items():
            id_sets[name] = set(df[ID])
            logging.info(f'{name} has {len(id_sets[name])} unique IDs')
        
        # Get the union of all IDs from each DataFrame
        all_ids = set().union(*id_sets.values())
        logging.info(f'Found {len(all_ids)} unique IDs across all DataFrames')
        
        # Create a sorted list of all IDs for consistent ordering
        all_ids_sorted = sorted(list(all_ids))
        
        # Align each DataFrame to include all IDs, filling missing rows with median values
        aligned_dict = {}
        for name, df in df_dict.items():
            # Create a new DataFrame with all IDs and fill missing rows with NaN
            aligned_df = pd.DataFrame({ID: all_ids_sorted})
            aligned_df = aligned_df.merge(df, on=ID, how='left')
            
            # Fill missing values in each column with the column's median (excluding ID column)
            for col in aligned_df.columns:
                if col != ID:
                    aligned_df[col] = aligned_df[col].fillna(aligned_df[col].median())
            
            # Sort by ID to ensure consistent order
            aligned_df = aligned_df.sort_values('ID').reset_index(drop=True)
            
            # Convert ID back to string format for consistency
            aligned_df[ID] = aligned_df[ID].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Store in new dictionary
            aligned_dict[name] = aligned_df
            
            # Log the changes
            logging.info(f'Aligned DataFrame {name}: Original shape {df.shape} -> New shape {aligned_df.shape}')
        
        # Verify all DataFrames have the same size and IDs
        sizes = {name: len(df) for name, df in aligned_dict.items()}
        if len(set(sizes.values())) > 1:
            raise ValueError(f'Error: DataFrames have different sizes after alignment: {sizes}')
        else:
            logging.info(f'All DataFrames successfully aligned to {next(iter(sizes.values()))} rows')
            
            # Final verification of ID consistency
            first_df_ids = set(aligned_dict[next(iter(aligned_dict))][ID])
            for name, df in aligned_dict.items():
                if set(df[ID]) != first_df_ids:
                    raise ValueError(f'ID mismatch found in {name} after alignment')
        
        return aligned_dict
    """
        
    def prepare_for_regression(self, df_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        '''
        Convert all data to consistent numeric format with standard decimal points.
        All values will be converted to float format, suitable for further processing.
        
        Parameters:
        df_dict (Dict[str, pd.DataFrame]): Dictionary containing dataframes to process
        
        Returns:
        Dict[str, pd.DataFrame]: Dictionary containing dataframes with standardized numeric values
        '''
        logging.info('Starting data preparation for regression')
        
        processed_dict = {}
        
        for key, df in df_dict.items():
            logging.info(f'Processing dataframe: {key}')
            
            # Create a copy to avoid modifying the original
            df_cleaned = df.copy()
            
            # Process each column
            for column in df_cleaned.columns:
                if column != ID:  # Skip identification column
                    try:
                        # Handle missing values and dashes
                        mask_missing = (df_cleaned[column] == '-') | pd.isna(df_cleaned[column])
                        
                        # Convert to string for consistent processing
                        series = df_cleaned[column].astype(str)
                        
                        # Clean and standardize the numeric strings
                        # Remove spaces and handle European number format
                        numeric_series = (series
                            .str.replace(' ', '')  # Remove spaces
                            .str.replace(r'\.(?=.*,)', '', regex=True)  # Remove thousand separators (dots before commas)
                            .str.replace(',', '.')  # Convert decimal commas to points
                        )
                        
                        # Convert to float
                        df_cleaned[column] = pd.to_numeric(numeric_series, errors='coerce')
                        
                        # Set missing values to 0.0
                        df_cleaned.loc[mask_missing, column] = 0.0
                        
                        # Log the number of NaN values in the column
                        nan_count = df_cleaned[column].isna().sum()
                        if nan_count > 0:
                            logging.warning(f'Column {column} in {key} contains {nan_count} NaN values')
                            
                    except Exception as e:
                        logging.error(f'Error processing column {column} in {key}: {str(e)}')
                        continue
            
            # Remove any rows where all numeric columns are NaN
            numeric_columns = df_cleaned.select_dtypes(include=[np.number]).columns
            df_cleaned = df_cleaned.dropna(subset=numeric_columns, how='all')
            
            # Log the shape of the cleaned dataframe
            logging.info(f'Cleaned {key} shape: {df_cleaned.shape}')
            
            processed_dict[key] = df_cleaned
        
        logging.info('Data preparation completed')
        return processed_dict

    @staticmethod
    def normalize(df_dict: Dict[str, pd.DataFrame], feature_range=(0, 1)) -> Dict[str, pd.DataFrame]:
        '''
        Normalizes columns within each dataframe independently using Min-Max scaling.
        
        Parameters:
        df_dict: Dictionary of DataFrames to normalize
        feature_range: Tuple of (min, max) for normalization range
        
        Returns:
        Dict[str, pd.DataFrame]: Dictionary of normalized DataFrames
        '''
        normalized_dict = {}
        
        # Store scalers for each dataframe and column
        scalers = {}
        
        # First pass: Initialize scalers and fit them
        for key, df in df_dict.items():
            scalers[key] = {}
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            
            for col in numeric_cols:
                scaler = MinMaxScaler(feature_range=feature_range)
                scaler.fit(df[[col]])
                scalers[key][col] = scaler
        
        # Second pass: Transform the data
        for key, df in df_dict.items():
            df_normalized = df.copy()
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            
            for col in numeric_cols:
                df_normalized[col] = scalers[key][col].transform(df[[col]])
                logging.info(f'{key} - {col}: min={df_normalized[col].min():.4f}, max={df_normalized[col].max():.4f}')
            
            normalized_dict[key] = df_normalized
        
        return normalized_dict
    

    @staticmethod
    def merge_dataframes(df_dict: Dict[str, pd.DataFrame], keys_to_merge: list) -> pd.DataFrame:
        """
        Merges selected DataFrames horizontally into a single DataFrame.
        The 'ID' column is kept only from the first DataFrame.
        Parameters:
        df_dict (Dict[str, pd.DataFrame]): Dictionary of aligned DataFrames
        keys_to_merge (list): List of keys from df_dict to merge
        Returns:
        pd.DataFrame: A single merged DataFrame
        """
        # Validate keys
        for key in keys_to_merge:
            if key not in df_dict:
                raise KeyError(f"Key '{key}' not found in DataFrame dictionary")
        
        # Start with the first specified DataFrame
        result_df = df_dict[keys_to_merge[0]]
        
        # For each subsequent DataFrame in the selection, add all columns except 'ID'
        for key in keys_to_merge[1:]:
            df = df_dict[key]
            cols_to_add = [col for col in df.columns if col != 'ID']
            result_df = pd.concat([result_df, df[cols_to_add]], axis=1)
                
        return result_df

    