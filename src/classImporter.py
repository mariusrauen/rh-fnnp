from dataclasses import dataclass
from pathlib import Path
import logging
import pandas as pd


@dataclass
class Importer:
    base_dir: Path
    def combine_eso(self, sub_dir, add_source_file=False, delimiter=','):
        '''
        Combines all CSV files in the specified directory into one DataFrame.
        If a file has extra columns, they are added to the DataFrame with missing values filled with zeros.
        Parameters:
        dir_path (Path): Path to the directory containing the CSV files.
        add_source_file (bool): If True, adds a column with the source file name to each row. Default is False.
        Returns:
        pd.DataFrame: A combined DataFrame of all the CSV files in the directory with aligned columns.
        '''
        # define directory path
        dir_path = self.base_dir / sub_dir
        # initalize empty lists to store DataFrames
        df_list = []
        # initialize empty list to keep track of all unique columns in the order they appear
        all_columns = []
        # loop through each csv file in the directroy
        for file in dir_path.glob('*.csv'):
            try:
                # read csv file in DataFrame with specific delimiter
                df = pd.read_csv(file, delimiter=delimiter, low_memory=False)
                # add the source file column 
                if add_source_file:
                    df['source_file'] = file.name
                # check for new columns and add them to the end of the all_columns list
                for column in df.columns:
                    if column not in all_columns:
                        all_columns.append(column)
                # append the DataFrame to the list
                df_list.append(df)
            except Exception as e:
                logging.info(f'Error reading {file.name}: {e}')

        # if there are no DataFrames to combine, return an empty DataFrame
        if not df_list:
            print("No files were concatenated due to errors.")
            return pd.DataFrame()
        # align all DataFrames to have the same columns and fill missing columns with zeros
        aligned_dfs = []
        for df in df_list:
            # add missing columns with zero values
            for column in all_columns:
                if column not in df.columns:
                    df[column] = 0
            # reorder columns to match all_columns
            df = df[all_columns]
            aligned_dfs.append(df)
        # concatenate all DataFrames in the list into one DataFrame
        combined_df = pd.concat(aligned_dfs, ignore_index=True)
        logging.info(f'Successfully combined {len(df_list)} DataFrames')
        return combined_df
    
    base_dir: Path

    
    def combine_smard(self, sub_dir, pos=0, delimiter=';'):
        '''
        Combines all CSV files in the specified directory into a single DataFrame.
        Merges horizontally and removes duplicate columns.
        Parameters:
        dir_path (Path): Path to the directory containing the CSV files.
        Returns:
        pd.DataFrame: A combined DataFrame of all the CSV files in the directory with aligned columns.
        '''
        # define directory path
        dir_path = self.base_dir / sub_dir
        # list to store individual DataFrames
        df_list = []
        # Iterate through all CSV files in the directory
        for file in dir_path.glob('*.csv'):
            try:
                # read CSV file into a DataFrame
                df = pd.read_csv(file, sep=delimiter, parse_dates=['Datum von', 'Datum bis'], decimal=',', low_memory=False)
                # extract the prefix from the filename
                prefix = file.stem.split('_')[pos]
                # rename columns by adding the prefix extracted from filename (except for 'Datum von' and 'Datum bis')
                df = df.rename(columns={col: f"{prefix}_{col}" if col not in ['Datum von', 'Datum bis'] else col for col in df.columns})
                df = df.rename(columns={col: col.replace('Originalauflösungen', '').strip() for col in df.columns})
                # append the DataFrame to the list
                df_list.append(df)
            except Exception as e:
                # print an error message if the file could not be read
                print(f"Error reading {file}: {e}") 

        if not df_list:
            logging.info('No CSV files could be read or the directory is empty.')
            return pd.DataFrame()
        
        # Combine all DataFrames and remove duplicates
        combined_df = pd.concat(df_list, axis=1)
        columns_to_remove = combined_df.columns.duplicated(keep='first')
        combined_df = combined_df.loc[:, ~columns_to_remove]
        
        logging.info(f'Successfully combined {len(df_list)} DataFrames')
        return combined_df
