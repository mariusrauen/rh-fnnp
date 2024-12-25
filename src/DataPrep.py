# ============================================================================================================================
## HEADER
# Module: Fuzzy Systems and Neural Networks  
# Name: Raun, Marius
# Matricle number: 131242002
# Contact: marius.rauen@rfh-campus.de

# Strucutre information:
# ! Familiarize yourself with the README file
# ! New sections are introduced with two hashtags, space and in capitol letters, e.g. ## HEADER
# ! Comments are introduced with one hashtag and space, e.g. # Base directory path for all data operations
# ! Inactive code is marked with one hastag and no space, e.g. #print()
# ============================================================================================================================



# ============================================================================================================================
## IMPORT LIBRARIES
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional
import logging
import pandas as pd

from modules.classMetaData import DatasetRegistry
from modules.classImporter import Importer
from modules.classTidy import Tidy
from modules.classTransfomer import Transformer
# ============================================================================================================================



# ============================================================================================================================
@dataclass
class DataProcessor:
    """
    A class for processing and managing data operations.
    Handles data import, tidy, transformation, and logging functionality.
    """
    base_dir: Path # Base directory path for all data operations  
    logger: logging.Logger = field(init=False) # Logger for logging messages
    df_dict: Dict[str, pd.DataFrame] = field(default_factory=dict) # Dictionary to store DataFrames

    def __post_init__(self):
        """
        Post-initialization setup:
        1. Configures logging system
        2. Initializes component classes for data processing
        """
        log_dir = Path(__file__).parent.parent / 'data' / 'processed' / 'DataPrep'  # Set the log directory path
        log_dir.mkdir(parents=True, exist_ok=True)  # Create the log directory if it doesn't exist

        # Set log file path and name
        log_file = log_dir / 'log_DataPrep.log'
        
        # Configure logging system
        logging.basicConfig(
        level=logging.DEBUG,    # Set the logging level
        filename=log_file,      # Set the log file path
        filemode='w',           # Set the file mode
        format='%(asctime)s - %(levelno)s - %(lineno)d - %(module)s - %(message)s',
        style='%'
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize processing classes
        self.importer = Importer(base_dir=self.base_dir / 'data' / 'raw') # Initialize Importer class
        self.tidy = Tidy() # Initialize Tidy class
        self.trans = Transformer() # Initialize Transformer class
        self.logger.info('Initialize classes successful') 

    # ------------------------------------------------------------------------------------------------------------------------

    def process_eso_data(self, filename: str, date_col: str, period_col: Optional[str] = None) -> pd.DataFrame:
        """Process ESO (Energy System Operator) data from a file and return standardized DataFrame.
        Args:
            filename (str): Path to the ESO data file to be processed
            date_col (str): Name of the column containing date information
            period_col (Optional[str], optional): Name of the column containing period information. Defaults to None.
        Returns:
            pd.DataFrame: Processed DataFrame with standardized time format and manipulated ESO log data
        """

        df = self.importer.combine_eso(filename, add_source_file=True)  # Combine ESO data from the input file
        df = self.tidy.standardize_time(df, date_col) # Standardize time format
        
        if period_col: # Check if period column is provided
            df = self.trans.manipulate_esolog(df, date_col, period_col) # Manipulate ESO log data with period column
        else: 
            df = self.trans.manipulate_esolog(df, date_col) # Manipulate ESO log data without period column
            
        df = self.trans.set_time_span(df) # Set time span for the dataset
        self.tidy.df_info(df) # Display DataFrame information

        self.logger.info(f'Return df {df.shape} {filename}') 
        return df
    
    # ------------------------------------------------------------------------------------------------------------------------

    def process_smard_data(self, filename: str, date_col: str, pos: Optional[int] = None) -> pd.DataFrame:
        """Process SMARD (German energy market) data from a file and return standardized DataFrame.
        Args:
            filename (str): Path to the SMARD data file to be processed
            date_col (str): Name of the column containing date information
            pos (Optional[int], optional): Position index for data selection. Defaults to None.
        Returns:
            pd.DataFrame: Processed DataFrame with standardized time format and manipulated SMARD log data
        """

        if pos is not None: # Check if position index is provided
            df = self.importer.combine_smard(filename, pos=pos) # Combine SMARD data from the input file with position index
        else:
            df = self.importer.combine_smard(filename) # Combine SMARD data from the input file without position index
            
        df = self.tidy.standardize_time(df, date_col)  # Standardize time format
        df = self.trans.manipulate_smardlog(df, date_col) # Manipulate SMARD log data
        df = self.trans.set_time_span(df) # Set time span for the dataset
        self.tidy.df_info(df) # Display DataFrame information

        self.logger.info(f'Return df {df.shape} {filename}')
        return df

    # ------------------------------------------------------------------------------------------------------------------------

    def process_all_data(self):
        """Process all configured datasets and store them in the instance dictionary.
        Iterates through dataset configurations from DatasetRegistry and processes each
        dataset according to its source type (ESO, SMARD, etc.). Processed DataFrames
        are stored in self.df_dict using the configuration key as dictionary key.
        Returns:
            None: Updates self.df_dict with processed DataFrames
        """
        
        configs = DatasetRegistry.get_configs() # Get dataset configurations from DatasetRegistry class instance of classMetaData.py
        
        for key, config in configs.items(): # Iterate through dataset configurations
            if config.source == 'eso': # Check if source is ESO get data from ESO
                self.df_dict[key] = self.process_eso_data( 
                    config.filename, 
                    config.date_column,
                    config.period_column
                )
            else:  # Otherwise get data from SMARD
                self.df_dict[key] = self.process_smard_data(
                    config.filename,
                    config.date_column,
                    config.position
                )

        self.logger.info('Mismatches analyzed') # Log message for mismatches analyzed

        self.df_dict = self.trans.align_dataframes(self.df_dict) # Align DataFrames

        self.df_dict = self.trans.prepare_for_regression(self.df_dict) # Prepare DataFrames for regression

        # Merge DataFrames and save to CSV
        df_eso = self.trans.merge_dataframes(self.df_dict, ['df_bc', 'df_bv', 'df_dd', 'df_dd', 'df_ci', 'df_si'])
        df_smard = self.trans.merge_dataframes(self.df_dict, ['df_gcf', 'df_ss'])
                
        logging.info(f'Shape of final ESO {df_eso.shape}')
        logging.info(f'Shape of final SMARD {df_smard.shape}')

        self.save_dataframes_to_csv(df_eso, df_smard)
    
    # ------------------------------------------------------------------------------------------------------------------------

    def save_dataframes_to_csv(self, df_eso, df_smard):
        """
        Saves the merged DataFrames to CSV files in the data directory.
        Args:
        save_path (str): Directory path where CSV files will be saved
        Side Effects:
            - Creates CSV files in the specified directory
            - Each DataFrame is saved as '{key}.csv'
        """

        # Create 'processed' directory if it doesn't exist
        processed_dir = self.base_dir / 'data' / 'processed' / 'DataPrep'
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Define the save directory paths
        eso_dir = processed_dir / 'eso'; eso_dir.mkdir(parents=True, exist_ok=True)
        ger_dir = processed_dir / 'ger'; ger_dir.mkdir(parents=True, exist_ok=True)

        # Save DataFrames to CSV
        df_eso.to_csv(processed_dir / 'eso' / 'df_eso.csv', index=False)
        df_smard.to_csv(processed_dir / 'ger'/ 'df_ger.csv', index=False)
        
        # Log messages
        self.logger.info(f'Saved ESO data to {processed_dir}/df_eso.csv')
        self.logger.info(f'Saved SMARD data to {processed_dir}/df_ger.csv')
# ============================================================================================================================



# ============================================================================================================================
def main():
    '''Initialize and run the data processor'''
    processor = DataProcessor(base_dir=Path(__file__).parent.parent)
    processor.process_all_data()
if __name__ == "__main__":
    main()
# ============================================================================================================================