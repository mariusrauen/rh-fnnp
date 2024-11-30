## START HEADER
# module: fuzzy systems and neural networks  
# name: rauen, marius
# matricle number: 131242002
# contact: marius.rauen@rfh-campus.de
# --------------------------------
# strucutre information:
# ! Familiarize yourself with the README file
# ! chapters are introduced with two hashtags and space and are written in capitol letters. e.g. ## HEADER
# ! sections are introduced with one hashtag and space and are written in capitol letters. e.g. # IMPORT OFFICIAL LIBRARIES 
# ! 'regular' comments are introduced with one hashtag and space and are written in small letters. e.g. # from src directory, point to the data directory
# ! inactive code is marked with hastag and no space e.g. #print()
## END HEADER



from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional
import logging
import pandas as pd


from classMetaData import DatasetRegistry
from classImporter import Importer
from classTidy import Tidy
from classTransfomer import Transformer


@dataclass
class DataProcessor:
    base_dir: Path
    logger: logging.Logger = field(init=False)
    df_dict: Dict[str, pd.DataFrame] = field(default_factory=dict)
    #print(df_dict)

    def __post_init__(self):
        '''This function is called immediately after __init__ to implement a logger'''
        log_dir = Path(__file__).parent.parent / 'data' / 'processed' / 'DataPrep'  # 'logs' will be a subdirectory of the current script's directory
        log_dir.mkdir(parents=True, exist_ok=True)  # Create the 'logs' directory if it doesn't exist

        # Log file path inside the 'logs' directory
        log_file = log_dir / 'log_DataPrep.log'
        
        logging.basicConfig(
        level=logging.DEBUG,
        filename=log_file,
        filemode='w',
        format='%(asctime)s - %(levelno)s - %(lineno)d - %(module)s - %(message)s',
        style='%'
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize classes
        #self.esoimport = ESOImport(base_dir=self.base_dir / 'data')
        #self.smardimport = SMARDImport(base_dir=self.base_dir / 'data')
        self.importer = Importer(base_dir=self.base_dir / 'data' / 'raw')

        self.tidy = Tidy()
        self.trans = Transformer()
        self.logger.info('Initialize classes successful')

    def process_eso_data(self, filename: str, date_col: str, period_col: Optional[str] = None) -> pd.DataFrame:
        '''Process ESO data'''
        #df = self.esoimport.combine_eso(filename, add_source_file=True)
        df = self.importer.combine_eso(filename, add_source_file=True)
        df = self.tidy.standardize_time(df, date_col)
        
        if period_col:
            df = self.trans.manipulate_esolog(df, date_col, period_col)
        else:
            df = self.trans.manipulate_esolog(df, date_col)
            
        df = self.trans.set_time_span(df)
        self.tidy.df_info(df)

        self.logger.info(f'Return df {df.shape} {filename}')
        return df

    def process_smard_data(self, filename: str, date_col: str, pos: Optional[int] = None) -> pd.DataFrame:
        '''Process SMARD data'''
        if pos is not None:
            df = self.importer.combine_smard(filename, pos=pos)
        else:
            df = self.importer.combine_smard(filename)
            
        df = self.tidy.standardize_time(df, date_col)
        df = self.trans.manipulate_smardlog(df, date_col)
        df = self.trans.set_time_span(df)
        self.tidy.df_info(df)

        self.logger.info(f'Return df {df.shape} {filename}')
        return df

    def process_all_data(self):
        '''Process all datasets based on configuration'''
        configs = DatasetRegistry.get_configs()
        
        for key, config in configs.items():
            if config.source == 'eso':
                self.df_dict[key] = self.process_eso_data(
                    config.filename,
                    config.date_column,
                    config.period_column
                )
            else:  # smard
                self.df_dict[key] = self.process_smard_data(
                    config.filename,
                    config.date_column,
                    config.position
                )

        # Analyze mismatches
        #self.trans.analyze_df_mismatches(self.df_dict)
        self.logger.info('Mismatches analyzed')

        self.df_dict = self.trans.align_dataframes(self.df_dict)

        self.df_dict = self.trans.prepare_for_regression(self.df_dict)

        #self.df_dict = self.trans.normalize(self.df_dict)

        df_eso = self.trans.merge_dataframes(self.df_dict, ['df_bc', 'df_bv', 'df_dd', 'df_dd', 'df_ci', 'df_si'])
        df_smard = self.trans.merge_dataframes(self.df_dict, ['df_gcf', 'df_ss'])
                
        logging.info(f'Shape of final ESO {df_eso.shape}')
        logging.info(f'Shape of final SMARD {df_smard.shape}')

        self.save_dataframes_to_csv(df_eso, df_smard)
    
        

    def save_dataframes_to_csv(self, df_eso, df_smard):
        """
        Saves the merged DataFrames to CSV files in the data directory.
        
        Parameters:
        df_eso (pd.DataFrame): Merged ESO DataFrame
        df_smard (pd.DataFrame): Merged SMARD DataFrame
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
        
        self.logger.info(f'Saved ESO data to {processed_dir}/df_eso.csv')
        self.logger.info(f'Saved SMARD data to {processed_dir}/df_ger.csv')



def main():
    '''Initialize and run the data processor'''
    processor = DataProcessor(base_dir=Path(__file__).parent.parent)
    processor.process_all_data()
if __name__ == "__main__":
    main()


