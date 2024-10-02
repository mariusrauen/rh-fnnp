import pandas as pd
import numpy as np
import os

def get_missing_meta_data(path_own_scenario):
    # Construct the file path
    file_name = os.path.join(path_own_scenario, 'meta_data_flows.xlsx')
    
    # Read the Excel file into a pandas DataFrame
    missing_meta_data = pd.read_excel(file_name, header=None).values.tolist()
    
    # Replace NaN values with empty lists (equivalent to MATLAB's [])
    missing_meta_data = [[None if pd.isna(cell) else cell for cell in row] for row in missing_meta_data]
    
    return missing_meta_data
