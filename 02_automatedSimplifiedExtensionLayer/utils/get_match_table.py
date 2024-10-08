import pandas as pd
import os

def get_match_table(path_Eco_list, matchingTableName):
    # Read the Excel file using pandas
    file_path = os.path.join(path_Eco_list, matchingTableName)
    match = pd.read_excel(file_path, header=None).values  # Read Excel into NumPy array

    # Replace NaN values with an empty list, similar to the MATLAB behavior
    match = [[[] if pd.isna(cell) else cell for cell in row] for row in match]
    
    return match
