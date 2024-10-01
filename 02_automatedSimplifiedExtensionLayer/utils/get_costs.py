from dataclasses import dataclass
import pandas as pd
import logging

@dataclass
class Costs:
    name: str
    value: float
    unit: str

def get_costs(file: pd.DataFrame) -> list[Costs]:
    """
    Function to extract cost information from the IHS Excel file and return a list of Costs objects.
    Parameters:
    (file: pd.DataFrame): This is the function's parameter list. It specifies that the function takes one argument named file, which is expected to be a pd.DataFrame
    Return:
    -> list[Costs]:: This is the function's return type annotation. It specifies that the function is expected to return a list of objects of the Costs class.
    """
    
    costs = []
    
    # Extract capacity and base unit from the DataFrame
    capacity = file.iloc[6, 9]  # MATLAB 7th row, 10th column -> Pandas 6, 9
    base_unit = file.iloc[5, 3][4:]  # MATLAB 6th row, 4th column -> Pandas 5, 3 and removing first 4 characters
    
    # Determine the conversion factor based on the base unit
    if base_unit == 'TONNE':
        factor = 1e-3
    elif base_unit == 'MNM3':
        factor = 1e-6
    else:
        logging.warning('Unit unknown in Cost')
        factor = 1

    unit = '$'  # Default unit

    # Investment costs
    # Inside battery
    costs.append(Costs(
        name=file.iloc[8, 7],  # MATLAB 9th row, 8th column -> Pandas 8, 7
        value=file.iloc[8, 9] * 1e8 / 1e9 / capacity,  # MATLAB 9th row, 10th column -> Pandas 8, 9
        unit=unit
    ))

    # Off sites
    costs.append(Costs(
        name=file.iloc[9, 7],  # MATLAB 10th row, 8th column -> Pandas 9, 7
        value=file.iloc[9, 9] * 1e8 / 1e9 / capacity,  # MATLAB 10th row, 10th column -> Pandas 9, 9
        unit=unit
    ))

    # Operating costs
    indices = [16, 17, 18, 19, 20, 22, 23, 25, 27]  # MATLAB indexing [17,18,19,20,21,23,24,26,28] -> Python zero-indexed
    
    for i in indices:
        costs.append(Costs(
            name=file.iloc[i, 7],  # MATLAB column 8 -> Pandas column 7
            value=file.iloc[i, 9] / 100,  # MATLAB column 10 -> Pandas column 9
            unit=unit
        ))

    return costs

