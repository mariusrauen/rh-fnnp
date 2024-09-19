from utils.excel_interaction import read_from_excel_cell
from dataclasses import dataclass
import pandas as pd

@dataclass
class Info:
    name: str
    abbreviation: str
    process_description: str
    mainflow: str
    location: str
    exact_location: str
    capacity: str
    unit_per_year: str | float

    exact_location = 'unknown'

    def find_main_product():
        pass

    def find_location() :
        pass

    def find_process_description():
        pass

def get_info(path: Path):
    string = read_from_excel_cell(1,2)
    # Get Product 
    product_search = re.search(r"Product: (.*?),", string) # Look for words after "Product:" until ","
    if product_search:
        mainflow = product_search.group(0) # Returns the first match of Regex 
    else:
        raise RuntimeError(f"Could not find Product here: {string}")
    # Get Geography
    geography_search = re.search(r"Geography: (.*?),", string) # Look for words after "Geography:" until ","
    if geography_search:
        mainflow = geography_search.group(0) # Returns the first match of Regex 
    else:
        raise RuntimeError(f"Could not find Geography here: {string}")
   
    # Get Process description
    first_column_s = pd.read_excel(path, sheet_name=0).iloc[:,0]
    for index, value in first_column_s.items():
        idx_process_description_start = 0
        if value == 'PROCESS DESCRIPTION':
            process_description = first_column_s[index+1]
            break
    return Info(\
            name = read_from_excel_cell(1,1),
            abbreviation = name , 
            process_description = process_description, 
            location = location,
            exact_location = 'unknown',
            capacity = read_from_excel_cell(7,10)*1000,
            unit_per_year = 't/a',
    ) 
