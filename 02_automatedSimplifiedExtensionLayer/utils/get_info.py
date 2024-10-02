#from utils.excel_interaction import read_from_excel_cell
from dataclasses import dataclass, field
import pandas as pd
import re

@dataclass
class Info:
    name: str = 'unknown'
    abbreviation: str = 'unknown'
    process_description: str = 'unknown'
    mainflow: str = 'unknown'
    location: str = 'unknown'
    exact_location: str = field(default='unknown')
    capacity: float = 0.0
    unit_per_year: str = field(default='t/a')

    def find_main_product(self, string: str) -> None:
        product_search = re.search(r"Product: (.*?),", string) # Look for words after "Product:" until ","
        if product_search:
            self.mainflow = product_search.group(1) # Extract main product
        else:
            raise RuntimeError(f"Could not find Product in: {string}")

    def find_location(self, string: str) -> None:
        """Find and set the location (geography) from the input string."""
        # Find location (Geography)
        geography_search = re.search(r"Geography: (.*?),", string)
        if geography_search:
            self.location = geography_search.group(1)  # Extract the location
        else:
            raise RuntimeError(f"Could not find Geography in: {string}")

    def find_process_description(self, file: pd.DataFrame) -> None:
        """Find and set the process description, name, abbreviation, and capacity."""
        # Find the process description in the first column
        first_column_s = file.iloc[:, 0]  # First column in the Excel file
        for index, value in first_column_s.items():
            if value == 'PROCESS DESCRIPTION':
                self.process_description = first_column_s[index + 1]  # Set the row after the description header
                break
        # Extract name and abbreviation (assuming abbreviation is the same as the name)
        self.name = file.iloc[0, 0]  # Cell (1, 1) -> row 0, column 0 in pandas
        self.abbreviation = self.name
        # Set capacity from cell (7, 10), which is (6, 9) in pandas
        self.capacity = file.iloc[6, 9] * 1000  # Multiplying capacity as per the original code


def get_info(file: pd.DataFrame) -> Info:
    """Main function to populate and return an Info object."""
    # Initialize Info object
    info = Info()

    # Read the string from cell (2, 1) which is row 1, column 0 in pandas
    string = file.iloc[1, 0]

    # Call class methods to populate the Info object
    info.find_main_product(string)
    info.find_location(string)
    info.find_process_description(file)

    # Return the populated Info object
    return info

