# This is a conversion of the matlab functions for database generation
#from pprint import pprint
import numpy as np
import pandas as pd
import re
from pathlib import Path
#import logging

from utils.cm_classes import Stream
#from utils.convert_units import convert_units

def get_streams(file: Path) -> list[Stream]:
    streams: list[Stream] = []
    first_column_s = pd.read_excel(file, sheet_name=0).iloc[:,0]
    string = first_column_s[2]
    # Find the main product name. If it can not be found raise an error.
    main_name_search = re.search(r"Product: (.*?),", string)
    if main_name_search:
        main_name = main_name_search.group(1) # Returns the first match of Regex 
    else:
        raise(RuntimeError('The regex search was not successful.'))
    # Find the main product's price. If it does not exist, define it with nan$/kg
    price_with_unit_search = re.search(r"Price: (.*?),", string) # Look for words after "Price:" until ","
    if price_with_unit_search:
        price_with_unit = price_with_unit_search.group(1) # Returns the first match of Regex 
        main_cost = float(price_with_unit.split(' ')[0])
        main_cost_unit = price_with_unit.split(' ')[1]
        print(main_cost_unit, main_cost)
    else:
        main_cost = np.nan
        main_cost_unit = "$/kg"
    # Define main product amount
    main_amount = 1
    # Define main product unit
    string2 = pd.read_excel(file, sheet_name=0).iloc[6,3]
    main_amount_unit_search = re.search(r"per (.*$)", string2) # Look for everything behind "per"
    if main_amount_unit_search:
        main_amount_unit = main_amount_unit_search.group(0)
    else:
        raise RuntimeError("Something is wrong with the regex for searching the main amount unit!")

    # Add main product to list of streams
    streams.append(Stream(name=main_name, cost=main_cost, cost_unit=main_cost_unit, amount=main_amount, amount_unit=main_amount_unit, cost_per_kg=np.nan, class_=1))


    # Generate Streams for Raw materials, by-products and utilities. In the future this can be encapsulated into a single function.

    # Generate Streams for Raw materials 
    ## Find the start index of the raw materials
    for index, value in first_column_s.items():
        idx_raw_materials_start = 0
        if value == 'RAW MATERIALS':
            idx_raw_materials_start = index+1
            break
    ## Only look for raw materials if they exist and find end index of materials.
    if idx_raw_materials_start>0:
        for index, value in first_column_s[idx_raw_materials_start:].items():
            if pd.isna(value):
                idx_raw_materials_end = index-1
                break 
    # Get Dataframe with all raw materials
    raw_materials_df = pd.read_excel(file, sheet_name=0).iloc[idx_raw_materials_start:idx_raw_materials_end+1]
    for raw_material_idx, values in raw_materials_df.iterrows():
        streams.append(Stream(name=values[0], cost=values[1], cost_unit=values[2], amount=-values[3], amount_unit=values[4], class_=1, cost_per_kg=-values[5]/100))
    
    # Generate Streams for by-products
    ## Find the start index of the by-products
    for index, value in first_column_s.items():
        idx_by_products = 0
        if value == 'BY-PRODUCT CREDITS':
            idx_by_products_start = index+1
            break
    ## Only look for raw materials if they exist and find end index of materials.
    if idx_by_products_start>0:
        for index, value in first_column_s[idx_by_products_start:].items():
            if pd.isna(value):
                idx_by_products_end = index-1
                break 
    # Get Dataframe with all raw materials
    by_products_df = pd.read_excel(file, sheet_name=0).iloc[idx_by_products_start:idx_by_products_end+1]
    for by_products_idx, values in by_products_df.iterrows():
        streams.append(Stream(name=values[0], 
                              cost=values[1],
                              cost_unit=values[2],
                              amount=-values[3],
                              amount_unit=values[4],
                              class_=1,
                              cost_per_kg=values[5]/100))

    # Generate Streams for utilities
    ## Find the start index of the by-products
    for index, value in first_column_s.items():
        idx_utilities = 0
        if value == 'UTILITIES':
            idx_utilities_start = index+1
            break
    ## Only look for raw materials if they exist and find end index of materials.
    if idx_utilities_start>0:
        for index, value in first_column_s[idx_utilities_start:].items():
            if pd.isna(value):
                idx_utilities_end = index-1
                break 
    # Get Dataframe with all raw materials
    utilities_df = pd.read_excel(file, sheet_name=0).iloc[idx_utilities_start:idx_utilities_end+1]
    for utilities_idx, values in utilities_df.iterrows():
        streams.append(Stream(name=values[0], 
                              cost=values[1],
                              cost_unit=values[2],
                              amount=-values[3],
                              amount_unit=values[4],
                              class_=1,
                              cost_per_kg=values[5]/100))
    convert_units(streams) 
    return streams
    #pprint(streams)



'''
# get_stream function with unused iteration variables like raw_material_idx, idx_by_products, by_products_idx, idx_utilities, utilities_idx
def get_streams(file: Path) -> list[Stream]:
    streams: list[Stream] = []
    first_column_s = pd.read_excel(file, sheet_name=0).iloc[:, 0]
    string = first_column_s[2]

    # Find the main product name
    main_name_search = re.search(r"Product: (.*?),", string)
    if main_name_search:
        main_name = main_name_search.group(1)  # Extracts the main product name
    else:
        raise RuntimeError('The regex search for the main product name failed.')

    # Find the main product's price, if it exists
    price_with_unit_search = re.search(r"Price: (.*?),", string)
    if price_with_unit_search:
        price_with_unit = price_with_unit_search.group(1)
        main_cost = float(price_with_unit.split(' ')[0])
        main_cost_unit = price_with_unit.split(' ')[1]
    else:
        main_cost = np.nan
        main_cost_unit = "$/kg"

    # Define main product amount and unit
    main_amount = 1
    string2 = pd.read_excel(file, sheet_name=0).iloc[6, 3]
    main_amount_unit_search = re.search(r"per (.*$)", string2)
    if main_amount_unit_search:
        main_amount_unit = main_amount_unit_search.group(0)
    else:
        raise RuntimeError("Error finding the main amount unit.")

    # Add main product to the list of streams
    streams.append(Stream(name=main_name, cost=main_cost, cost_unit=main_cost_unit, amount=main_amount,
                          amount_unit=main_amount_unit, cost_per_kg=np.nan, class_=1))

    # Generate streams for Raw Materials, By-products, and Utilities

    # Raw Materials
    idx_raw_materials_start = first_column_s[first_column_s == 'RAW MATERIALS'].index[0] + 1
    idx_raw_materials_end = first_column_s[idx_raw_materials_start:][first_column_s.isna()].index[0] - 1
    raw_materials_df = pd.read_excel(file, sheet_name=0).iloc[idx_raw_materials_start:idx_raw_materials_end + 1]
    
    for _, values in raw_materials_df.iterrows():
        streams.append(Stream(name=values[0], cost=values[1], cost_unit=values[2], amount=-values[3],
                              amount_unit=values[4], class_=1, cost_per_kg=-values[5] / 100))

    # By-products
    idx_by_products_start = first_column_s[first_column_s == 'BY-PRODUCT CREDITS'].index[0] + 1
    idx_by_products_end = first_column_s[idx_by_products_start:][first_column_s.isna()].index[0] - 1
    by_products_df = pd.read_excel(file, sheet_name=0).iloc[idx_by_products_start:idx_by_products_end + 1]
    
    for _, values in by_products_df.iterrows():
        streams.append(Stream(name=values[0], cost=values[1], cost_unit=values[2], amount=-values[3],
                              amount_unit=values[4], class_=1, cost_per_kg=values[5] / 100))

    # Utilities
    idx_utilities_start = first_column_s[first_column_s == 'UTILITIES'].index[0] + 1
    idx_utilities_end = first_column_s[idx_utilities_start:][first_column_s.isna()].index[0] - 1
    utilities_df = pd.read_excel(file, sheet_name=0).iloc[idx_utilities_start:idx_utilities_end + 1]

    for _, values in utilities_df.iterrows():
        streams.append(Stream(name=values[0], cost=values[1], cost_unit=values[2], amount=-values[3],
                              amount_unit=values[4], class_=2, cost_per_kg=-values[5] / 100))

    # Convert units
    convert_units(streams)

    return streams
'''