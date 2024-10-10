import os
import pandas as pd
import numpy as np

def get_process_adding(path_alignments):
    # List files in the specified directory
    adding_files = [f for f in os.listdir(path_alignments) if os.path.isfile(os.path.join(path_alignments, f))]

    add_manual_processes = []
    counter = 0

    for i in range(2, len(adding_files)):  # Start from index 2 (equivalent to MATLAB starting from 3)
        print(adding_files[i])
        filename = os.path.join(path_alignments, adding_files[i])

        process_adding = {}

        # Process names (meta_data_processes)
        process_adding['meta_data_processes'] = pd.read_excel(filename, sheet_name='Process_meta_data', header=None).values.tolist()
        process_adding['meta_data_processes'] = [[None if pd.isna(cell) else cell for cell in row] for row in process_adding['meta_data_processes']]

        # Sheet SUMMARY A
        process_adding['A'] = {}
        process_adding['A']['mean_values'] = pd.read_excel(filename, sheet_name='SUMMARY A', usecols="D:IA", skiprows=1, nrows=198).values
        types = pd.read_excel(filename, sheet_name='SUMMARY A', usecols="A:C", skiprows=1, nrows=198).values
        process_adding['meta_data_flows'] = pd.read_excel(filename, sheet_name='SUMMARY A', usecols="A:C", skiprows=1, nrows=198).values.tolist()

        if process_adding['A']['mean_values'].size > 0:
            process_adding['A']['mean_values'] = np.nan_to_num(process_adding['A']['mean_values'], nan=0.0)

        if len(types) > 0:
            for idx, val in enumerate(types):
                process_adding['meta_data_flows'][idx][1] = val

        # Sheet SUMMARY B
        process_adding['B'] = {}
        process_adding['B']['mean_values'] = pd.read_excel(filename, sheet_name='SUMMARY B', usecols="D:IA", skiprows=1, nrows=498).values
        process_adding['meta_data_elementary_flows'] = pd.read_excel(filename, sheet_name='SUMMARY B', usecols="A:C", skiprows=1, nrows=498).values.tolist()

        if process_adding['B']['mean_values'].size > 0:
            process_adding['B']['mean_values'] = np.nan_to_num(process_adding['B']['mean_values'], nan=0.0)

        # Sheet SUMMARY F
        process_adding['F'] = {}
        process_adding['F']['mean_values'] = pd.read_excel(filename, sheet_name='SUMMARY F', usecols="C:IA", skiprows=1, nrows=198).values
        process_adding['meta_data_factor_requirements'] = pd.read_excel(filename, sheet_name='SUMMARY F', usecols="A:C", skiprows=1, nrows=198).values.tolist()

        if process_adding['F']['mean_values'].size > 0:
            process_adding['F']['mean_values'] = np.nan_to_num(process_adding['F']['mean_values'], nan=0.0)

        # Add the processed data to the list
        counter += 1
        add_manual_processes.append(process_adding)

    return add_manual_processes, adding_files
