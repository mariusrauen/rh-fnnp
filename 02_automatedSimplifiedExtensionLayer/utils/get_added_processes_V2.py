import numpy as np
import pandas as pd

def get_added_processes_V2(Model, process_adding, correspondanceFile, pathGlobalInput, ecoinventVersion):
    # MANIPULATE MATRICES
    a = process_adding['A']['mean_values'].shape[1]
    b = process_adding['B']['mean_values'].shape[1]
    f = process_adding['F']['mean_values'].shape[1]
    pd_size = process_adding['meta_data_processes'].shape[1] - 1
    
    # catch problems
    if (process_adding['A']['mean_values'].size == 0 or 
        process_adding['meta_data_processes'].shape[1] == 1 or
        (a != b and b > 0) or 
        (a != f and f > 0) or
        (a != pd_size and pd_size > 0)):
        print('Process adding is cancelled, because provided data is either empty or sizes of matrices do not fit.')
        print('Check data given. If no processes should be added, ignore this message.')
        return Model, [], []

    # Find row numbers of flows in A
    for i in range(process_adding['A']['mean_values'].shape[0]):
        found = False
        for j in range(1, Model['meta_data_flows'].shape[0]):
            if (process_adding['meta_data_flows'][i, 0] == Model['meta_data_flows'][j, 0] and 
                process_adding['meta_data_flows'][i, 1] == Model['meta_data_flows'][j, 1] and 
                process_adding['meta_data_flows'][i, 2] == Model['meta_data_flows'][j, 5]):
                
                process_adding['A']['flows']['row_indices'][i] = j
                found = True
                break

        if not found:
            # Create missing flow meta data
            unit_cat = None
            if process_adding['meta_data_flows'][i, 2] == 'kg':
                unit_cat = 'Mass'
            elif process_adding['meta_data_flows'][i, 2] == 'Nm3':
                unit_cat = 'Volumen'
            elif process_adding['meta_data_flows'][i, 2] == 'MJ':
                unit_cat = 'Energy'
            elif process_adding['meta_data_flows'][i, 2] == 'tkm':
                unit_cat = 'ton-kilometer'

            new_line_meta_data_flows = [
                process_adding['meta_data_flows'][i, 0],
                process_adding['meta_data_flows'][i, 1],
                '\x01', '\x01', unit_cat, process_adding['meta_data_flows'][i, 2], None, None, None, None, None, None, None, None, None, None, None
            ]
            Model['meta_data_flows'] = np.vstack([Model['meta_data_flows'], new_line_meta_data_flows])

            # Create new flow line in A
            Model['matrices']['A']['mean_values'] = np.vstack([Model['matrices']['A']['mean_values'], np.zeros(Model['matrices']['A']['mean_values'].shape[1])])
            process_adding['A']['flows']['row_indices'][i] = Model['matrices']['A']['mean_values'].shape[0] - 1

    oldEcoinventVersion = process_adding['meta_data_processes'][-1, 1]
    if oldEcoinventVersion != ecoinventVersion:
        mappingFile = pd.read_excel(f"{pathGlobalInput}/{correspondanceFile}", sheet_name=f"{oldEcoinventVersion} - {ecoinventVersion}")
        rowNames = pd.read_excel(f"{pathGlobalInput}/{correspondanceFile}", sheet_name=f"{oldEcoinventVersion} - {ecoinventVersion}")
        mappingFile.columns = rowNames.iloc[0, :]
    
    # find row numbers of flows in B
    for i in range(process_adding['B']['mean_values'].shape[0]):
        found = False
        for j in range(1, Model['meta_data_elementary_flows'].shape[0]):
            if (process_adding['meta_data_elementary_flows'][i, 0] == Model['meta_data_elementary_flows'][j, 0] and
                process_adding['meta_data_elementary_flows'][i, 1] == Model['meta_data_elementary_flows'][j, 1] and
                process_adding['meta_data_elementary_flows'][i, 2] == Model['meta_data_elementary_flows'][j, 2]):

                process_adding['B']['flows']['row_indices'][i] = j
                found = True
                break
        
        if not found:
            # Handle mapping
            match = mappingFile[
                (mappingFile[f'name_{oldEcoinventVersion}'] == process_adding['meta_data_elementary_flows'][i, 0]) &
                (mappingFile[f'compartment_{oldEcoinventVersion}'] == process_adding['meta_data_elementary_flows'][i, 1]) &
                (mappingFile[f'subcompartment_{oldEcoinventVersion}'] == process_adding['meta_data_elementary_flows'][i, 2])
            ].iloc[0]

            process_adding['meta_data_elementary_flows'][i, 0] = match[f'name_{ecoinventVersion}']
            process_adding['meta_data_elementary_flows'][i, 1] = match[f'compartment_{ecoinventVersion}']
            process_adding['meta_data_elementary_flows'][i, 2] = match[f'subcompartment_{ecoinventVersion}']

            if process_adding['meta_data_elementary_flows'][i, 0] == "":
                process_adding['B']['flows']['row_indices'][i] = 0
            else:
                for j in range(1, Model['meta_data_elementary_flows'].shape[0]):
                    if (process_adding['meta_data_elementary_flows'][i, 0] == Model['meta_data_elementary_flows'][j, 0] and
                        process_adding['meta_data_elementary_flows'][i, 1] == Model['meta_data_elementary_flows'][j, 1] and
                        process_adding['meta_data_elementary_flows'][i, 2] == Model['meta_data_elementary_flows'][j, 2]):

                        process_adding['B']['flows']['row_indices'][i] = j
                        break

    # Delete rows without a found flow
    elementary_flow_matching = process_adding['meta_data_elementary_flows']
    
    if process_adding['B']['mean_values'].shape[0] > 0:
        rows_to_delete = process_adding['B']['flows']['row_indices'] == 0
        elementary_flows_to_delete = process_adding['meta_data_elementary_flows'][rows_to_delete, 2:6]
        process_adding['meta_data_elementary_flows'] = process_adding['meta_data_elementary_flows'][~rows_to_delete, :3]
        process_adding['B']['mean_values'] = process_adding['B']['mean_values'][~rows_to_delete, :]
        process_adding['B']['flows']['row_indices'] = process_adding['B']['flows']['row_indices'][~rows_to_delete]
    else:
        elementary_flows_to_delete = []

    # Additional processing for F (similar to above handling for A and B)
    # ...

    return Model, elementary_flows_to_delete, elementary_flow_matching
