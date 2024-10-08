import numpy as np
import pandas as pd

def get_added_processes_V2_Layer23(Model, process_adding):
    # Manipulate matrices
    a = process_adding['A']['mean_values'].shape[1]
    b = process_adding['B']['mean_values'].shape[1]
    f = process_adding['F']['mean_values'].shape[1]
    pd_len = process_adding['meta_data_processes'].shape[1] - 1

    # Check for mismatched matrix sizes
    if process_adding['A']['mean_values'].size == 0 or process_adding['meta_data_processes'].shape[1] == 1 or \
       (a != b and b > 0) or (a != f and f > 0) or (a != pd_len and pd_len > 0):
        print('Process adding is cancelled, because provided data is either empty or sizes of matrices do not fit.')
        print('Check data given. If no processes should be added, ignore this message.')
        return Model

    # Find row numbers of flows in A
    for i in range(process_adding['A']['mean_values'].shape[0]):
        found = False
        for j in range(1, Model['meta_data_flows'].shape[0]):
            if (process_adding['meta_data_flows'][i, 0] == Model['meta_data_flows'][j, 0] and
               process_adding['meta_data_flows'][i, 1] == Model['meta_data_flows'][j, 1] and
               process_adding['meta_data_flows'][i, 2] == Model['meta_data_flows'][j, 5]):
                process_adding['A']['flows']['row_indices'][i] = j - 1  # Save index
                found = True
                break
        
        if not found:  # If flow was not found, create a missing flow meta data entry
            unit_cat = ''
            if process_adding['meta_data_flows'][i, 2] == 'kg':
                unit_cat = 'Mass'
            elif process_adding['meta_data_flows'][i, 2] == 'Nm3':
                unit_cat = 'Volumen'
            elif process_adding['meta_data_flows'][i, 2] == 'MJ':
                unit_cat = 'Energy'
            elif process_adding['meta_data_flows'][i, 2] == 'tkm':
                unit_cat = 'ton-kilometer'

            new_line_meta_data_flows = [process_adding['meta_data_flows'][i, 0], process_adding['meta_data_flows'][i, 1],
                                        '', '', unit_cat, process_adding['meta_data_flows'][i, 2], '', '', '', '', '', '', '', '', '', '', '']
            Model['meta_data_flows'] = np.vstack([Model['meta_data_flows'], new_line_meta_data_flows])

            # Create new flow line in A
            Model['matrices']['A']['mean_values'] = np.vstack([Model['matrices']['A']['mean_values'], np.zeros((1, Model['matrices']['A']['mean_values'].shape[1]))])
            process_adding['A']['flows']['row_indices'][i] = Model['matrices']['A']['mean_values'].shape[0] - 1

    # Find row numbers of flows in B
    for i in range(process_adding['B']['mean_values'].shape[0]):
        found = False
        for j in range(1, Model['meta_data_elementary_flows'].shape[0]):
            if (process_adding['meta_data_elementary_flows'][i, 0] == Model['meta_data_elementary_flows'][j, 0] and
               process_adding['meta_data_elementary_flows'][i, 1] == Model['meta_data_elementary_flows'][j, 1] and
               process_adding['meta_data_elementary_flows'][i, 2] == Model['meta_data_elementary_flows'][j, 2]):
                process_adding['B']['flows']['row_indices'][i] = j - 1
                found = True
                break
        
        if not found:
            new_line_meta_data_elementary_flows = [process_adding['meta_data_elementary_flows'][i, 0],
                                                   process_adding['meta_data_elementary_flows'][i, 1],
                                                   process_adding['meta_data_elementary_flows'][i, 2], '', '', '']
            Model['meta_data_elementary_flows'] = np.vstack([Model['meta_data_elementary_flows'], new_line_meta_data_elementary_flows])
            Model['matrices']['B']['mean_values'] = np.vstack([Model['matrices']['B']['mean_values'], np.zeros((1, Model['matrices']['B']['mean_values'].shape[1]))])
            process_adding['B']['flows']['row_indices'][i] = Model['matrices']['B']['mean_values'].shape[0] - 1

    # Find row numbers of flows in F
    for i in range(process_adding['F']['mean_values'].shape[0]):
        found = False
        for j in range(1, Model['meta_data_factor_requirements'].shape[0]):
            if (process_adding['meta_data_factor_requirements'][i, 0] == Model['meta_data_factor_requirements'][j, 0] and
               process_adding['meta_data_factor_requirements'][i, 1] == Model['meta_data_factor_requirements'][j, 1]):
                process_adding['F']['flows']['row_indices'][i] = j - 1
                found = True
                break

        if not found:
            new_line_meta_data_factor_requirements = [process_adding['meta_data_factor_requirements'][i, 0],
                                                      process_adding['meta_data_factor_requirements'][i, 1], '']
            Model['meta_data_factor_requirements'] = np.vstack([Model['meta_data_factor_requirements'], new_line_meta_data_factor_requirements])
            Model['matrices']['F']['mean_values'] = np.vstack([Model['matrices']['F']['mean_values'], np.zeros((1, Model['matrices']['F']['mean_values'].shape[1]))])
            process_adding['F']['flows']['row_indices'][i] = Model['matrices']['F']['mean_values'].shape[0] - 1

    # Executed if all flows in all matrices exist
    for i in range(process_adding['meta_data_processes'].shape[1] - 1):
        if np.all(process_adding['A']['mean_values'][:, i] == 0) and \
           np.all(process_adding['B']['mean_values'][:, i] == 0) and \
           np.all(process_adding['F']['mean_values'][:, i] == 0):
            continue  # Skip this process if empty

        # Create new columns in all relevant matrices (A, B, F)
        Model['matrices']['A']['mean_values'] = np.hstack([Model['matrices']['A']['mean_values'], np.zeros((Model['matrices']['A']['mean_values'].shape[0], 1))])
        Model['matrices']['B']['mean_values'] = np.hstack([Model['matrices']['B']['mean_values'], np.zeros((Model['matrices']['B']['mean_values'].shape[0], 1))])
        Model['matrices']['F']['mean_values'] = np.hstack([Model['matrices']['F']['mean_values'], np.zeros((Model['matrices']['F']['mean_values'].shape[0], 1))])

        # Write meta-data of added process
        Model['meta_data_processes'] = np.hstack([Model['meta_data_processes'], process_adding['meta_data_processes'][:, [i + 1]]])

        # Write data in A.mean_values
        for j in range(process_adding['A']['mean_values'].shape[0]):
            if process_adding['A']['mean_values'][j, i] == 0:
                continue
            Model['matrices']['A']['mean_values'][process_adding['A']['flows']['row_indices'][j], -1] = process_adding['A']['mean_values'][j, i]

        # Write data in B.mean_values
        for j in range(process_adding['B']['mean_values'].shape[0]):
            if process_adding['B']['mean_values'][j, i] == 0:
                continue
            Model['matrices']['B']['mean_values'][process_adding['B']['flows']['row_indices'][j], -1] = process_adding['B']['mean_values'][j, i]

        # Write data in F.mean_values
        for j in range(process_adding['F']['mean_values'].shape[0]):
            if process_adding['F']['mean_values'][j, i] == 0:
                continue
            Model['matrices']['F']['mean_values'][process_adding['F']['flows']['row_indices'][j], -1] = process_adding['F']['mean_values'][j, i]

    return Model
