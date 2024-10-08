import numpy as np

def TransferModelToProcessAdding(Model):
    # Initialize process_adding dictionary
    process_adding = {}

    # Copy meta_data_processes
    process_adding['meta_data_processes'] = Model['meta_data_processes']

    # Copy A matrix and corresponding meta data
    process_adding['A'] = Model['matrices']['A']
    process_adding['meta_data_flows'] = np.column_stack((
        Model['meta_data_flows'][1:, 0],  # Flow names
        Model['meta_data_flows'][1:, 1],  # Flow types
        Model['meta_data_flows'][1:, 5]   # Flow units
    ))

    # Copy B matrix and corresponding meta data for elementary flows
    process_adding['B'] = Model['matrices']['B']
    process_adding['meta_data_elementary_flows'] = np.column_stack((
        Model['meta_data_elementary_flows'][1:, 0],  # Flow names
        Model['meta_data_elementary_flows'][1:, 1],  # Compartments
        Model['meta_data_elementary_flows'][1:, 2]   # Subcompartments
    ))

    # Find rows in B matrix where values are not all zero and filter them
    rows_not_zero = np.where(np.sum(process_adding['B']['mean_values'] != 0, axis=1) != 0)[0]
    process_adding['B']['mean_values'] = process_adding['B']['mean_values'][rows_not_zero, :]
    process_adding['meta_data_elementary_flows'] = process_adding['meta_data_elementary_flows'][rows_not_zero, :]

    # Copy F matrix and corresponding meta data for factor requirements
    process_adding['F'] = Model['matrices']['F']
    process_adding['meta_data_factor_requirements'] = np.column_stack((
        Model['meta_data_factor_requirements'][1:, 0],  # Requirement names
        Model['meta_data_factor_requirements'][1:, 1]   # Units
    ))

    return process_adding

