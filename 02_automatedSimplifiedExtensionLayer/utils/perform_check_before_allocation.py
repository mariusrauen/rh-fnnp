import numpy as np
import pandas as pd

def perform_check_before_allocation(Model):
    # Get A Matrix only with output values and without utilities
    A_alloc = Model['matrices']['A']['mean_values'].copy()
    A_alloc[A_alloc < 0] = 0  # only outputs

    # without utilities
    lns_utilities = np.array([int(flow[1]) == 2 for flow in Model['meta_data_flows'][1:]])
    A_alloc[lns_utilities, :] = 0

    # if there is only one output -> no allocation has to be performed
    counter = A_alloc.copy()
    counter[counter > 0] = 1
    counter = np.sum(counter, axis=0)
    lns_oneoutput = counter == 1
    A_alloc[:, lns_oneoutput] = 0

    # Get Only Processes where energy allocation should be performed
    allocationType = np.full(len(Model['meta_data_processes'][9]), np.nan)
    idx_strings = [isinstance(val, str) for val in Model['meta_data_processes'][9]]
    allocationType[~np.array(idx_strings)] = [int(val) for val in Model['meta_data_processes'][9] if not isinstance(val, str)]
    allocationType[np.array(idx_strings)] = 0

    lns_energyalloc = allocationType[1:] == 1
    A = A_alloc.copy()
    A[:, ~lns_energyalloc] = 0

    # Find all flows that still have positive outputs somewhere in A Matrix
    lns_alloc = np.sum(A, axis=1) != 0

    # Find all flows that are allocated by energy but don't have a LHV
    lns_pos = np.array([float(flow[8]) > 0 for flow in Model['meta_data_flows'][1:]])
    lns_neg = np.array([float(flow[8]) < 0 for flow in Model['meta_data_flows'][1:]])
    lns_missing = ~(lns_pos | lns_neg)

    lns_problematicflows = lns_missing & lns_alloc

    flows_missingLHV = []
    problematic_processes = []  # Initialize problematic_processes

    # If there are missing LHVs
    if np.any(lns_problematicflows):
        flows_missingLHV = [Model['meta_data_flows'][i + 1] for i, val in enumerate(lns_problematicflows) if val]
        VarNames = Model['meta_data_flows'][0]
        flows_missingLHV_df = pd.DataFrame(flows_missingLHV, columns=VarNames)

        # Find all problematic processes
        A[~lns_problematicflows, :] = 0
        row, col = np.nonzero(A)
        processName = [Model['meta_data_processes'][0][c + 1] for c in col]
        mainFlow = [Model['meta_data_processes'][3][c + 1] for c in col]
        problematicOutputFlow = [Model['meta_data_flows'][r + 1][0] for r in row]
        problematic_processes = pd.DataFrame({
            'processName': processName,
            'mainFlow': mainFlow,
            'problematicOutputFlow': problematicOutputFlow
        })
    else:
        flows_missingLHV_df = pd.DataFrame()
        problematic_processes = pd.DataFrame()  # Ensure problematic_processes is an empty DataFrame

    flows_missingLHV_energy = flows_missingLHV_df
    problematic_processes_energy = problematic_processes  # Now properly assigning problematic_processes

    # Get Only Processes where price allocation could be performed
    lns_pricealloc = allocationType[1:] == 2
    A = A_alloc.copy()
    A[:, ~lns_pricealloc] = 0

    # Find all flows that still have positive outputs somewhere in A Matrix
    lns_alloc = np.sum(A, axis=1) != 0

    # Find all flows that are allocated by price but don't have a LHV
    lns_pos = np.array([float(flow[6]) > 0 for flow in Model['meta_data_flows'][1:]])
    lns_neg = np.array([float(flow[6]) < 0 for flow in Model['meta_data_flows'][1:]])
    lns_missing = ~(lns_pos | lns_neg)

    lns_problematicflows = lns_missing & lns_alloc

    flows_missingLHV = []
    problematic_processes = []  # Re-initialize problematic_processes for price allocation

    # If there are missing LHVs
    if np.any(lns_problematicflows):
        flows_missingLHV = [Model['meta_data_flows'][i + 1] for i, val in enumerate(lns_problematicflows) if val]
        VarNames = Model['meta_data_flows'][0]
        flows_missingLHV_df = pd.DataFrame(flows_missingLHV, columns=VarNames)

        # Find all problematic processes
        A[~lns_problematicflows, :] = 0
        row, col = np.nonzero(A)
        processName = [Model['meta_data_processes'][0][c + 1] for c in col]
        mainFlow = [Model['meta_data_processes'][3][c + 1] for c in col]
        problematicOutputFlow = [Model['meta_data_flows'][r + 1][0] for r in row]
        problematic_processes = pd.DataFrame({
            'processName': processName,
            'mainFlow': mainFlow,
            'problematicOutputFlow': problematicOutputFlow
        })
    else:
        flows_missingLHV_df = pd.DataFrame()
        problematic_processes = pd.DataFrame()  # Ensure problematic_processes is an empty DataFrame

    flows_missingLHV_price = flows_missingLHV_df
    problematic_processes_price = problematic_processes  # Now properly assigning problematic_processes

    return flows_missingLHV_energy, problematic_processes_energy, flows_missingLHV_price, problematic_processes_price
