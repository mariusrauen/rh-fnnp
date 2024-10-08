def get_flow_analysis(Model):
    # Inputs
    flow_analysis_inputs = []  # Initialize list to store input flow analysis

    for i in range(1, len(Model['meta_data_flows'])):  # Start at 1 (Python is 0-based)
        row = [
            Model['meta_data_flows'][i][0],  # flow
            Model['meta_data_flows'][i][3],  # CAS
            Model['meta_data_flows'][i][1],  # type
            Model['meta_data_flows'][i][5]   # unit
        ]

        # Vector for processes with negative values (inputs)
        vector = (Model['matrices']['A']['mean_values'][i-1, :] < 0)
        col_names = Model['meta_data_processes'][0][1:]  # Skip first column
        processes = [col_names[j] for j, v in enumerate(vector) if v]

        # Append processes to the row
        row.extend(processes)
        flow_analysis_inputs.append(row)

    # Add headers for flow_analysis_inputs
    flow_analysis_inputs.insert(0, ['flow', 'CAS', 'type', 'unit', 'using processes'])

    # Outputs
    flow_analysis_outputs = []  # Initialize list to store output flow analysis

    for i in range(1, len(Model['meta_data_flows'])):
        row = [
            Model['meta_data_flows'][i][0],  # flow
            Model['meta_data_flows'][i][13],  # SMILES
            Model['meta_data_flows'][i][1],  # type
            Model['meta_data_flows'][i][5]   # unit
        ]

        # Vector for processes with positive values (outputs)
        vector = (Model['matrices']['A']['mean_values'][i-1, :] > 0)
        col_names = Model['meta_data_processes'][0][1:]  # Skip first column
        processes = [col_names[j] for j, v in enumerate(vector) if v]

        # Append processes to the row
        row.extend(processes)
        flow_analysis_outputs.append(row)

    # Add headers for flow_analysis_outputs
    flow_analysis_outputs.insert(0, ['flow', 'SMILES', 'type', 'unit', 'using processes'])

    # Save inputs and outputs in the result
    flows_in_processes = {
        'inputs': flow_analysis_inputs,
        'outputs': flow_analysis_outputs
    }

    return flows_in_processes
