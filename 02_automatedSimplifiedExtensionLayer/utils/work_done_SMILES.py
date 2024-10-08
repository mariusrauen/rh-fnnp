def work_done_SMILES(Model):
    # Calculate how many flows were already assessed by words

    work_to_do = 0  # Initialize counter for work still to be done
    flow_amounts = len(Model['meta_data_flows'][1:])  # Number of flows excluding the header (row 1 in MATLAB)
    counter = 0  # Counter for missing flows
    flow_name_missing_CAS = []  # Initialize an empty list for missing CAS flows

    # Loop through all rows starting from index 1 to match MATLAB's i = 2 (skipping header)
    for i in range(1, len(Model['meta_data_flows'])):
        if not Model['meta_data_flows'][i][13]:  # Check if the 14th column (index 13) is empty
            work_to_do += 1
            counter += 1
            flow_name_missing_CAS.append(Model['meta_data_flows'][i][0])  # Add the flow name to the list

    flow_name_missing_CAS = list(flow_name_missing_CAS)  # Ensure it's a list
    progress = (flow_amounts - work_to_do) / flow_amounts  # Calculate the progress

    return progress, flow_name_missing_CAS
