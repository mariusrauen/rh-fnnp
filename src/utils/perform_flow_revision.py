def perform_flow_revision(Model, all_flows_meta_data):
    # First: delete possible space at the end of flow names in Model meta data and all_flows_meta_data
    # Assumption: space at the end of the flow name is a mistake
    
    # Handle Model meta data (meta_data_flows)
    for i in range(1, len(Model['meta_data_flows'])):  # Python starts at index 0, so use range(1, ...) to match MATLAB's 2
        if Model['meta_data_flows'][i][0][-1] == ' ':  # Check if the last character is a space
            Model['meta_data_flows'][i][0] = Model['meta_data_flows'][i][0][:-1]  # Remove the last character (space)
    
    # Handle all_flows_meta_data
    for i in range(1, len(all_flows_meta_data)):  # Again, start at index 1 for MATLAB's 2
        if all_flows_meta_data[i][0][-1] == ' ':  # Check if the last character is a space
            all_flows_meta_data[i][0] = all_flows_meta_data[i][0][:-1]  # Remove the last character (space)

    # Create string list from Model's meta_data_flows
    string_list = [row[0] for row in Model['meta_data_flows']]

    # Add 'molecular mass' in the 15th column of the first row (index 14 in Python)
    Model['meta_data_flows'][0][14] = 'molecular mass'

    # Iterate through each row of the Model meta data flows
    for i in range(1, len(Model['meta_data_flows'])):
        # Find rows in all_flows_meta_data that match the flow name from the Model meta data flows
        row_flow = [index for index, row in enumerate(all_flows_meta_data) if row[0] == string_list[i]]

        # Continue if no matching row was found
        if not row_flow:
            continue

        # Go through each found row
        for k in row_flow:
            # Check if units (6th column, index 5) and types (2nd column, index 1) are equal
            if Model['meta_data_flows'][i][5] == all_flows_meta_data[k][5] and Model['meta_data_flows'][i][1] == all_flows_meta_data[k][1]:
                # Overwrite all meta data
                Model['meta_data_flows'][i] = all_flows_meta_data[k]

    return Model
