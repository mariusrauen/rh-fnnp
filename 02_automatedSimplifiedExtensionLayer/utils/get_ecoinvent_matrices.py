import numpy as np

def get_ecoinvent_matrices(Model, Ecoinvent_data, match, regions):
    # Initialize matrices
    counter = 0
    B_eco_mean = np.empty((0, 0))  # Empty array for now
    A_eco_mean = np.zeros((Model['matrices']['A']['mean_values'].shape[0], len(match) - 1))
    F_eco_mean = np.eye(len(match) - 1)
    
    # Update match headers
    match[0][7] = 'Ecoinvent_process_column'
    match[0][8] = 'region'
    
    # Find columns of Ecoinvent processes
    for i in range(1, len(match)):
        found = False
        for j in range(Ecoinvent_data['col_names'].shape[0]):
            for k in range(len(regions)):
                if (match[i][3] == Ecoinvent_data['col_names'][j][1] and
                    match[i][4] == Ecoinvent_data['col_names'][j][0] and
                    regions[k] == Ecoinvent_data['col_names'][j][2] and
                    not found):
                    match[i][7] = j
                    match[i][8] = regions[k]
                    found = True

        if not found:
            print('WARNING, the following flow was not found:')
            print(match[i][0])
            print(match[i][3])
            print(match[i][4])
            match[i][7] = None
    
    # Make B_eco_mean, A_eco_mean, F_eco_mean
    for i in range(1, len(match)):
        counter += 1
        
        # Normal Ecoinvent processes
        if match[i][6] >= 0:
            B_eco_mean = np.hstack([B_eco_mean, Ecoinvent_data['mean_values'][:, match[i][7]] * match[i][6]])
            
            row_flow = np.where(Model['meta_data_flows'][:, 0] == match[i][0])[0]
            
            if row_flow.size == 0:
                continue
            
            for k in row_flow:
                if (Model['meta_data_flows'][k][1] == match[i][1] and
                    Model['meta_data_flows'][k][5] == match[i][2]):
                    A_eco_mean[k - 1, i - 1] = 1  # Overwrite line in A
                    
                    price = Model['meta_data_flows'][k][6]
                    if price is None:
                        price = 0
                    
                    F_eco_mean[i - 1, :] = F_eco_mean[i - 1, :] * price
        
        # Avoided burden
        elif match[i][6] < 0:
            B_eco_mean = np.hstack([B_eco_mean, Ecoinvent_data['mean_values'][:, match[i][7]] * -match[i][6]])
            
            row_flow = np.where(Model['meta_data_flows'][:, 0] == match[i][0])[0]
            
            if row_flow.size == 0:
                continue
            
            for k in row_flow:
                if (Model['meta_data_flows'][k][1] == match[i][1] and
                    Model['meta_data_flows'][k][5] == match[i][2]):
                    A_eco_mean[k - 1, i - 1] = -1  # Overwrite line in A
                    
                    price = Model['meta_data_flows'][k][6]
                    if price is None:
                        price = 0
                    
                    F_eco_mean[i - 1, :] = F_eco_mean[i - 1, :] * -price
    
    # Create meta-data for ecoinvent processes
    eco_process_meta = [
        match[1:, 3],
        [None] * (len(match) - 1),
        [None] * (len(match) - 1),
        match[1:, 0],
        match[1:, 8],
        [None] * (len(match) - 1),
        [None] * (len(match) - 1),
        [None] * (len(match) - 1),
        [None] * (len(match) - 1),
        [4] * (len(match) - 1)  # Specific case as per the original
    ]
    
    # Delete columns with only zeros in A_eco_mean
    delete_columns = np.all(A_eco_mean == 0, axis=0)
    
    A_eco_mean = A_eco_mean[:, ~delete_columns]
    B_eco_mean = B_eco_mean[:, ~delete_columns]
    F_eco_mean = F_eco_mean[:, ~delete_columns]
    
    eco_process_meta = np.array(eco_process_meta)[:, ~delete_columns]
    
    # Delete rows in F_eco_mean with only zeros
    delete_rows_F = np.all(F_eco_mean == 0, axis=1)
    F_eco_mean = F_eco_mean[~delete_rows_F, :]
    
    return B_eco_mean, A_eco_mean, F_eco_mean, eco_process_meta

