import numpy as np

def write2model(process_list):
    Model = {}

    # Create empty sheets
    technical_flows = []
    A_distribution = []
    A_mean = []
    A_std_dev = []
    factor_requirements = []
    F_distribution = []
    F_mean = []
    F_std_dev = []

    processes = []

    for i in range(len(process_list)):
        # Add Process Info
        process_info = process_list[i].info
        processes.append([
            process_info.name,
            process_info.abbrevation,
            process_info.process_description,
            process_info.mainflow,
            process_info.location,
            process_info.exact_location,
            process_info.capacity,
            process_info.unit_per_year,
            '', ''  # Comments and type placeholders
        ])

        # Technical Flows
        for k in range(len(process_list[i].streams)):
            flag = 1
            for j in range(len(technical_flows)):
                stream = process_list[i].streams[k]
                if (technical_flows[j][0] == stream.name and 
                    technical_flows[j][1] == stream.stream_class or k == 1 and 
                    technical_flows[j][5] == stream.amount_unit):

                    # Ensure lists are large enough
                    if len(A_distribution) <= j:
                        A_distribution.append([None] * len(process_list))
                        A_mean.append([None] * len(process_list))
                        A_std_dev.append([None] * len(process_list))

                    A_distribution[j][i] = 2
                    A_mean[j][i] = stream.amount if not np.isnan(stream.amount) else stream.cost_per_kg
                    A_std_dev[j][i] = 0
                    flag = 0
                    break

            # Add new flow to technical_flows if not found
            if flag:
                stream = process_list[i].streams[k]
                technical_flow_entry = [
                    stream.name, 
                    stream.stream_class, 
                    '', '',  # Concentration, CAS
                    stream.unit_type if not np.isnan(stream.cost_unit) else 'Pieces',
                    stream.amount_unit if not np.isnan(stream.cost_unit) else '$',
                    stream.cost if not np.isnan(stream.cost) else 1,
                    0, 0,  # LHV, HHV
                    0, '', '', '', '', '', ''  # Additional fields
                ]
                technical_flows.append(technical_flow_entry)

                # Update distributions and mean values
                j = len(technical_flows) - 1
                A_distribution.append([None] * len(process_list))
                A_mean.append([None] * len(process_list))
                A_std_dev.append([None] * len(process_list))

                A_distribution[j][i] = 2
                A_mean[j][i] = stream.amount if not np.isnan(stream.amount) else stream.cost_per_kg
                A_std_dev[j][i] = 0

        # Factor requirement flows
        for k in range(len(process_list[i].costs)):
            flag = 1
            for j in range(len(factor_requirements)):
                cost = process_list[i].costs[k]
                if factor_requirements[j][0] == cost.name:
                    if len(F_distribution) <= j:
                        F_distribution.append([None] * len(process_list))
                        F_mean.append([None] * len(process_list))
                        F_std_dev.append([None] * len(process_list))

                    F_distribution[j][i] = 2
                    F_mean[j][i] = cost.value
                    F_std_dev[j][i] = 0
                    flag = 0
                    break

            # Add new flow to factor_requirements if not found
            if flag:
                cost = process_list[i].costs[k]
                factor_requirement_entry = [cost.name, cost.unit, '']
                factor_requirements.append(factor_requirement_entry)

                j = len(factor_requirements) - 1
                F_distribution.append([None] * len(process_list))
                F_mean.append([None] * len(process_list))
                F_std_dev.append([None] * len(process_list))

                F_distribution[j][i] = 2
                F_mean[j][i] = cost.value
                F_std_dev[j][i] = 0

    # Create k matrix
    k_mean = [[1] for _ in range(len(F_mean))]
    k_distribution = [[2] for _ in range(len(F_mean))]
    k_std_dev = [[0] for _ in range(len(F_mean))]

    # Add legends
    tech_name = [
        ['name', 'category', 'concentration/purity', 'CAS-Nr.', 'unit(choice)', 
         '[unit choice]', 'Market price', 'LHV', 'HHV', 'chemical formula', 
         'location(choice)', 'exact location', 'comments', 'SMILES CODE', 
         'molecular mass', 'flowCategory', 'flowSubcategory']
    ]
    
    process_names = [
        ['name', 'abbrevation', 'process description', 'mainflow', 
         'location (choice)', 'exact location', 'capacity', 'unit per year (choice)', 
         'comments', 'type']
    ]
    
    req_names = [['name', 'unit', 'comment']]
    cost_name = [['name', 'cost']]  # Defined cost_name to be used with cost-related data

    # Concatenate legends with actual data
    processes = process_names + processes
    technical_flows = tech_name + technical_flows
    factor_requirements = req_names + factor_requirements

    # Assign matrices to Model structure
    Model['meta_data_flows'] = technical_flows
    Model['meta_data_processes'] = processes

    # Handle empty cells in A_mean and convert to numpy array
    A_mean = [[0 if item is None else item for item in row] for row in A_mean]
    A_mean = np.array(A_mean)
    Model['matrices'] = {'A': {'mean_values': A_mean}}

    # Same for F_mean
    F_mean = [[0 if item is None else item for item in row] for row in F_mean]
    F_mean = np.array(F_mean)
    Model['matrices']['F'] = {'mean_values': F_mean}

    # Same for k_mean, k_distribution, k_std_dev
    k_mean = [[0 if item is None else item for item in row] for row in k_mean]
    k_mean = np.array(k_mean)
    Model['matrices']['k'] = {
        'mean_values': k_mean,
        'distribution': np.array(k_distribution),
        'std_dev': np.array(k_std_dev)
    }

    # Use cost_name to create a table or include it in the Model
    Model['meta_data_costs'] = cost_name  # Added cost_name as part of Model

    return Model
