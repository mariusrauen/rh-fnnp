def manipulate_matrixes(Model, B_eco_mean, A_eco_mean, F_eco_mean, eco_process_meta):
    # Manipulate matrices and include B_mean values

    # Meta data processes
    Model['meta_data_processes'] = Model['meta_data_processes'] + eco_process_meta

    # A matrix
    for row_idx, row in enumerate(A_eco_mean):
        Model['matrices']['A']['mean_values'][row_idx].extend(row)

    # B matrix
    B_eco_mean_zero = [[0] * len(B_eco_mean[0]) for _ in range(len(Model['matrices']['B']['mean_values']))]
    for i in range(len(B_eco_mean)):
        B_eco_mean_zero[i] = B_eco_mean[i]

    for row_idx, row in enumerate(B_eco_mean_zero):
        Model['matrices']['B']['mean_values'][row_idx].extend(row)

    # F matrix
    F_high = [row + [0] * len(F_eco_mean[0]) for row in Model['matrices']['F']['mean_values']]
    F_low = [[0] * len(Model['matrices']['F']['mean_values'][0]) + row for row in F_eco_mean]
    Model['matrices']['F']['mean_values'] = F_high + F_low

    # Modify meta_data_factor_requirements
    purchase = [f"purchase {process}" for process in eco_process_meta[3]]
    row_name_add_F = [[purchase[i], '$', None] for i in range(len(purchase))]

    Model['meta_data_factor_requirements'] += row_name_add_F

    # Modify k matrix
    Model['matrices']['k']['mean_values'] = [[1] for _ in range(len(Model['matrices']['F']['mean_values']))]

    return Model
