import numpy as np

def get_energy_matrix_A(Model):
    # Initialize possible
    possible = 1

    # Get heating values (conversion factors)
    conversion_factors = Model['meta_data_flows'][1:, 8].copy()  # 9th column in MATLAB is 8th in Python (0-based indexing)

    # Find missing conversion factors
    missing_conversion_factors = [i for i, val in enumerate(conversion_factors) if val == 'missing']

    # Set missing factors to zero
    if missing_conversion_factors:
        for idx in missing_conversion_factors:
            conversion_factors[idx] = 0

    # Set conversion factors of flows already in MJ to 1
    energy_flows = [i for i, val in enumerate(Model['meta_data_flows'][1:, 5]) if val == 'MJ']  # 6th column in MATLAB is 5th in Python

    if energy_flows:
        for idx in energy_flows:
            conversion_factors[idx] = 1

    # Convert conversion_factors to numeric (float) array
    energy_vector = np.array(conversion_factors, dtype=float)

    # Check if size of energy_vector matches the number of rows in meta_data_flows (for conversion factors)
    if energy_vector.shape[0] < Model['meta_data_flows'][1:, 8].shape[0]:
        possible = 0
        raise ValueError('No energy allocation possible, because not all prices are given or one price is zero.')

    # Get matrix A
    A = Model['matrices']['A']['mean_values'].copy()
    A[A < 0] = 0  # Set negative values to 0

    # Create energy matrix by repeating energy_vector across columns
    energy_matrix = np.tile(energy_vector[:, np.newaxis], (1, A.shape[1]))

    # Compute Amatrixforallocationfactors
    Amatrixforallocationfactors = A * energy_matrix

    return Amatrixforallocationfactors, possible
