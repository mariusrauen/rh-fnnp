import numpy as np

def get_price_matrix_A(Model):
    possible = 1

    # Get price vector
    price_vector = np.array([float(flow[6]) for flow in Model['meta_data_flows'][1:]])

    # Check if size of price_vector matches and all prices are non-negative
    if price_vector.shape[0] < len(Model['meta_data_flows'][1:, 6]) or not np.any(price_vector <= 0):
        possible = 0
        raise ValueError('No price allocation possible, because not all prices are given or one price is zero.')

    # Get A matrix and set all negative values to 0
    A = Model['matrices']['A']['mean_values'].copy()
    A[A < 0] = 0

    # Create price matrix by repeating price_vector across columns
    price_matrix = np.tile(price_vector[:, np.newaxis], (1, A.shape[1]))

    # Compute Amatrixforallocationfactors
    Amatrixforallocationfactors = A * price_matrix

    # Test if allocation according to price is needed and reset 'allocation types' to 2
    # without utilities
    lns_utilities = np.array([int(flow[1]) == 2 for flow in Model['meta_data_flows'][1:]])
    A[lns_utilities, :] = 0

    # if there is only one output, no allocation needs to be performed
    counter = A.copy()
    counter[counter > 0] = 1
    counter = np.sum(counter, axis=0)
    lns_oneoutput = counter == 1
    A[:, lns_oneoutput] = 0
    A[A > 0] = price_matrix[A > 0]

    # Reset 'allocation types' for these processes to 0
    Model['meta_data_processes'][9][np.r_[False, lns_oneoutput]] = 0

    # Get all processes that might be allocated according to price
    allocationType = np.full(len(Model['meta_data_processes'][9]), np.nan)
    idx_strings = [isinstance(val, str) for val in Model['meta_data_processes'][9]]
    allocationType[~np.array(idx_strings)] = [int(val) for val in Model['meta_data_processes'][9] if not isinstance(val, str)]
    allocationType[np.array(idx_strings)] = 0
    lns_pricealloc = allocationType[1:] == 2
    A[:, ~lns_pricealloc] = 0

    # Find all relevant processes
    nonzero_cols = ~(np.all(A == 0, axis=0))
    A_nonzero_cols = A[:, nonzero_cols]

    # Get max, min, and factor between output prices
    maxA = np.max(A_nonzero_cols, axis=0)
    minA = np.zeros(maxA.shape)
    for i in range(minA.shape[0]):
        A_min = A_nonzero_cols[:, i]
        minA[i] = np.min(A_min[A_min != 0])

    factor = maxA / minA

    # Check if any factor is <= 0 or NaN
    if np.any(factor <= 0) or np.any(np.isnan(factor)):
        raise ValueError('Some price allocation factors are non-positive or NaN.')

    # Create the full factor vector and extract only those where allocation factor is > 5
    fullFactor = np.zeros(A.shape[1])
    fullFactor[nonzero_cols] = factor
    priceAllocation = fullFactor > 5

    # Set values to price allocation
    Model['meta_data_processes'][9][np.r_[False, priceAllocation]] = 3

    return Model, Amatrixforallocationfactors, possible

