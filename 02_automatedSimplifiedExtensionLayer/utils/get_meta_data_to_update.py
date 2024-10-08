def get_meta_data_to_update(data):
    counter = 0
    meta_data_to_update = []

    for i in range(1, len(data)):  # Start from 1 to match MATLAB's indexing (i=2 in MATLAB)

        if (not data[i][3] or    # CAS (column 4 in MATLAB, index 3 in Python)
            data[i][6] == 1 or   # price (column 7 in MATLAB, index 6 in Python)
            data[i][6] == 0 or   # price (column 7 in MATLAB, index 6 in Python)
            not data[i][6] or    # price (column 7 in MATLAB, index 6 in Python)
            not data[i][8] or    # HHV (column 9 in MATLAB, index 8 in Python)
            not data[i][9] or    # formula (column 10 in MATLAB, index 9 in Python)
            not data[i][14]):    # molecular mass (column 15 in MATLAB, index 14 in Python)

            counter += 1
            meta_data_to_update.append(data[i])  # Add the row to the list

    # Prepend the first row (headers) to meta_data_to_update
    meta_data_to_update.insert(0, data[0])

    if counter > 0:
        print('FLOW REVISION IS NEEDED. CHECK MISSING META DATA.')

    return meta_data_to_update
