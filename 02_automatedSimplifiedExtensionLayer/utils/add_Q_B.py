import os
import numpy as np
import scipy.io

def add_Q_B(Model, PathEcoinvent):
    # Load the Q matrix from the .mat file
    mat_data = scipy.io.loadmat(os.path.join(PathEcoinvent, 'Q.mat'))
    Q = mat_data['Q']

    # Access the A matrix from the Model
    A = Model['matrices']['A']['mean_values']

    # Add B - metadata for elementary flows
    meta_data_elementary_flows = ['name', 'compartment', 'sub compartment', 'unit']

    # Horizontally concatenate the column names and units from Q
    elementary_flows = np.hstack((Q['col_name'], Q['units']))

    # Add metadata for elementary flows to the Model
    Model['meta_data_elementary_flows'] = [meta_data_elementary_flows] + elementary_flows.tolist()

    # Initialize the B matrix with zeros, matching the size of elementary_flows and A
    Model['matrices']['B'] = {
        'mean_values': np.zeros((elementary_flows.shape[0], A.shape[1]))
    }

    # Add Q matrix metadata
    Model['meta_data_impact_categories'] = Q['row_name'].tolist()
    Model['matrices']['Q'] = {
        'mean_values': Q['mean_values']
    }

    return Model