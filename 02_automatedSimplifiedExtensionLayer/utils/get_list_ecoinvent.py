from scipy.io import loadmat
import os

def get_list_ecoinvent(path_inputs):
    # Construct the full path to the Ecoinvent_data.mat file
    file_path = os.path.join(path_inputs, 'Ecoinvent_data.mat')
    
    # Load the .mat file using scipy.io.loadmat
    data = loadmat(file_path)
    
    # Extract the 'Ecoinvent_data' variable from the .mat file
    Ecoinvent_data = data.get('Ecoinvent_data', None)  # Use .get() to avoid KeyError if the variable is missing
    
    return Ecoinvent_data
