import scipy.io as sio
import os

def make_waste_and_elementary_flows(Model, path_inputs):
    # Load the .mat files
    elements = sio.loadmat(os.path.join(path_inputs, 'elements.mat'))['elements']
    M_e = sio.loadmat(os.path.join(path_inputs, 'M_e.mat'))['M_e']
    raw = sio.loadmat(os.path.join(path_inputs, 'raw.mat'))['raw']
    
    # Call the function to get waste and elementary flows
    Model_waste = get_waste_and_elementary_flows_direct_emissions(Model, elements, M_e, raw)
    
    return Model_waste
