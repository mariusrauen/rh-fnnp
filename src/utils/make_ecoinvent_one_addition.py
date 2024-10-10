from get_match_table import get_match_table
from get_list_ecoinvent import get_list_ecoinvent
from get_ecoinvent_matrices import get_ecoinvent_matrices
from manipulate_matrixes import manipulate_matrixes

def make_ecoinvent_one_addition(Model, path_backup, path_input_regionalized_models, regions, allocation_ecoinvent, matchingTableName):
    # Step 1: Get match data
    match = get_match_table(path_input_regionalized_models, matchingTableName)
    
    # Step 2: Get Ecoinvent Data
    Ecoinvent_data = get_list_ecoinvent(path_backup)
    
    # Step 3: Create Ecoinvent matrices and manipulate model matrices
    B_eco_mean, A_eco_mean, F_eco_mean, eco_process_meta = get_ecoinvent_matrices(Model, Ecoinvent_data, match, regions)
    Model_ecoinvent = manipulate_matrixes(Model, B_eco_mean, A_eco_mean, F_eco_mean, eco_process_meta)
    
    return Model_ecoinvent
