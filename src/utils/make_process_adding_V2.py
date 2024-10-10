from get_added_processes_V2 import get_added_processes_V2

def make_process_adding_V2(Model, process_adding, correspondanceFile, pathGlobalInput, ecoinventVersion):
    # Call the equivalent Python function for get_added_processes_V2
    Model, elementary_flows_to_delete, elementary_flow_matching = get_added_processes_V2(
        Model, process_adding, correspondanceFile, pathGlobalInput, ecoinventVersion
    )
    
    return Model, elementary_flows_to_delete, elementary_flow_matching
