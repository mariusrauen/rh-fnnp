function [Model, elementary_flows_to_delete,elementary_flow_matching] = make_process_adding_V2(Model, process_adding, correspondanceFile, pathGlobalInput , ecoinventVersion)

[ Model,elementary_flows_to_delete,elementary_flow_matching ] = get_added_processes_V2( Model , process_adding, correspondanceFile, pathGlobalInput , ecoinventVersion);

end