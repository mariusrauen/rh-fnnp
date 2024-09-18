function [] = generate_results(Model,...
    path_output_folder_full,...
    missing_meta_data)

 
% generate_missing_meta_data
[~,~,meta_data_to_update,flows_in_processes] = make_flow_revision(Model, missing_meta_data);
xlswrite([path_output_folder_full,'\','missing_meta_data.xlsx'],...
    meta_data_to_update);

% generate_flows_in_processes
xlswrite([path_output_folder_full,'\','flows_in_processes.xlsx'],...
    flows_in_processes.inputs,'inputs');
xlswrite([path_output_folder_full,'\','flows_in_processes.xlsx'],...
    flows_in_processes.outputs,'outputs');

% generate_processes
[processes]=get_processes(Model);
[processes_outputs]=get_processes_outputs(Model);
xlswrite([path_output_folder_full,'\','processes.xlsx'],processes);
xlswrite([path_output_folder_full,'\','processes_outputs.xlsx'],processes_outputs);

% generate_meta_data_full
xlswrite([path_output_folder_full,'\','meta_data_full.xlsx'],Model.meta_data_flows)

%% needed from ecoinvent
[needed_from_ecoinvent] = get_flows_needed_ecoinvent(Model);
xlswrite([path_output_folder_full,'\','needed_from_ecoinvent.xlsx'],needed_from_ecoinvent)


% generate full data of model
output_proportions(Model, path_output_folder_full)

end