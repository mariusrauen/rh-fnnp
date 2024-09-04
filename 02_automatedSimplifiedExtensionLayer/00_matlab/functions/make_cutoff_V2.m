function [Model,Cutoff,processes] = ...
    make_cutoff_V2(Model, included_processes,cut_off_parameter_over_all,cut_off_parameter_each_flow)

%% remove processes and cut-off flows

% generate process table and exclude processes

[ processes ] = get_processes_V2(Model);

% [ processes ] = get_excluded_processes_V2(processes, included_processes);

% cut off and delete processes (values in brackets are cut-off rules, 5% 
% per process and 1% per flow)

[ Model, Cutoff ] = perform_cutoff_V2(Model,processes,cut_off_parameter_over_all,cut_off_parameter_each_flow); % values or cut_off rules



end