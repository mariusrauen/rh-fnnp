function [Model_ecoinvent] = make_ecoinvent_one_addition(Model, path_backup,path_input_regionalized_models, regions, allocation_ecoinvent,matchingTableName)

%% get data (exchange later)

[ match ] = get_match_table(path_input_regionalized_models,matchingTableName);

%% Get Ecoinvent Data

[ Ecoinvent_data ] = get_list_ecoinvent(path_backup);

%%  Create ecoinvent matrices and manipulate model matrices

[B_eco_mean, A_eco_mean, F_eco_mean,eco_process_meta] = get_ecoinvent_matrices(Model,Ecoinvent_data,match,regions);
[Model_ecoinvent] = manipulate_matrixes(Model, B_eco_mean, A_eco_mean, F_eco_mean, eco_process_meta);

end