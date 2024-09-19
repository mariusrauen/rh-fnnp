function [Model_ecoinvent]=make_ecoinvent(Model, path_EcoMatch, path_input_regionalized_models, regions, allocation_ecoinvent,matchingTableName,EcoinventModelCoreLayer)
%
%% get data (exchange later)

[ match ] = get_match_table(path_EcoMatch,matchingTableName);

%% Get Ecoinvent Data

[ Ecoinvent_data ] = get_list_ecoinvent(path_input_regionalized_models);

%%  Create ecoinvent matrices and manipulate model matrices
[A_eco_mean, B_eco_mean, F_eco_mean,eco_process_meta] = include_EcoinventCoreLayer(Model,EcoinventModelCoreLayer,regions);
[Model_ecoinvent] = manipulate_matrixes(Model, B_eco_mean, A_eco_mean, F_eco_mean, eco_process_meta);

end