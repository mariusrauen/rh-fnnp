function [ Model_layer_1_2_3, missing_elementary_flows ] = include_layer_3_data(Model, Amatrix, Bmatrix, database_scope, path_alignments)

%% add paths
% current_path = mfilename('fullpath');
% filename = mfilename;
% 
% current_path = current_path(1:end-length(filename));
% 
% if strcmp(filename,'LiveEditorEvaluationHelperESectionEval')
%     current_path = [pwd,'\'];
% end
% 
% path_inputs = [current_path];

% for debugging only:
% current_path = [pwd,'\01_make_regional_technology_datasets\'];
path_inputs = [path_alignments,'layer_3\'];

%% reduce layer 3 to scope
[Amatrix, Bmatrix] = ...
    reduce_layer_3_to_scope(Amatrix, Bmatrix, database_scope);
if isempty(Amatrix) & isempty(Bmatrix)
    disp('No layer 3 included')
    Model_layer_1_2_3 = Model;
    missing_elementary_flows = 'no layer 3';
    return
end

%% combine layer 1, 2 and 3
[Model_layer_1_2_3, missing_elementary_flows] = combine_layer_3(Amatrix, Bmatrix, Model,path_inputs);


