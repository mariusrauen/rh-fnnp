%% EXPLANATIONS

% Specify the name.
% Make sure on the left side you are seeing your chemicals folder contents.

% questions --> Raoul.Meys@carbon-minds.com


clear % do not change
clc % do not change


%% specify the name
name = 'Model';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% NO CHANGES FROM HERE

file = ['flows_to_revise_',name,'.xlsx']; 
delete(file);

file = ['flows_in_processes_',name,'.xlsx'];
delete(file);

%% NO MODIFICATIONS FROM HERE

% folder paths
output_dir = [pwd]; 

input_dir = [pwd,'\01_IHS_chosen\'];

path_alignments = 'G:\Geteilte Ablagen\03 Tools and Products\CM.CHEMICALS DATABASE\03_final_technology_datasets\00b_alignments';

addpath(genpath(['G:\Geteilte Ablagen\03 Tools and Products\CM.CHEMICALS DATABASE\03_final_technology_datasets\00a_matlab\functions\']));
addpath(genpath(path_alignments));


%% Load model

disp('Model is loaded ...');

load('Model.mat');


%% make flow matching

disp('Flow matching is performed...please wait.')

[ flow_matching ] = get_flow_matching(output_dir);
[ flow_splitting ] = get_flow_splitting(path_alignments); % 

[ Model ] = make_matching(Model, flow_matching, flow_splitting);

%% make cut-off

disp('Monetary flows are excluded ...');

[Model,Cutoff,processes] = make_cutoff(Model);

%% update flow meta data

disp('Meta data of flows is updated...')

% get meta data
[missing_meta_data] = get_missing_meta_data(path_alignments);
[Model,~,meta_data_to_update,flows_in_processes] = make_flow_revision(Model, missing_meta_data);


%% safe Excel table

disp('Excel table for internal energy recovery and flow revision is generated...')

% generate_flows_in_processes
warning('off', 'MATLAB:xlswrite:AddSheet');

xlswrite(['flows_in_processes_',name,'.xlsx'],...
    flows_in_processes.inputs,'inputs');
xlswrite(['flows_in_processes_',name,'.xlsx'],...
    flows_in_processes.outputs,'outputs');

xlswrite(['flows_to_revise_',name,'.xlsx'],meta_data_to_update);

%% save model

file = 'Model.mat';

delete(file);

disp('Model with revised flows is saved ...');

save([pwd,'\Model.mat'],'Model');

%% clear paths
rmpath(genpath(['G:\Geteilte Ablagen\03 Tools and Products\CM.CHEMICALS DATABASE\03_final_technology_datasets\00a_matlab\functions\']));
rmpath(genpath(path_alignments));

clear

disp('DONE')