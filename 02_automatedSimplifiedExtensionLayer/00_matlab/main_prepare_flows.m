%% Working Instruction:
%  !!!!!! This code is not meant to run separately but belongs to the script
%  "main_run_all.m" !!!!!!
%  
%  1. Please specify the folder name of your Batch below
%
%  2. Please specify your layer (core layer or extension layer) below
%
%  For any questions, please contact Laura Stellner, Aline Kalousdian, or
%  other members from the Database Team.

%%
clear % do not change
clc % do not change

%% Specify folder name of batch and layer here!
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%% Specify folder name of batch and layer here! %%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% specify whether you are in the core or extension layer
% either: layerName = "02_coreLayer";
% or    : layerName = "03_extensionLayer";
layerName = "03_extensionLayer";

% specify folder name of Batch (e.g. folderNameBatch = "BATCH_30_Updated_White_Oil")
folderNameBatch = "BATCH_30_Updated_White_Oil";

%% NO MODIFICATIONS FROM HERE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%% Do not change anything from here onwards %%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% folder paths
workingPath = pwd;

output_dir = fullfile(workingPath,'..',layerName,"01_finalizedBatches",folderNameBatch,"\");

path_globalInput = fullfile(workingPath,'..','..',"00_inputData\");

path_alignments = fullfile(workingPath,'..',"01_input","wasteCode\");
PathFunctions = fullfile(workingPath,"functions\");
addpath(genpath(PathFunctions));
addpath(genpath(path_alignments));

%%
file = fullfile(output_dir,strcat("flows_to_revise_",folderNameBatch,".xlsx"));
if exist(file) == 2
    delete(file);
end

file = fullfile(output_dir,strcat("flows_in_processes_",folderNameBatch,".xlsx")); 
if exist(file) == 2
    delete(file);
end

%% Load model

disp('Model is loaded ...');
load(fullfile(output_dir,'Model.mat'));


%% make flow matching

disp('Flow matching is performed...please wait.')

[ flow_matching ] = get_flow_matching(output_dir);
[ flow_splitting ] = get_flow_splitting(output_dir); % 

[ Model ] = make_matching(Model, flow_matching, flow_splitting);

%% make cut-off

disp('Monetary flows are excluded ...');

[Model,Cutoff,processes] = make_cutoff(Model);

%% update flow meta data

disp('Meta data of flows is updated...')

% get meta data
[missing_meta_data] = get_missing_meta_data(path_globalInput);
[Model,~,meta_data_to_update,flows_in_processes] = make_flow_revision(Model, missing_meta_data);


%% safe Excel table

disp('Excel table for internal energy recovery and flow revision is generated...')

% generate_flows_in_processes
% warning('off', 'MATLAB:xlswrite:AddSheet');

writecell(flows_in_processes.inputs,fullfile(output_dir,strcat("flows_in_processes_",folderNameBatch,".xlsx")),...
    'Sheet','inputs','WriteMode','overwritesheet');
writecell(flows_in_processes.outputs,fullfile(output_dir,strcat("flows_in_processes_",folderNameBatch,".xlsx")),...
    'Sheet','outputs','WriteMode','overwritesheet');

writecell(meta_data_to_update,fullfile(output_dir,strcat("flows_to_revise_",folderNameBatch,".xlsx")),...
    'Sheet','Sheet1','WriteMode','overwritesheet');

%% save model
delete(strcat(output_dir,"Model.mat"));
disp('Model with revised flows is saved ...');
save(strcat(output_dir,"Model.mat"),'Model');

%% clear paths
rmpath(genpath(PathFunctions));
rmpath(genpath(path_alignments));

disp('DONE')