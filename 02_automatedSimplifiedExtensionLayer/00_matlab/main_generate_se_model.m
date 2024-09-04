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
file = fullfile(output_dir,"SystemExpansion.xlsx");
if exist(file) == 2
    delete(file);
end

%% Load model

disp('Model with revised flows is loaded ...');

load(fullfile(output_dir,'Model.mat'));

%% get elementary flows

disp('Elementary flows for each process are calculated ...');

[ Model ] = make_waste_and_elementary_flows(Model,path_alignments);

% revise elementary flows

B = Model.matrices.B.mean_values;

if size(B,2) == 1
    
    ElementaryFlowsIncluded = B>0;
    
else
    
    ElementaryFlowsIncluded = any(B')';
    
end

B = B(ElementaryFlowsIncluded,:);

Model.matrices.B.mean_values = B;

MetaDataElementaryFlows = Model.meta_data_elementary_flows;
MetaDataElementaryFlows(find(~ElementaryFlowsIncluded)+1,:) = [];

Model.meta_data_elementary_flows = MetaDataElementaryFlows;

%% generate process model for system expansion
MetaDataProcesses = Model.meta_data_processes;

Model.meta_data_processes{10,1} = 'Allocation procedure';

A = Model.matrices.A.mean_values;
MetaDataTechnicalFlows = Model.meta_data_flows(:,[1,2,6]);

F = Model.matrices.F.mean_values;
MetaDataMonetaryFlows = Model.meta_data_factor_requirements(:,1:2);

%% Save System Expansion
writecell(MetaDataProcesses,...
    fullfile(output_dir,"SystemExpansion.xlsx"),...
    'Sheet','Process_meta_data','WriteMode','overwritesheet');

writecell([[MetaDataTechnicalFlows(1,:),MetaDataProcesses(1,2:end)];[MetaDataTechnicalFlows(2:end,:),num2cell(A)]],...
    fullfile(output_dir,"SystemExpansion.xlsx"),...
    'Sheet','SUMMARY A','WriteMode','overwritesheet');

writecell([[MetaDataElementaryFlows(1,:),MetaDataProcesses(1,2:end)];[MetaDataElementaryFlows(2:end,:),num2cell(B)]],...
    fullfile(output_dir,"SystemExpansion.xlsx"),...
    'Sheet','SUMMARY B','WriteMode','overwritesheet');

writecell([[MetaDataMonetaryFlows(1,:),MetaDataProcesses(1,2:end)];[MetaDataMonetaryFlows(2:end,:),num2cell(F)]],...
    fullfile(output_dir,"SystemExpansion.xlsx"),...
    'Sheet','SUMMARY F','WriteMode','overwritesheet');

%% save model
delete(strcat(output_dir,"Model.mat"));
disp('Model with revised flows is saved ...');
save(strcat(output_dir,"Model.mat"),'Model');

%% clear paths
rmpath(genpath(PathFunctions));
rmpath(genpath(path_alignments));

disp('DONE')