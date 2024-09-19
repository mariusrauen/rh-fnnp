%% Working Instruction:
%  !!!!!! This code is not meant to run separately but belongs to the script
%  "main_run_all.m" !!!!!!
%  
%  1. Please specify the folder name of your Batch below
%
%  2. Please specify your layer (core layer or extension layer) below
%
%  3. Make sure that you have prepared all data in your Batch folder.
%    3a. You need to include the IHS files into the 01_IHS_chosen
%         folder of your Batch.
%    3b. You need to make sure that the flow_matching_extra.xlsx is filled
%         out correctly.
%    3c. In case you have flow splitting, discuss in the team
%
%  For more information, please also check the Working Instruction in
%  
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

% specify ecoinvent version here (ALWAYS ecoinventVersion = "3.8_2021")
ecoinventVersion = "3.8_2021";

% folder paths
workingPath = pwd;

output_dir = fullfile(workingPath,'..',layerName,"01_finalizedBatches",folderNameBatch,"\");
input_dir = fullfile(output_dir,"01_IHS_chosen","\");

%%
path_alignments = fullfile(workingPath,'..',"01_input","wasteCode\");
PathEcoinvent = fullfile(workingPath,'..','..',"01_ecoinvent","01_preprocessing","Output",ecoinventVersion,"cut-off\");
PathFunctions = fullfile(workingPath,"functions\");
addpath(genpath(PathFunctions));
addpath(genpath(path_alignments));

%% find all files in input directory

file_list = dir(input_dir); % delete void entries
file_list = file_list(~cell2mat({file_list.isdir}));

%% remove hidden fields 

index = [];
for i = 1:length(file_list)
    if ~strcmp(file_list(i).name(1),'.')
        index(end+1) = i;
    end
end

file_list = file_list(index);

%% main loop to include excel files

disp('reading IHS files...')

for i = 1:length(file_list)
    % read file 
    [~,~,file] = xlsread(strcat(input_dir,string(file_list(i).name)));
    
    % get process
    process_list(i).streams = get_streams(file);
    process_list(i).costs = get_costs(file);
    process_list(i).info = get_info(file);   
end 

%% write results to COMPASS input

disp('processing files...')
Model = write2model(process_list);

%% Add Q and empty B matrix 

Model = add_Q_B(Model,PathEcoinvent);

%% set up final model

disp('Model is cleaned from crazy symbols...')
[Model] = clean_up_TM_symbols(Model);

Model.meta_data_processes{10,1} = 'type';
Model.meta_data_processes(10,2:end) = num2cell(1);

%% save model

disp('Initial Model is saved ...');
delete(strcat(output_dir,"Model.mat"));

save(strcat(output_dir,"Model.mat"),'Model');

%% clear paths
rmpath(genpath(PathFunctions));
rmpath(genpath(path_alignments));

disp('DONE')