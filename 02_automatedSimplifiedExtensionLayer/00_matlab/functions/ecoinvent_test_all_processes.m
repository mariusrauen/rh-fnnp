% main script to include LCI datasets in the country specific LCI models
% from technological data.

%%%%% Only works of ECO-List.xlsx is given %%%%%

% Contact RMe (Raoul.Meys@ltt.rwth-aachen.de)

%% clear workspace and output data
clc

clear

% DO NOT CHANGE!
addpath(genpath('\\das\est-compass-db\02_ecoinvent_processing_matlab'));
% DO NOT CHANGE!

%%%%%%%%%%%%%%%%%%%%%%%% Change inputs from here %%%%%%%%%%%%%%%%%%%%%%%%%%

%% 1. Give path to LCI model, Eco_list and output folder 

path_model = 'E:\COMPASS Database Build-Up\ZZ_final_database_scripts\07_test_models';
name_model = 'Database_V0.mat';

path_Eco_list='E:\COMPASS Database Build-Up\ZZ_final_database_scripts\07_test_models';

path_output_folder = 'E:\COMPASS Database Build-Up\ZZ_final_database_scripts\07_test_models';

%% 2. give favored regions for ecoinvent

region{1}='GLO';
region{2}='RER';
region{3}='RoW';

%% 3. specify which allocation type you prefer

% 1 = allocation cut-off ; 2 = allocation at point of substitution ; 3 = allocation consequential

allocation_ecoinvent=1;

%% 4. Choose outputs (1=generate output)

% a) model_Ecoinvent: saves the final ecoinvent model in output folder
generate_ecoinvent_model=1;


%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% until here %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% NO CHANGES FROM HERE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% include Ecoinvent processes

%% load model

temp=load([path_model,'\',name_model]);
temp=struct2cell(temp(1));
Model=temp{1,1};

clear temp path_model name_model

%% include ecoinvent data

tic
display('ecoinvent in progress');
[Model_Ecoinvent]=make_ecoinvent(Model, path_Eco_list,region,allocation_ecoinvent);
toc

%% gemerate results

% generate models
if generate_ecoinvent_model
save([path_output_folder,'\','model_Ecoinvent.mat'],'Model_Ecoinvent');
end

clear
