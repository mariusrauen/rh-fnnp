%% EXPLANATIONS

% Nothing to specify here.
% Make sure on the left side you are seeing your chemicals folder contents.

% questions --> Raoul.Meys@carbon-minds.com

clear % do not change
clc % do not change

name_final__se_table = 'ETHYLENE_AND_PROPYLENE.xlsx';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% NO CHANGES FROM HERE

file = ['Allocation.xlsx']; 
delete(file);

% folder paths
output_dir = [pwd]; 

input_dir = [pwd,'\01_IHS_chosen\'];

path_alignments = 'G:\Geteilte Ablagen\03 Tools and Products\CM.CHEMICALS DATABASE\03_final_technology_datasets\00b_alignments\';

addpath(genpath(['G:\Geteilte Ablagen\03 Tools and Products\CM.CHEMICALS DATABASE\03_final_technology_datasets\00a_matlab\functions\']));
addpath(genpath(path_alignments));

%% load finalized system expansion

disp('Matrices are allocated according to mass...');
A = xlsread(name_final__se_table,'SUMMARY A','D2:Z100');
B = xlsread(name_final__se_table,'SUMMARY B','E2:Z100');
F = xlsread(name_final__se_table,'SUMMARY F','C2:Z100');

[~,~,meta_data_processes] = xlsread(name_final__se_table,'Process_meta_data');

ProcessSize = size(meta_data_processes,1);

[~,~,meta_data_flows] = xlsread(name_final__se_table,'SUMMARY A','A1:C100');
FlowSize = size(meta_data_flows,2);
meta_data_flows(cellfun(@(meta_data_flows) any(isnan(meta_data_flows)),meta_data_flows)) = [];
meta_data_flows = reshape(meta_data_flows,[],FlowSize);

[~,~,meta_data_elementary_flows] = xlsread(name_final__se_table,'SUMMARY B','A1:D100');
FlowSize = size(meta_data_elementary_flows,2);
meta_data_elementary_flows(cellfun(@(meta_data_elementary_flows) any(isnan(meta_data_elementary_flows)),meta_data_elementary_flows)) = [];
meta_data_elementary_flows = reshape(meta_data_elementary_flows,[],FlowSize);

[~,~,meta_data_monetary_flows] = xlsread(name_final__se_table,'SUMMARY F','A1:B100');
FlowSize = size(meta_data_monetary_flows,2);
meta_data_monetary_flows(cellfun(@(meta_data_monetary_flows) any(isnan(meta_data_monetary_flows)),meta_data_monetary_flows)) = [];
meta_data_monetary_flows = reshape(meta_data_monetary_flows,[],FlowSize);

meta_data_processes(cellfun(@(meta_data_processes) any(isnan(meta_data_processes)),meta_data_processes)) = {''};

% set positive values for allocation calculation of utilities to zero

UtilityFlows = cell2mat(meta_data_flows(2:end,2)) == 2;

Utilities = A(UtilityFlows,:);

MetaUtilities = meta_data_flows(find([0;UtilityFlows]),:);

meta_data_flows(find([0;UtilityFlows]),:) = [];

A(UtilityFlows,:) = [];

Amatrixforallocationfactors = A;

output = perform_allocation(...
    A,...
    B,...
    F,...
    Utilities,...
    Amatrixforallocationfactors,...
    meta_data_processes,...
    meta_data_flows);

%% generate process model for system expansion
disp('Matrices are stored ...');

A = [output.A;output.Utilities];
F = output.F;
B = output.B;

MetaDataProcesses = output.meta_data_processes;
MetaDataTechnicalFlows = [meta_data_flows;MetaUtilities];
MetaDataMonetaryFlows = meta_data_monetary_flows;
MetaDataElementaryFlows = meta_data_elementary_flows;

warning('off', 'MATLAB:xlswrite:AddSheet');

xlswrite(['Allocation.xlsx'],...
    MetaDataProcesses,'Process_meta_data');

xlswrite(['Allocation.xlsx'],...
    [[MetaDataTechnicalFlows(1,:),MetaDataProcesses(1,2:end)];[MetaDataTechnicalFlows(2:end,:),num2cell(A)]],'SUMMARY A');

xlswrite(['Allocation.xlsx'],...
    [[MetaDataElementaryFlows(1,:),MetaDataProcesses(1,2:end)];[MetaDataElementaryFlows(2:end,:),num2cell(B)]],'SUMMARY B');

xlswrite(['Allocation.xlsx'],...
    [[MetaDataMonetaryFlows(1,:),MetaDataProcesses(1,2:end)];[MetaDataMonetaryFlows(2:end,:),num2cell(F)]],'SUMMARY F');

%% clear paths
rmpath(genpath(['G:\Geteilte Ablagen\03 Tools and Products\CM.CHEMICALS DATABASE\03_final_technology_datasets\00a_matlab\functions\']));
rmpath(genpath(path_alignments));

clear

disp('DONE')