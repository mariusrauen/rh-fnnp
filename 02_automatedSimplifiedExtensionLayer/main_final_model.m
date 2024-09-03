%% *SCRIPT TO BUILT THE TECHNOLOGY DATA INPUTS*
% Original Author: Raoul.Meys@carbon-minds.com
% Edits and Responsibility: Laura.Stellner@carbon-minds.com

% make sure you are in the right folder where also this code is stored 
  
clear
clc

workingPath = pwd;
addpath(genpath(pwd));

pathFunctions = fullfile(workingPath,'..','00_matlab','functions');
addpath(genpath(pathFunctions));

%% User interface to specify the required ecoinvent version and the ecoinvent system model (needed for the right elementary flows in the technology data)
promt1 = "Ecoinvent version required? (e.g. 3.9.1_2022)\n";
ecoinventVersion = input(promt1,'s');
promt2 = "Ecoinvent system model required? (e.g. apos or cut-off)\n";
ecoinventSystemModel = input(promt2,'s');

%% Add paths
pathInputPreprocessing = fullfile(workingPath,'..','..','01_ecoinvent','01_preprocessing','Output',ecoinventVersion,ecoinventSystemModel);

pathGlobalInput = fullfile(workingPath,'..','..','00_inputData');

pathInput = fullfile(workingPath,'..','01_input');
pathOutput = fullfile(workingPath,'00_TechModel',ecoinventVersion,ecoinventSystemModel);
mkdir(pathOutput)

pathRevisedModels = fullfile(workingPath,'02_revisedChemicals');
%pathRevisedModels_noAllocation = fullfile(workingPath,'02b_revisedChemicals_noAllocation');

pathInputWaste = fullfile(workingPath,'..','01_input','wasteCode');

pathEcoMatchingTable = fullfile(workingPath,'..','..','01_ecoinvent','03_modelExtensionLayer','01_input');
matchingTableName = ['EcoMatch_',ecoinventVersion,'_',ecoinventSystemModel,'.xlsx'];
correspondanceFile = 'CorrespondanceEF.xlsx';

ModelName = 'Layer3Model';

PathLayer23 = fullfile(workingPath,'..','IncludedChemicals.xlsx');

PathInputsModel = fullfile(workingPath,'..','03_extensionLayer','00_TechModel',ecoinventVersion,ecoinventSystemModel,'Layer2Model_Layer3Testing.mat');

%% find all files in input directory

disp('Setting up model structure...');

file_list = dir(fullfile(pathInput,'dummy_IHS\')); % delete void entries

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

for i = 1:length(file_list)
    % read file 
    [~,~,file] = xlsread(fullfile(pathInput,'dummy_IHS\',file_list(i).name));
    
    % get process
    process_list(i).streams = get_streams(file);
    process_list(i).costs = get_costs(file);
    process_list(i).info = get_info(file);   
end 

%% write results to COMPASS input

Model = write2model(process_list);

%% Add Q and empty B matrix 

Model = add_Q_B(Model,pathInputPreprocessing);

%% set up final model

[Model] = clean_up_TM_symbols(Model);
Model.meta_data_processes{10,1} = 'allocation type';
% Include finished datasets

%%
disp('Revised datasets are included...')
[ add_manual_processes,adding_files  ] = get_process_adding(pathRevisedModels);

deleted_EF = [];
matched_EF = [];
for i = 1:length(add_manual_processes)
    disp(adding_files(i+2));
    [ Model,elementary_flows_to_delete, elementary_flow_matching] = make_process_adding_V2(Model, add_manual_processes(i), correspondanceFile, pathGlobalInput , ecoinventVersion);
    deleted_EF = [deleted_EF;elementary_flows_to_delete];
    matched_EF = [matched_EF;elementary_flow_matching];
end
	% create unique values
    [~,idx]=unique(cell2mat(deleted_EF),'rows');
    deleted_EF = deleted_EF(idx,:);
	% create unique values
    matched_EF = cell2table(matched_EF);
    if ~isempty(matched_EF)
        matched_EF.Properties.VariableNames = {'name_new','compartment_new','subcompartment_new','name_old','compartment_old','subcompartment_old'};
        matched_EF = unique(matched_EF);
    end
    
%%    
% Exclude IHS Dummy

Model.matrices.A.mean_values(:,1) = [];
Model.matrices.A.mean_values(1:4,:) = [];
Model.matrices.F.mean_values(:,1) = [];
Model.matrices.B.mean_values(:,1) = [];

Model.meta_data_processes(:,2) = [];

Model.meta_data_flows(2:5,:) = [];

disp('Excel files included')

%% Include waste flows

% get meta data
[missing_meta_data] = get_missing_meta_data(pathGlobalInput);
[Model,~,meta_data_to_update,flows_in_processes] = make_flow_revision(Model, missing_meta_data);
disp('include waste flows');

[ Model ] = make_waste_and_elementary_flows(Model,pathInputWaste);

% get meta data
[missing_meta_data] = get_missing_meta_data(pathGlobalInput);
[Model,~,meta_data_to_update,flows_in_processes] = make_flow_revision(Model, missing_meta_data);

%% include process descriptions

disp('Process description is generated automatically.');
[Model] = generic_process_decription_Layer3(Model);

%% load finalized system expansion

disp('Matrices are allocated ...');

A = Model.matrices.A.mean_values;
B = Model.matrices.B.mean_values;
F = Model.matrices.F.mean_values;

meta_data_processes = Model.meta_data_processes;

ProcessSize = size(meta_data_processes,1);

meta_data_flows = Model.meta_data_flows(:,[1,2,6]);

meta_data_elementary_flows = Model.meta_data_elementary_flows;

meta_data_monetary_flows = Model.meta_data_factor_requirements;

FlowSize = size(meta_data_monetary_flows,2);
meta_data_monetary_flows = reshape(meta_data_monetary_flows,[],FlowSize);

% set positive values for allocation calculation of utilities to zero

UtilityFlows = or(cell2mat(meta_data_flows(2:end,2)) == 2,string(meta_data_flows(2:end,1)) == "carbon dioxide");

Utilities = A(UtilityFlows,:);

MetaUtilities = meta_data_flows(find([0;UtilityFlows]),:);

meta_data_flows(find([0;UtilityFlows]),:) = [];

% Include meta data of flows

disp('Meta data of flows is updated...')

%% Before performing allocation, Test if all flows used in energy allocation have a LHV >0
[flows_missingLHV_energy,problematic_processes_energy,flows_missingLHV_price,problematic_processes_price] =  perform_check_before_allocation(Model);

%% save the systemExpansion before the allocation
writecell(Model.meta_data_processes,...
    fullfile(pathOutput,'SystemExpansion_beforeAllocation.xlsx'),'Sheet','Process_meta_data');
writecell([[Model.meta_data_flows(1,:),Model.meta_data_processes(1,2:end)];[Model.meta_data_flows(2:end,:),num2cell(Model.matrices.A.mean_values)]],...
    fullfile(pathOutput,'SystemExpansion_beforeAllocation.xlsx'),'Sheet','SUMMARY A');
writecell([[Model.meta_data_elementary_flows(1,:),Model.meta_data_processes(1,2:end)];[Model.meta_data_elementary_flows(2:end,:),num2cell(Model.matrices.B.mean_values)]],...
    fullfile(pathOutput,'SystemExpansion_beforeAllocation.xlsx'),'Sheet','SUMMARY B');
writecell([[Model.meta_data_factor_requirements(1,:),Model.meta_data_processes(1,2:end)];[Model.meta_data_factor_requirements(2:end,:),num2cell(Model.matrices.F.mean_values)]],...
    fullfile(pathOutput,'SystemExpansion_beforeAllocation.xlsx'),'Sheet','SUMMARY F');

%% Allocation
A_alloc_mass = A;
A_alloc_mass(A_alloc_mass<0) = 0;
[ A_alloc_energy , ~ ] = get_energy_matrix_A(Model);
[ Model,A_alloc_price , ~ ] = get_price_matrix_A(Model);

allocationType = nan(size(Model.meta_data_processes(10,:)));
idx_strings = cellfun(@ischar,Model.meta_data_processes(10,:));
allocationType(~idx_strings) = cell2mat(Model.meta_data_processes(10,~idx_strings));
allocationType(idx_strings) = 0;
alloc_procedure = allocationType(2:end);

if ~isequal(size(alloc_procedure,2),size(A,2))
    
    disp('PROBLEM WITH MULTIFUNCTIONALITY --> SCRIPT STOPPED. DISCUSS IN GROUP');
    return
    
end

%% Add the allocation types for the allocation descriptions that are strings (e.g. PCR)
% add the string descriptions of allogaion for PCRs from meta_data_processes
nan_idx = find(~cellfun(@isnumeric, meta_data_processes(10,:)))';
nan_idx = nan_idx(2:end);
allocationTypesNew = table(Model.meta_data_processes(10,:)');
allocationTypesNew.Var1(nan_idx) = meta_data_processes(10,nan_idx)';
% Replace all 0, 1, 2, 3 values with the replacement string
for i=1:height(allocationTypesNew)
    if isequal(allocationTypesNew.Var1(i),{0})
        allocationTypesNew.Var1{i} = 'no allocation needed in this unit process';
    elseif isequal(allocationTypesNew.Var1(i),{1})
        allocationTypesNew.Var1{i} = 'allocation via lower heating value';
    elseif isequal(allocationTypesNew.Var1(i),{2})
        allocationTypesNew.Var1{i} = 'allocation via mass';
    elseif isequal(allocationTypesNew.Var1(i),{3})
        allocationTypesNew.Var1{i} = 'allocation via price';
    end
end
allocationTypesNew.Var1 = string(allocationTypesNew.Var1);
allocationTypesNew.Var2 = allocationTypesNew.Var1 + repmat(". In case of energy co-production, allocation via avoided burden is applied for the co-produced energy.",height(allocationTypesNew),1);
allocationTypesNew.Var2(1) = allocationTypesNew.Var1(1);
Model.meta_data_processes(10,:) = cellstr(allocationTypesNew.Var2');

energy = alloc_procedure==1;
mass = alloc_procedure==2;
prices = alloc_procedure==3;

Amatrixforallocationfactors = zeros(size(A));

Amatrixforallocationfactors(:,energy) = A_alloc_energy(:,energy);
Amatrixforallocationfactors(:,mass) = A_alloc_mass(:,mass);
Amatrixforallocationfactors(:,prices) = A_alloc_price(:,prices);

%% ALLOCATION
% Amatrixforallocationfactors = A_alloc_mass;

A(UtilityFlows,:) = [];
Amatrixforallocationfactors(UtilityFlows,:) = [];

output = perform_allocation(...
    A,...
    B,...
    F,...
    Utilities,...
    Amatrixforallocationfactors,...
    Model.meta_data_processes,...
    meta_data_flows);

if any(sum(output.A>0)>1)
    disp('PROBLEM WITH MULTIFUNCTIONALITY --> SCRIPT STOPPED. DISCUSS IN GROUP');
    return
end

scaling_A = max(output.A);

A = [output.A./scaling_A;output.Utilities./scaling_A];
F = output.F./scaling_A;
B = output.B./scaling_A;

MetaDataProcesses = output.meta_data_processes;
MetaDataTechnicalFlows = [meta_data_flows;MetaUtilities];
MetaDataMonetaryFlows = meta_data_monetary_flows;
MetaDataElementaryFlows = meta_data_elementary_flows;

Model.matrices.A.mean_values = A;
Model.matrices.B.mean_values = B;
Model.matrices.F.mean_values = F;

Model.meta_data_processes = MetaDataProcesses;
Model.meta_data_factor_requirements = MetaDataMonetaryFlows;
Model.meta_data_elementary_flows = MetaDataElementaryFlows;
Model.meta_data_flows(:,[1,2,6]) = MetaDataTechnicalFlows;

% get meta data
[missing_meta_data] = get_missing_meta_data(pathGlobalInput);
[Model,~,meta_data_to_update,flows_in_processes] = make_flow_revision(Model, missing_meta_data);

%% Exclude unwanted by-products

opts = spreadsheetImportOptions("NumVariables", 2);

% Specify sheet and range
opts.Sheet = "LAYER3";

% Specify column names and types
opts.VariableNames = ["IncludedProcesses", "IncludedChemical"];
opts.VariableTypes = ["string", "string"];

% Specify variable properties
opts = setvaropts(opts, ["IncludedProcesses", "IncludedChemical"], "WhitespaceRule", "preserve");
opts = setvaropts(opts, ["IncludedProcesses", "IncludedChemical"], "EmptyFieldRule", "auto");

IncludedChemicalsTable = readtable(PathLayer23, opts, "UseExcel", false);

Layer2Chemicals = IncludedChemicalsTable.IncludedChemical(2:end);
Layer2Processes = IncludedChemicalsTable.IncludedProcesses(2:end);

ProcessesMainFlow = Model.meta_data_processes(4,2:end);
ProcessesNames = Model.meta_data_processes(1,2:end);

IncludedProcesses = zeros(length(Layer2Chemicals),1);

missing_Processes = [];
missing_MainFlows = [];
for i = 1:length(Layer2Chemicals)
    try
        IncludedProcesses(i,1) = find(strcmp(ProcessesMainFlow,Layer2Chemicals(i))&...
            strcmp(ProcessesNames,Layer2Processes(i)));
    catch
        missing_Processes = [missing_Processes;Layer2Processes(i)];
        missing_MainFlows = [missing_MainFlows;Layer2Chemicals(i)];
    end
end

if ~isempty(missing_Processes)
    for i=1:size(missing_Processes,1)
        disp(["ERROR: The process "+missing_Processes(i,1)+" with mainFlow "+missing_MainFlows(i,1)+" is in the IncludedChemicals table but is not included in the Model!"]);
    end
    error("ERROR: Please correct error in IncludedChemicals table for Layer2 or Model and run the code again!")
end

IncludedProcesses = sort(IncludedProcesses);
ExcludedProcesses = 1:size(Model.matrices.A.mean_values,2);
ExcludedProcesses(IncludedProcesses) = [];
ExcludedProcesses = sort(ExcludedProcesses);

ProcessMeta_excluded = Model.meta_data_processes;
ProcessMeta_excluded = ProcessMeta_excluded(:,[1, ExcludedProcesses+1]);

ProcessMeta_excluded = ProcessMeta_excluded([1,4],2:end)';

% Revise model
% A matrix and meta data
A = Model.matrices.A.mean_values(:,IncludedProcesses);
TechFlowMeta = Model.meta_data_flows;
TechFlows2Delete = any(A');
A(~TechFlows2Delete,:) = [];
TechFlowMeta = TechFlowMeta([1,find(TechFlows2Delete)+1],:);

ProcessMeta = Model.meta_data_processes;
ProcessMeta = ProcessMeta(:,[1; IncludedProcesses+1]);

% Other matrices
B = Model.matrices.B.mean_values(:,IncludedProcesses);
F = Model.matrices.F.mean_values(:,IncludedProcesses);

% rewrite model
Model.matrices.A.mean_values = A;
Model.matrices.B.mean_values = B;
Model.matrices.F.mean_values = F;
Model.meta_data_processes = ProcessMeta;
Model.meta_data_flows = TechFlowMeta;

%% Testing

% Load Layer1 including ecoinvent and convert Layer 2 to process adding
load(PathInputsModel);

[AddLayer2InclEcoinvent] = TransferModelToProcessAdding(Model);

% combine models
[ Model_Layer1_and_2_and_3 ] = make_process_adding_V2_Layer23(Model_Ecoinvent_Layer1_and_2, AddLayer2InclEcoinvent);

% get meta data
[missing_meta_data] = get_missing_meta_data(pathGlobalInput);
[Model_Layer1_and_2_and_3,~,meta_data_to_update,flows_in_processes] = make_flow_revision(Model_Layer1_and_2_and_3, missing_meta_data);

disp('Ecoinvent inclusion.')

regions = {'GLO', 'RER', 'RoW','DE'}';

allocation_ecoinvent = 1;

[ Model_Ecoinvent_Layer1_and_2_and_3 ] = ...
    make_ecoinvent_one_addition(Model_Layer1_and_2_and_3,...
    pathInputPreprocessing,...
    pathEcoMatchingTable,...
    regions,...
    allocation_ecoinvent,...
    matchingTableName);

% Check Model

[process_errors,impacts_processes,s,y] = ...
    check_model_Ecoinvent(Model_Ecoinvent_Layer1_and_2_and_3);

%% check model for main flows
main_flow_missing = 0;
main_flow_value_zero = 0;
main_flows = Model.meta_data_processes(4,2:end);

for i = 1:size(main_flows,2)
    
    column = i;
    process = i+1;
    
    positive_values = Model.matrices.A.mean_values(:,i) > 0;
    
    positive_flows = Model.meta_data_flows(find(positive_values)+1,1);
    
    check_mainflow = any(contains(positive_flows,main_flows(i)));
    
    value_mainflow = Model.matrices.A.mean_values(positive_values,i);
    
    if ~check_mainflow
        disp(Model.meta_data_processes(1,process));
        main_flow_missing = 1;
    end
    
    if any(value_mainflow<=0)
        disp(Model.meta_data_processes(1,process),' with flow ',Model.meta_data_processes(4,process));
        main_flow_value_zero = 1;
    end
    
end

if isempty(find(any(y<-1e-10)))
    disp('y looks fine');
end

if isempty(process_errors{1,1})
    disp('processes look fine');
end

if main_flow_missing
    disp('some main flows are missing');
else
    disp('main flows look fine');
end

if main_flow_value_zero
    disp('some main flows are equal/below zero');
else
    disp('main flow values look fine');
end

%% SAVE RESULTS AND CLEAR MATLAB

save(fullfile(pathOutput,[ModelName,'.mat']),'Model');
save(fullfile(pathOutput,[ModelName,'_CustomTesting.mat']),'Model_Ecoinvent_Layer1_and_2');
%%
writecell(ProcessMeta',...
    fullfile(pathOutput,'ProcessMeta.xlsx'));

writecell(Model.meta_data_processes,...
    fullfile(pathOutput,'SystemExpansion.xlsx'),'Sheet','Process_meta_data');
writecell([[Model.meta_data_flows(1,:),Model.meta_data_processes(1,2:end)];[Model.meta_data_flows(2:end,:),num2cell(Model.matrices.A.mean_values)]],...
    fullfile(pathOutput,'SystemExpansion.xlsx'),'Sheet','SUMMARY A');
writecell([[Model.meta_data_elementary_flows(1,:),Model.meta_data_processes(1,2:end)];[Model.meta_data_elementary_flows(2:end,:),num2cell(Model.matrices.B.mean_values)]],...
    fullfile(pathOutput,'SystemExpansion.xlsx'),'Sheet','SUMMARY B');
writecell([[Model.meta_data_factor_requirements(1,:),Model.meta_data_processes(1,2:end)];[Model.meta_data_factor_requirements(2:end,:),num2cell(Model.matrices.F.mean_values)]],...
    fullfile(pathOutput,'SystemExpansion.xlsx'),'Sheet','SUMMARY F');

rmpath(genpath(pathFunctions));

if or(~isempty(flows_missingLHV_energy),~isempty(problematic_processes_energy))
    
    writetable(problematic_processes_energy,fullfile(pathOutput,'problematic_processes_and_flows_energy.xlsx'),...
    'Sheet','problematic Processes');
    writetable(flows_missingLHV_energy,fullfile(pathOutput,'problematic_processes_and_flows_energy.xlsx'),...
    'Sheet','problematic Flows missing LHV');

    disp('!!!WARNING!!!: Allocation by energy does not perform correctly!')
    disp('There are flows that are missing the LHV value but which occur as outputs in a process that is allocated by energy!')
    disp('Please check flows_missingLHV and problematic processes to see which flows and processes cause this issue.')
    disp('There are 2 things you can do:')
    disp('1. Add the LHV value in G:\Geteilte Ablagen\09a_NEW_cm_chemicals_data\01_final_technology_datasets\00b_alignments\meta_data_flows.xlsx')
    disp('2. Change the allocation type in the corresponding System Expansion.xlsx (Sheet:Process_meta_data, row:type) in REVISED_CHEMICALS from 1(=energy allocation) to 2(=mass allocation)')
end
if or(~isempty(flows_missingLHV_price),~isempty(problematic_processes_price))
    
    writetable(problematic_processes_price,fullfile(pathOutput,'problematic_processes_and_flows_price.xlsx'),...
    'Sheet','problematic Processes');
    writetable(flows_missingLHV_price,fullfile(pathOutput,'problematic_processes_and_flows_price.xlsx'),...
    'Sheet','problematic Flows missing LHV');

    disp('!!!WARNING!!!: Allocation by price does not perform correctly!')
    disp('There are flows that are missing the price value but which occur as outputs in a process that may be allocated by price!')
    disp('Please check flows_missingprice and problematic processes to see which flows and processes cause this issue.')
    disp('There are 2 things you can do:')
    disp('1. Add the LHV value in G:\Geteilte Ablagen\09a_NEW_cm_chemicals_data\01_final_technology_datasets\00b_alignments\meta_data_flows.xlsx')
    disp('2. Change the allocation type in the corresponding System Expansion.xlsx (Sheet:Process_meta_data, row:type) in REVISED_CHEMICALS from 1(=energy allocation) to 2(=mass allocation)')
end

disp('DONE')
