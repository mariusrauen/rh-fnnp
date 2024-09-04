function [Model,Cutoff] = perform_cutoff_V2(Model,file,cutoff_rule_cumulated,cutoff_rule)

Cutoff.row_names = Model.meta_data_flows(2:end,1);
Cutoff.col_names = Model.meta_data_processes(1,2:end);

% get data from 'file'

Cutoff.exclude_processes.process_names = file(2:end,1);
Cutoff.exclude_processes.factor = cell2mat(file(2:end,2));

% Calculate columns of processes to delete (Cutoff.exclude_processes.cols_to_delete)
Cutoff.exclude_processes.process_names_to_delete = Cutoff.exclude_processes.process_names;
Cutoff.exclude_processes.process_names_to_delete(Cutoff.exclude_processes.factor==1,:) = []; % new vector, contains all process names to delete (processes with factor 0)
Cutoff.exclude_processes.cols_to_delete = [];


all_columns = 1:size(Cutoff.exclude_processes.process_names,1);
% all_columns(Cutoff.exclude_processes.factor) = [];
Cutoff.exclude_processes.cols_to_delete = all_columns;

%% Create new struct Cutoff_final (contains modified 'Model'-struct and other solutions
Cutoff_final.matrices = Model.matrices; 

% Get Meta_data
Cutoff_final.meta_data_processes = Model.meta_data_processes;
Cutoff_final.meta_data_processes(:,Cutoff.exclude_processes.cols_to_delete+1)=[];
Cutoff_final.meta_data_elementary_flows = Model.meta_data_elementary_flows;

% Modify Matrix A
Cutoff_final.matrices.A.mean_values(:,Cutoff.exclude_processes.cols_to_delete) = [];

% Modify Matrix B
Cutoff_final.matrices.B.mean_values(:,Cutoff.exclude_processes.cols_to_delete) = [];

% Modify Matrix F
Cutoff_final.matrices.F.mean_values(:,Cutoff.exclude_processes.cols_to_delete) = [];

%% Create new struct for cut off 

% Copy matrices and vectors of struct 'Model' and struct 'sheet'
Cutoff.A = Cutoff_final.matrices.A.mean_values;
Cutoff.unit = Model.meta_data_flows(2:end,6);
Cutoff.type = cell2mat(Model.meta_data_flows(2:end,2));


%% Seperate flow vectors for utilities and raw_materials

Cutoff.raw_materials.A           = Cutoff.A;
Cutoff.utilities.A               = Cutoff.A;
Cutoff.utilities.rows_other_type = [];

for i=1:size(Cutoff.A,1)
    if  Cutoff.type(i,1) == 2                               % if flow is utility
      Cutoff.raw_materials.A(i,:) = 0;                      % A matrix for raw materials in kg 
      Cutoff.utilities.rows_other_type(end+1,1)=i;          % row numbers of raw materials not in kg
    else % if flow is raw material 
      Cutoff.utilities.A(i,:) = 0;          % A matrix for raw materials not in kg
    end 
end

%% Seperate flows in A_raw_materials.A if unit is not kg 

% Create matrix A_kg and A_other_units for raw materials 
Cutoff.raw_materials.A_kg             = Cutoff.raw_materials.A;
Cutoff.raw_materials.A_other_units    = Cutoff.raw_materials.A;
Cutoff.raw_materials.rows_other_units = [];


for i=1:size(Cutoff.raw_materials.A,1)
    
    if  strcmp('kg',Cutoff.unit(i,1)) == 0 && Cutoff.type(i,1) == 1  % if unit of raw materials is not kg 
      
      Cutoff.raw_materials.A_kg(i,:) = 0;                   % A matrix for raw materials in kg 
      Cutoff.raw_materials.rows_other_units(end+1,1)=i;     % row numbers of raw materials not in kg
    
    else % if unit of raw material is kg 
      
      Cutoff.raw_materials.A_other_units(i,:) = 0;          % A matrix for raw materials not in kg
    
    end 
end


%% Seperate Output and Input flows

 % Inputs
Cutoff.raw_materials.A_kg_inputs = min(Cutoff.raw_materials.A_kg,0);  % only inputs (negative values in A)

% Outputs
Cutoff.raw_materials.A_kg_outputs = max(Cutoff.raw_materials.A_kg,0); % only outputs (positive values in A)

%% Calculate mass proportion of inputs and outputs

% Inputs
Cutoff.raw_materials.inputs_sum = sum(Cutoff.raw_materials.A_kg_inputs);                    % sum of inputs for each process
Cutoff.raw_materials.inputs_sum_matrix = repmat(Cutoff.raw_materials.inputs_sum,size(Cutoff.raw_materials.A_kg_inputs,1),1); % create sum matrix for calculation of mass proportion

% Calculate mass proportion (Inputs)
Cutoff.raw_materials.proportion_inputs=Cutoff.raw_materials.A_kg_inputs./Cutoff.raw_materials.inputs_sum_matrix;
Cutoff.raw_materials.proportion_inputs(isnan(Cutoff.raw_materials.proportion_inputs))=0;  % convert NANs (result of inputs not in kg or output flows) to 0

% Outputs
Cutoff.raw_materials.outputs_sum = sum(Cutoff.raw_materials.A_kg_outputs);                    % sum of outputs for each process
Cutoff.raw_materials.outputs_sum_matrix = repmat(Cutoff.raw_materials.outputs_sum,size(Cutoff.raw_materials.A_kg_outputs,1),1); % create sum matrix for calculation of mass proportion

% Calculate mass proportion (Outputs)
Cutoff.raw_materials.proportion_outputs=Cutoff.raw_materials.A_kg_outputs./Cutoff.raw_materials.outputs_sum_matrix;
Cutoff.raw_materials.proportion_outputs(isnan(Cutoff.raw_materials.proportion_outputs))=0;  % convert NANs (result of outputs not in kg or input flows) to 0

%% Calculate cumulated mass proportions for Inputs

% Sort and cumulate Inputs
[Cutoff.raw_materials.cumulated_inputs, Idx] = sort(Cutoff.raw_materials.proportion_inputs,1);      % sort
Cutoff.raw_materials.cumulated_inputs = cumsum(Cutoff.raw_materials.cumulated_inputs,1);            % cumulate

% Resort Inputs
ids = cell(1,ndims(Cutoff.raw_materials.cumulated_inputs));
[ids{:}] = ind2sub(size(Cutoff.raw_materials.cumulated_inputs),1:prod(size(Cutoff.raw_materials.cumulated_inputs)));
ids{1} = Idx(:)';
Cutoff.raw_materials.cumulated_inputs(sub2ind(size(Cutoff.raw_materials.cumulated_inputs),ids{:})) = Cutoff.raw_materials.cumulated_inputs(1:prod(size(Cutoff.raw_materials.cumulated_inputs)));
clear ids; clear Idx;

%% Calculate cumulated mass proportions for Outputs
% Sort and cumulate outputs
[Cutoff.raw_materials.cumulated_outputs, Idx] = sort(Cutoff.raw_materials.proportion_outputs,1);    % sort
Cutoff.raw_materials.cumulated_outputs = cumsum(Cutoff.raw_materials.cumulated_outputs,1);          % cumulate

% Resort Outputs
ids = cell(1,ndims(Cutoff.raw_materials.cumulated_outputs));
[ids{:}] = ind2sub(size(Cutoff.raw_materials.cumulated_outputs),1:prod(size(Cutoff.raw_materials.cumulated_outputs)));
ids{1} = Idx(:)';
Cutoff.raw_materials.cumulated_outputs(sub2ind(size(Cutoff.raw_materials.cumulated_outputs),ids{:})) = Cutoff.raw_materials.cumulated_outputs(1:prod(size(Cutoff.raw_materials.cumulated_outputs)));
clear ids; clear Idx; 

%% Calculate rows_to_delete
% Inputs cumulated
Cutoff.raw_materials.rows_found_cumulated_inputs = [];   % find all input flows that have a small cumulated mass proportion in every process
for i=1:size(Cutoff.raw_materials.cumulated_inputs,1)
    if Cutoff.raw_materials.cumulated_inputs(i,:) < cutoff_rule_cumulated 
        Cutoff.raw_materials.rows_found_cumulated_inputs(end+1,1)=i;
    end
end

% Inputs
Cutoff.raw_materials.rows_found_inputs = [];   % find all flows that have a small mass proportion in matrix proportion (including outputs and units not in kg!!)
for i=1:size(Cutoff.raw_materials.proportion_inputs,1)
    if Cutoff.raw_materials.proportion_inputs(i,:) <= cutoff_rule 
        Cutoff.raw_materials.rows_found_inputs(end+1,1)=i;
    end
end

% Outputs
Cutoff.raw_materials.rows_found_cumulated_outputs = [];   % find all output flows that have a small cumulated mass proportion in every process
for i=1:size(Cutoff.raw_materials.cumulated_outputs,1)
    if Cutoff.raw_materials.cumulated_outputs(i,:) < cutoff_rule_cumulated
        Cutoff.raw_materials.rows_found_cumulated_outputs(end+1,1)=i;
    end
end

% Outputs
Cutoff.raw_materials.rows_found_outputs = [];   % find all flows that have a small mass proportion in matrix proportion (including outputs and units not in kg!!)
for i=1:size(Cutoff.raw_materials.proportion_outputs,1)
    if Cutoff.raw_materials.proportion_outputs(i,:) <= cutoff_rule 
        Cutoff.raw_materials.rows_found_outputs(end+1,1)=i;
    end
end

%% Find Rows with no entries (caused by process deletion)
Cutoff.rows_to_delete_other_units =[];

for i=1:size(Cutoff_final.matrices.A.mean_values,1)
    if Cutoff_final.matrices.A.mean_values(i,:) == 0
        Cutoff.rows_to_delete_other_units(end+1,1)=i;
    end
end

        
% Intersection of Inputs and Outpts, each cumulated and not cumulated
Cutoff.raw_materials.rows_found = intersect(intersect(Cutoff.raw_materials.rows_found_cumulated_outputs, Cutoff.raw_materials.rows_found_cumulated_inputs),intersect(Cutoff.raw_materials.rows_found_outputs, Cutoff.raw_materials.rows_found_inputs));

% rows_to_delete are all rows that have a small mass proportion (in and outputs) except for flows not in kg and utilities
Cutoff.rows_to_delete_kg=setdiff(setdiff(Cutoff.raw_materials.rows_found,Cutoff.raw_materials.rows_other_units),Cutoff.utilities.rows_other_type)';

Cutoff.rows_to_delete = union (Cutoff.rows_to_delete_kg,Cutoff.rows_to_delete_other_units);
%% Modify matrices

A_mean_cutoff=Cutoff_final.matrices.A.mean_values(Cutoff.rows_to_delete,:);
% y_cutoff = Cutoff_final.matrices.y.mean_values(Cutoff.rows_to_delete,:);
meta_data_cutoff = [Model.meta_data_flows(1,:);Model.meta_data_flows(Cutoff.rows_to_delete+1,:)];

% Modify Meta Data 
Cutoff_final.meta_data_flows = Model.meta_data_flows;
Cutoff_final.meta_data_flows(Cutoff.rows_to_delete+1,:) = []; % cutoff meta data (raw materials and utilities)

Cutoff_final.meta_data_factor_requirements = Model.meta_data_factor_requirements;
Cutoff_final.meta_data_impact_categories = Model.meta_data_impact_categories;

% Modify Matrix A
Cutoff_final.matrices.A.mean_values(Cutoff.rows_to_delete,:) = [];

%% List all flows deleted in matlab

Cutoff_final.cutoff.A=A_mean_cutoff;
Cutoff_final.cutoff.meta_data=meta_data_cutoff;

%% Create hybrid matrices

Cutoff_final.hybrid.A_hybrid=[];
Cutoff_final.hybrid.meta_data_hybrid=Model.meta_data_flows(1,:);
found_rows_hybrid=[];

row=2;
for i=1:size(Cutoff_final.matrices.A.mean_values,1)
    % create matrices for hybrid (units $, pcs and BOAT)
    if strcmp('$',Cutoff_final.meta_data_flows(i+1,6)) == 1 || strcmp('pcs',Cutoff_final.meta_data_flows(i+1,6)) == 1 || strcmp('BOAT',Cutoff_final.meta_data_flows(i+1,6)) == 1
        Cutoff_final.hybrid.A_hybrid(end+1,:)=Cutoff_final.matrices.A.mean_values(i,:);
        
        Cutoff_final.hybrid.meta_data_hybrid(row,:)=Cutoff_final.meta_data_flows(i+1,:);
        found_rows_hybrid(1,end+1)=i;
        row=row+1;
    end
end
clear row i

%% Include cutoffs in hybrid matrices

Cutoff_final.hybrid.A_hybrid=[Cutoff_final.hybrid.A_hybrid;A_mean_cutoff];
Cutoff_final.hybrid.meta_data_hybrid=[Cutoff_final.hybrid.meta_data_hybrid;meta_data_cutoff(2:end,:)];


%% Delete hybid flows in Matrix A,y and meta_data

% Modify Meta Data 
Cutoff_final.meta_data_flows(found_rows_hybrid+1,:) = []; 

% Modify Matrix A
Cutoff_final.matrices.A.mean_values(found_rows_hybrid,:) = [];

clear found_rows_hybrid

%% unit  conversion of hybrid matrices --> Matrices A_hybrid_dollar, y_hybrid_dollar, meta_data_hybrid_dollar
 Cutoff_final.hybrid.A_hybrid_dollar=Cutoff_final.hybrid.A_hybrid;
 Cutoff_final.hybrid.meta_data_hybrid_dollar=Cutoff_final.hybrid.meta_data_hybrid;
 
for i=1:size(Cutoff_final.hybrid.A_hybrid,1)
    if strcmp('$',Cutoff_final.hybrid.meta_data_hybrid(i+1,6)) == 0
        if isempty(Cutoff_final.hybrid.meta_data_hybrid{i+1,7})
           Cutoff_final.hybrid.meta_data_hybrid{i+1,7}=0; 
        end
        Cutoff_final.hybrid.A_hybrid_dollar(i,:)=Cutoff_final.hybrid.A_hybrid_dollar(i,:)./cell2mat(Cutoff_final.hybrid.meta_data_hybrid(i+1,7));
%         Cutoff_final.hybrid.y_hybrid_dollar(i,:)=Cutoff_final.hybrid.y_hybrid_dollar(i,:)./cell2mat(Cutoff_final.hybrid.meta_data_hybrid(i+1,7));
        Cutoff_final.hybrid.meta_data_hybrid_dollar{i+1,6}='$';
        Cutoff_final.hybrid.meta_data_hybrid_dollar(i+1,7)=mat2cell(1,1);
    end
end

Model=Cutoff_final; 

