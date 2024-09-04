function [A_eco_mean, B_eco_mean, F_eco_mean,eco_process_meta] = include_EcoinventCoreLayer(Model,EcoinventModelCoreLayer,regions)

load(EcoinventModelCoreLayer);

%% get only the prioritized regions from the regions list
BackgroundModel.rowA.Prio(:) = 0;
regions = string(regions);

for i=1:size(regions,1)
    BackgroundModel.rowA.Prio(BackgroundModel.rowA.location == regions(i,1)) = i;
end

BackgroundModel.rowA(BackgroundModel.rowA.Prio == 0,:) = [];
BackgroundModel.rowA = sortrows(BackgroundModel.rowA,'Prio','ascend');
unique_chemicals = unique(BackgroundModel.rowA.name);
first_appearances = zeros(size(unique_chemicals));
for i = 1:length(unique_chemicals)
    first_appearances(i) = find(strcmp(BackgroundModel.rowA.name, unique_chemicals(i)), 1);
end
BackgroundModel.rowA = BackgroundModel.rowA(first_appearances, :);
BackgroundModel.rowA = removevars(BackgroundModel.rowA, 'Prio');

% remove chemicals from Background model that are not needed in the core
% Layer
BackgroundModel.rowA(~ismember(BackgroundModel.rowA.name,string(Model.meta_data_flows(2:end,1))),:) = [];

%% adjust all other tables
rowA_toKeep = BackgroundModel.rowA.rowA;
colP_toKeep = BackgroundModel.rowA.colP;
BackgroundModel.colP(~ismember(BackgroundModel.colP.colP,colP_toKeep),:) = [];
BackgroundModel.A(~ismember(BackgroundModel.A.colP,colP_toKeep),:) = [];
BackgroundModel.A(~ismember(BackgroundModel.A.rowA,rowA_toKeep),:) = [];
BackgroundModel.B(~ismember(BackgroundModel.B.colP,colP_toKeep),:) = [];
BackgroundModel.F(~ismember(BackgroundModel.F.colP,colP_toKeep),:) = [];

%% sort rowA according to Model
% Find position of rows in BackgroundModel in Model
[~, idx] = ismember(BackgroundModel.rowA.name, string(Model.meta_data_flows(2:end,1)), 'rows');
BackgroundModel.rowA.rowANew = idx;

[~, idx] = ismember(BackgroundModel.A.rowA, BackgroundModel.rowA.rowA, 'rows');
BackgroundModel.A.rowANew = BackgroundModel.rowA.rowANew(idx);
BackgroundModel.A = removevars(BackgroundModel.A, 'rowA');
BackgroundModel.A.Properties.VariableNames{'rowANew'} = 'rowA';
BackgroundModel.A(BackgroundModel.A.rowA ==0,:) = [];

BackgroundModel.rowA = removevars(BackgroundModel.rowA, 'rowA');
BackgroundModel.rowA.Properties.VariableNames{'rowANew'} = 'rowA';
BackgroundModel.rowA(BackgroundModel.rowA.rowA ==0,:) = [];

%% sort rowB according to Model
% create keys
BackgroundModel.rowB.key = BackgroundModel.rowB.flow + BackgroundModel.rowB.compartment+BackgroundModel.rowB.sub_compartment+BackgroundModel.rowB.units;
B = table(string(Model.meta_data_elementary_flows(2:end,1)),string(Model.meta_data_elementary_flows(2:end,2)),string(Model.meta_data_elementary_flows(2:end,3)),string(Model.meta_data_elementary_flows(2:end,4)));
B.key = B.Var1+B.Var2+B.Var3+B.Var4;

% Find position of rows in BackgroundModel in Model
[~, idx] = ismember(BackgroundModel.rowB.key, B.key, 'rows');
BackgroundModel.rowB.rowBNew = idx;

[~, idx] = ismember(BackgroundModel.B.rowB, BackgroundModel.rowB.rowB, 'rows');
BackgroundModel.B.rowBNew = BackgroundModel.rowB.rowBNew(idx);
BackgroundModel.B = removevars(BackgroundModel.B, 'rowB');
BackgroundModel.B.Properties.VariableNames{'rowBNew'} = 'rowB';
BackgroundModel.B(BackgroundModel.B.rowB ==0,:) = [];

BackgroundModel.rowB = removevars(BackgroundModel.rowB, 'rowB');
BackgroundModel.rowB.Properties.VariableNames{'rowBNew'} = 'rowB';
BackgroundModel.rowB(BackgroundModel.rowB.rowB ==0,:) = [];

%% sort rowF according to Model
% Find position of rows in BackgroundModel in Model
[~, idx] = ismember(BackgroundModel.rowF.nameF, string(Model.meta_data_factor_requirements(2:end,1)), 'rows');
BackgroundModel.rowF.rowFNew = idx;

[~, idx] = ismember(BackgroundModel.F.rowF, BackgroundModel.rowF.rowF, 'rows');
BackgroundModel.F.rowFNew = BackgroundModel.rowF.rowFNew(idx);
BackgroundModel.F = removevars(BackgroundModel.F, 'rowF');
BackgroundModel.F.Properties.VariableNames{'rowFNew'} = 'rowF';
BackgroundModel.F(BackgroundModel.F.rowF ==0,:) = [];

BackgroundModel.rowF = removevars(BackgroundModel.rowF, 'rowF');
BackgroundModel.rowF.Properties.VariableNames{'rowFNew'} = 'rowF';
BackgroundModel.rowF(BackgroundModel.rowF.rowF ==0,:) = [];

%% create colP from 1:n
% Find position of rows in BackgroundModel in Model
[~, idx] = ismember(BackgroundModel.rowA.name, string(Model.meta_data_flows(2:end,1)), 'rows');
BackgroundModel.rowA.colPNew = (1:height(BackgroundModel.rowA))';

[~, idx] = ismember(BackgroundModel.A.colP, BackgroundModel.rowA.colP, 'rows');
BackgroundModel.A.colPNew = BackgroundModel.rowA.colPNew(idx);
BackgroundModel.A = removevars(BackgroundModel.A, 'colP');
BackgroundModel.A.Properties.VariableNames{'colPNew'} = 'colP';
BackgroundModel.A(BackgroundModel.A.colP ==0,:) = [];

[~, idx] = ismember(BackgroundModel.B.colP, BackgroundModel.rowA.colP, 'rows');
BackgroundModel.B.colPNew = BackgroundModel.rowA.colPNew(idx);
BackgroundModel.B = removevars(BackgroundModel.B, 'colP');
BackgroundModel.B.Properties.VariableNames{'colPNew'} = 'colP';
BackgroundModel.B(BackgroundModel.B.colP ==0,:) = [];

[~, idx] = ismember(BackgroundModel.F.colP, BackgroundModel.rowA.colP, 'rows');
BackgroundModel.F.colPNew = BackgroundModel.rowA.colPNew(idx);
BackgroundModel.F = removevars(BackgroundModel.F, 'colP');
BackgroundModel.F.Properties.VariableNames{'colPNew'} = 'colP';
BackgroundModel.F(BackgroundModel.F.colP ==0,:) = [];

[~, idx] = ismember(BackgroundModel.colP.colP, BackgroundModel.rowA.colP, 'rows');
BackgroundModel.colP.colPNew = BackgroundModel.rowA.colPNew(idx);
BackgroundModel.colP = removevars(BackgroundModel.colP, 'colP');
BackgroundModel.colP.Properties.VariableNames{'colPNew'} = 'colP';
BackgroundModel.colP(BackgroundModel.colP.colP ==0,:) = [];

BackgroundModel.rowA = removevars(BackgroundModel.rowA, 'colP');
BackgroundModel.rowA.Properties.VariableNames{'colPNew'} = 'colP';
BackgroundModel.rowA(BackgroundModel.rowA.colP ==0,:) = [];

%% create tables in matrix format
% Create A_eco_mean
A_eco_mean = sparse(BackgroundModel.A.rowA, BackgroundModel.A.colP, BackgroundModel.A.coefficientA, size(Model.matrices.A.mean_values,1), max(BackgroundModel.colP.colP));
A_eco_mean = full(A_eco_mean);
% Create B_eco_mean
B_eco_mean = sparse(BackgroundModel.B.rowB, BackgroundModel.B.colP, BackgroundModel.B.coefficientB, size(Model.matrices.B.mean_values,1), max(BackgroundModel.colP.colP));
B_eco_mean = full(B_eco_mean);
% Create F_eco_mean
if ~isempty(BackgroundModel.F)
    F_eco_mean = sparse(BackgroundModel.F.rowF, BackgroundModel.A.colP, BackgroundModel.F.coefficientF, size(Model.matrices.F.mean_values,1), max(BackgroundModel.colP.colP));
    F_eco_mean = full(F_eco_mean);
else
    F_eco_mean = zeros(size(Model.matrices.F.mean_values,1), max(BackgroundModel.colP.colP));
end
% Create eco_process_meta
colP = BackgroundModel.colP;
eco_process_meta = cell(10, max(BackgroundModel.colP.colP));
eco_process_meta(1,:) = cellstr(colP.name');
eco_process_meta(4,:) = cellstr(colP.mainFlow');
eco_process_meta(5,:) = cellstr(colP.countryCode');
eco_process_meta(10,:) = repmat({4}, 1, max(BackgroundModel.colP.colP));

end