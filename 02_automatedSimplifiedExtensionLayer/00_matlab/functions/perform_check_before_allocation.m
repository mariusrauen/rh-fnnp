function [flows_missingLHV_energy,problematic_processes_energy,flows_missingLHV_price,problematic_processes_price] =  perform_check_before_allocation(Model)

%% Get A Matrix only with output values and without utilities
% only outputs
A_alloc = Model.matrices.A.mean_values;
A_alloc(A_alloc<0) = 0;
% without utilities
lns_utilities = cell2mat(Model.meta_data_flows(2:end,2)) == 2;
A_alloc(lns_utilities,:) = 0;
% if there is only one output -> no allocation has to be performed -> we
% can set this process to 0
counter = A_alloc;
counter(counter>0) = 1;
counter = sum(counter,1);
lns_oneoutput = counter == 1;
A_alloc(:,lns_oneoutput) = 0;

clear lns_utilities lns_oneoutput

%% Get Only Processes where energy allocation should be performed
allocationType = nan(size(Model.meta_data_processes(10,:)));
idx_strings = cellfun(@ischar,Model.meta_data_processes(10,:));
allocationType(~idx_strings) = cell2mat(Model.meta_data_processes(10,~idx_strings));
allocationType(idx_strings) = 0;

lns_energyalloc = allocationType(2:end) == 1;
% remove all processes from A Matrix which are not allocated by energy
A = A_alloc;
A(:,~lns_energyalloc) = 0;
clear lns_energyalloc

% Find all flows that still have positive outputs somewhere in A Matrix
lns_alloc = sum(A,2) ~= 0;

%% Find all flows that are allocated by energy but don't have a LHV
lns_pos = cell2mat(Model.meta_data_flows(2:end,9)) > 0;
lns_neg = cell2mat(Model.meta_data_flows(2:end,9)) < 0;
lns_missing = ~or(lns_pos,lns_neg);
%clear lns_pos lns_neg

lns_problematicflows = and(lns_missing,lns_alloc);

flows_missingLHV = [];
problematic_processes = [];

%% If there are missing LHVs: write flows which are missing the LHV value and the corresponding 
if ~isempty(find(lns_problematicflows,1))
    % write all flows where LHV is missing but needed for allocation to flows_missingLHV
    flows_missingLHV = Model.meta_data_flows([true;lns_problematicflows],:);
    VarNames = flows_missingLHV(1,:);
    flows_missingLHV = cell2table(flows_missingLHV(2:end,:));
    flows_missingLHV.Properties.VariableNames = VarNames;
    
    % find all processes which are problematic, because at least 1 of their
    % outputs cannot be allocated by energy due to missing LHV
    A(~lns_problematicflows,:) = 0;
    [row,col] = find(A);
    processName = string(Model.meta_data_processes(1,col+1)');
    mainFlow = string(Model.meta_data_processes(4,col+1)');
    problematicOutputFlow = string(Model.meta_data_flows(row+1,1));
    problematic_processes = table(processName,mainFlow,problematicOutputFlow,'VariableNames',{'processName','mainFlow','problematicOutputFlow'});
end

flows_missingLHV_energy = flows_missingLHV;
problematic_processes_energy = problematic_processes;



%% Get Only Processes where price allocation could be performed
lns_pricealloc = allocationType(2:end) == 2;
A = A_alloc;
A(:,~lns_pricealloc) = 0;
clear lns_pricealloc

% Find all flows that still have positive outputs somewhere in A Matrix
lns_alloc = sum(A,2) ~= 0;

%% Find all flows that are allocated by energy but don't have a LHV
lns_pos = cell2mat(Model.meta_data_flows(2:end,7)) > 0;
lns_neg = cell2mat(Model.meta_data_flows(2:end,7)) < 0;
lns_missing = ~or(lns_pos,lns_neg);
clear lns_pos lns_neg

lns_problematicflows = and(lns_missing,lns_alloc);

flows_missingLHV = [];
problematic_processes = [];

%% If there are missing LHVs: write flows which are missing the LHV value and the corresponding 
if ~isempty(find(lns_problematicflows,1))
    % write all flows where LHV is missing but needed for allocation to flows_missingLHV
    flows_missingLHV = Model.meta_data_flows([true;lns_problematicflows],:);
    VarNames = flows_missingLHV(1,:);
    flows_missingLHV = cell2table(flows_missingLHV(2:end,:));
    flows_missingLHV.Properties.VariableNames = VarNames;
    
    % find all processes which are problematic, because at least 1 of their
    % outputs cannot be allocated by energy due to missing LHV
    A(~lns_problematicflows,:) = 0;
    [row,col] = find(A);
    processName = string(Model.meta_data_processes(1,col+1)');
    mainFlow = string(Model.meta_data_processes(4,col+1)');
    problematicOutputFlow = string(Model.meta_data_flows(row+1,1));
    problematic_processes = table(processName,mainFlow,problematicOutputFlow,'VariableNames',{'processName','mainFlow','problematicOutputFlow'});
end

flows_missingLHV_price = flows_missingLHV;
problematic_processes_price = problematic_processes;


end

