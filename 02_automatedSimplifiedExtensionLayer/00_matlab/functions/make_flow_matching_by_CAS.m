function [flow_matching] = make_flow_matching_by_CAS(flows_CLICC, flows, path_inputs)

%% get data of kept flows
clear flow_matching

flow_matching.names_1 = flows_CLICC(:,1);
flow_matching.cats_1 = ones(size(flows_CLICC,1),1);
flow_matching.units_1 = flows_CLICC(:,2);

counter = 0;
for j = 1 : size(flow_matching.names_1,1)
    
    if isequal(flow_matching.names_1{j},'Electricity (MJ)') || isequal(flow_matching.names_1{j},'Steam (MJ)')
        counter = counter + 1;
        row_deleted(counter) = j;
        
    end
    
end

flow_matching.names_1(row_deleted) = [];
flow_matching.cats_1(row_deleted) = [];
flow_matching.units_1(row_deleted) = [];

clear counter row_deleted

%% get data of IHS matched
counter = 0;
flow_matching.names_2 = cell(size(flow_matching.names_1));
flow_matching.cats_2 = ones(size(flow_matching.cats_1));
flow_matching.units_2 = cell(size(flow_matching.units_1));

for i = 1 : size(flow_matching.names_1,1)
    
    row_IHS = find(strcmp(flows_CLICC{i,3},flows(:,4)));
    
    if isempty(row_IHS)
        
        counter = counter + 1;
        row_deleted(counter) = i;
        
    elseif isequal(size(row_IHS,1),1)
        
        flow_matching.names_2(i,1) = flows(row_IHS,1);
        flow_matching.cats_2(i,1) = flows{row_IHS,2};
        flow_matching.units_2(i,1) = flows(row_IHS,6);
        
    else %% double CAS in IHS data (not allowed) --> revise
        
        counter = counter + 1;
        row_deleted(counter) = i;
        
    end
    
end

flow_matching.names_2(row_deleted) = [];
flow_matching.cats_2(row_deleted) = [];
flow_matching.units_2(row_deleted) = [];

flow_matching.names_1(row_deleted) = [];
flow_matching.cats_1(row_deleted) = [];
flow_matching.units_1(row_deleted) = [];

% change to keep IHS
CLICC.names_1 = flow_matching.names_1;
CLICC.cats_1 = flow_matching.cats_1;
CLICC.units_1 = flow_matching.units_1;

flow_matching.names_1 = flow_matching.names_2;
flow_matching.cats_1 = flow_matching.cats_2;
flow_matching.units_1 = flow_matching.units_2;

flow_matching.names_2 = CLICC.names_1;
flow_matching.cats_2 = CLICC.cats_1;
flow_matching.units_2 = CLICC.units_1;

flow_matching.conv_factors = ones(size(flow_matching.names_1));

clear counter row_deleted

%% finalize with Electricity and Steam matching
[flow_matching] = get_flow_matching_CLICC(path_inputs, flow_matching);

end