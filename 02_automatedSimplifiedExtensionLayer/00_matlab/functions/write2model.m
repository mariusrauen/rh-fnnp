function Model =  write2model(process_list)

%% create empty sheets
technical_flows = [];
A_distribution = [];
A_mean = [];
A_std_dev = [];
factor_requirements = [];
F_distribution = [];
F_mean = [];
F_std_dev = [];

for i = 1:length(process_list)
   
    %% Add Process Info
    processes{1,i} = process_list(i).info.name;
    processes{2,i} = process_list(i).info.abbrevation;
    processes{3,i} = process_list(i).info.process_description;
    processes{4,i} = process_list(i).info.mainflow;
    processes{5,i} = process_list(i).info.location;
    processes{6,i} = process_list(i).info.exact_location;
    processes{7,i} = process_list(i).info.capacity;
    processes{8,i} = process_list(i).info.unit_per_year;
    processes{9,i} = '';
    processes{10,i} = '';
    
    %% Technical Flows 
    % search if stream that already exists 
    for k = 1:length(process_list(i).streams)
        flag = 1;
        for j = 1:size(technical_flows,1) 
            if strcmp(technical_flows{j,1},process_list(i).streams(k).name) && (technical_flows{j,2} ==  process_list(i).streams(k).class || k==1) && strcmp(technical_flows{j,6},process_list(i).streams(k).amount_unit)
                
                A_distribution{j,i} = 2;
                A_mean{j,i} = process_list(i).streams(k).amount;
                
                if isnan(A_mean{j,i})
                    A_mean{j,i} = process_list(i).streams(k).cost_per_kg;
                end
                
                A_std_dev{j,i} = 0;
                flag = 0;
                break;
            end
        end
        
        %% add new flow to tec_streams
        if flag
            technical_flows{end+1,1} =  process_list(i).streams(k).name; %name
            technical_flows{end,2} =  process_list(i).streams(k).class;  %category
            technical_flows{end,3} = ''; %concentration
            technical_flows{end,4} = ''; %CAS
            technical_flows{end,5} = process_list(i).streams(k).unit_type; %unit typ
            if isnan(process_list(i).streams(k).cost_unit)
                technical_flows{end,5} = 'Pieces';
            end
            technical_flows{end,6} = process_list(i).streams(k).amount_unit; %unit
            if isnan(process_list(i).streams(k).cost_unit)
                technical_flows{end,6} = '$';
            end
            technical_flows{end,7} = process_list(i).streams(k).cost; %cost
            if isnan(process_list(i).streams(k).cost)
                technical_flows{end,7} = 1;
            end
            technical_flows{end,8} = 0; %LHV
            technical_flows{end,9} = 0; %HHV
            technical_flows{end,10} = 0; %Formula 
            technical_flows{end,11} = '';
            technical_flows{end,12} = '';
            technical_flows{end,13} = '';
            technical_flows{end,14} = '';
            technical_flows{end,15} = '';
            technical_flows{end,16} = '';
            technical_flows{end,17} = '';
            
            if isempty(j)
                j = 0;
            end
            A_distribution{j+1,i} = 2;
            A_mean{j+1,i} = process_list(i).streams(k).amount;
            A_std_dev{j+1,i} = 0;
            if isnan(A_mean{j+1,i})
                A_mean{j+1,i} = process_list(i).streams(k).cost_per_kg;
            end
        end
    end
    
    %% Factor requirement flows
    for k = 1:length(process_list(i).costs)
        flag = 1;
        for j = 1:size(factor_requirements,1) 
            if strcmp(factor_requirements{j,1},process_list(i).costs(k).name)
                F_distribution{j,i} = 2;
                F_mean{j,i} = process_list(i).costs(k).value;
                F_std_dev{j,i} = 0;
                flag = 0;
                break;
            end
        end    
        %% add new flow to tec_streams
        if flag
            factor_requirements{end+1,1} =  process_list(i).costs(k).name; %name
            factor_requirements{end,2} = process_list(i).costs(k).unit;
            factor_requirements{end,3} = '';
            
            if isempty(j)
                j = 0;
            end
            
            F_distribution{j+1,i} = 2;
            F_mean{j+1,i} = process_list(i).costs(k).value;
            F_std_dev{j+1,i} = 0;
        end
    end
end

%% create k 

k_mean = cell(size(F_mean,1),1);
k_distribution = cell(size(F_mean,1),1);
k_std_dev = cell(size(F_mean,1),1);

k_mean(:,1) = {1}; 
k_distribution(1:end) = {2}; 


%% Add legend to Tables

%% Technical Flows
tech_name{1,1} = 'name';
tech_name{1,2} = 'category';
tech_name{1,3} = 'concentration/purity';
tech_name{1,4} = 'CAS-Nr.';
tech_name{1,5} = 'unit(choice)';
tech_name{1,6} = '[unit choice]';
tech_name{1,7} = 'Market price';
tech_name{1,8} = 'LHV';
tech_name{1,9} = 'HHV';
tech_name{1,10} = 'chemical formular[C2C4 format]';
tech_name{1,11} = 'location(choice)';
tech_name{1,12} = 'exact location';
tech_name{1,13} = 'comments';
tech_name{1,14} = 'SMILES CODE';
tech_name{1,15} = 'molecular mass';
tech_name{1,16} = 'flowCategory';
tech_name{1,17} = 'flowSubcategory';

%% Processes
process_names{1,1} = 'name';
process_names{2,1} = 'abbrevation';
process_names{3,1} = 'process description';
process_names{4,1} = 'mainflow';
process_names{5,1} = 'location (choice)';
process_names{6,1} = 'exact location';
process_names{7,1} = 'capacity';
process_names{8,1} = 'unit per year (choice)';
process_names{9,1} = 'comments';
process_names{10,1} = 'type';
%% factor requierments
req_names{1,1} = 'name';
req_names{1,2} = 'unit';
req_names{1,3} = 'comment';

%% cost
cost_name{1,1} = 'name';
cost_name{1,2} = 'cost';

%% Add the legends 
processes = horzcat(process_names,processes);
technical_flows = vertcat(tech_name,technical_flows);
factor_requirements = vertcat(req_names,factor_requirements);

% process_names = processes(1,1:end);
% flow_names = technical_flows(2:end,1);
% requirement_names = factor_requirements(2:end,1);

% A_distribution = horzcat(flow_names,A_distribution);
% A_distribution = vertcat(process_names,A_distribution);

% A_mean = horzcat(flow_names,A_mean);
% A_mean = vertcat(process_names,A_mean);

% A_std_dev = horzcat(flow_names,A_std_dev);
% A_std_dev = vertcat(process_names,A_std_dev);

% k_mean = horzcat(requirement_names,k_mean);
% k_mean = vertcat(cost_name,k_mean);
% 
% k_std_dev = horzcat(requirement_names,k_std_dev);
% k_std_dev = vertcat(cost_name,k_std_dev);
% 
% k_distribution = horzcat(requirement_names,k_distribution);
% k_distribution = vertcat(cost_name,k_distribution);
% 
% F_distribution = horzcat(requirement_names,F_distribution);
% F_distribution = vertcat(process_names,F_distribution);
% 
% F_mean = horzcat(requirement_names,F_mean);
% F_mean = vertcat(process_names,F_mean);
% 
% F_std_dev = horzcat(requirement_names,F_std_dev);
% F_std_dev = vertcat(process_names,F_std_dev);


Model.meta_data_flows = technical_flows;
Model.meta_data_processes = processes;

A_mean(find(cellfun(@(C)...
any(isempty(C(:))), A_mean)))={0};  

A_mean = cell2mat(A_mean);

Model.matrices.A.mean_values = A_mean;

% Model.matrices.A.distribution = A_distribution;
% Model.matrices.A.std_dev = A_std_dev;

Model.meta_data_factor_requirements = factor_requirements;

F_mean(find(cellfun(@(C)...
any(isempty(C(:))), F_mean)))={0};  

F_mean = cell2mat(F_mean);

Model.matrices.F.mean_values = F_mean;

% Model.matrices.F.distribution = F_distribution;
% Model.matrices.F.std_dev = F_std_dev;

k_mean(find(cellfun(@(C)...
any(isempty(C(:))), k_mean)))={0};  

k_mean = cell2mat(k_mean);

Model.matrices.k.mean_values = k_mean;

% Model.matrices.k.distribution = k_distribution;
% Model.matrices.k.std_dev = k_std_dev;

end