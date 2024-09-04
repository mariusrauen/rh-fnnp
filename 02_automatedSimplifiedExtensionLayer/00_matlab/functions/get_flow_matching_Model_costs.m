function [ Model ] = get_flow_matching_Model_costs( Model , Control_match )
% FLOW_MATCHING factor requierment matching for chemistry database building

%% DISCLAIMER
% make sure to close the control sheet before executing the function!

Model.Control_match_costs.deleted_metadata={[]};
Model.Control_match_costs.log.skipped_rows={[]};
Model.Control_match_costs.log.created_flows={[]};
Model.Control_match_costs.log.main_flows={[]};
deleted_metadata={[]};
Control_match.log.skipped_rows={[]};
Control_match.log.created_flows={[]};
Control_match.log.main_flows={'1'};

%% MANIPULATE MATRICES
% prepare matrix containing information about deleted metadata with headers
deleted_metadata=cell(length(Control_match.names_1)+1,3); 
deleted_metadata(1,:) = [ {'name'} {'unit'} {'comments'} ]; 

k = 1; % write index for skipped rows
l = 1; % write index for created flows

% execute the manipulations
for i = 1:length(Control_match.names_1) 
    flag = false; % reset factor requierment 2 not found flag 
    
    % find row number of second factor requierment
    for j = 2:size(Model.meta_data_factor_requirements,1)
        
       if(strcmp(Control_match.names_2(i) , Model.meta_data_factor_requirements{j,1} )) % compare names
               if isequal(Control_match.units_2{i},Model.meta_data_factor_requirements{j,2}) % compare unit
               Control_match.row_nums_2(i) = j - 1; % save if found. meta_data has titles, therefore j - 1
               break; % stop searching
               end          
       end
       
       if(j == size(Model.meta_data_factor_requirements,1)) % factor requierment was not found
           flag = true;
           break; % stop searching
       end
       
    end
    
    clear j
    
    % log if row was skipped because of non-existance
    if flag == true;
        Control_match.log.skipped_rows{k,1} = i;
        Control_match.log.skipped_rows{k,2} = 'Factor requierment 2 was not found.';
        k = k + 1;
        continue; % skip row (go to next manipulation)
    end
    
    % find row number of first factor requierment
    for j = 2:size(Model.meta_data_factor_requirements,1)
        
       if(strcmp(Control_match.names_1{i} , Model.meta_data_factor_requirements{j,1} ))
               if isequal(Control_match.units_1{i}, Model.meta_data_factor_requirements{j,2})
               Control_match.row_nums_1(i) = j - 1; % save row number. meta_data_flows has titles therefore j - 1
               break;
               end          
       end
       
       if(j == size(Model.meta_data_factor_requirements,1)) % factor requierment was not found
           % log created factor requierment
           Control_match.log.created_flows{l,1} = i;
           Control_match.log.created_flows{l,2} = Control_match.names_1{i};
           Control_match.log.created_flows{l,3} = Control_match.units_1{i};
           l = l + 1;
           
           % create new empty factor requierment row in all relevant A submatrices
           Model.matrices.F.mean_values = [Model.matrices.F.mean_values;zeros(1,size(Model.matrices.F.mean_values,2))];
         
           % create new factor requierment in meta_data_factor_requirement matrix
           new_metadata = cell(1,size(Model.meta_data_factor_requirements,2)); %empty cell line %_________________
           new_metadata{1} = Control_match.names_1{i}; % assign name
           new_metadata{2} = Control_match.units_1{i}; % assign unit
           
           % assign new metadata 
           Model.meta_data_factor_requirements = [Model.meta_data_factor_requirements;new_metadata];
           % assign index
           Control_match.row_nums_1(i) = size(Model.matrices.F.mean_values,1);
       end
    end
    
    clear j
    
    % check if names, categorie and unitss of factor requierment 1 and 2 equal 
    if( Control_match.row_nums_1(i) == Control_match.row_nums_2(i) )
            if isequal(Control_match.units_1(i),Control_match.units_2(i))
            % log if equal
            Control_match.log.skipped_rows{k,1} = i;
            Control_match.log.skipped_rows{k,2} = 'Factor requierment 1 and Factor requierment 2 are equal.';
            k = k + 1;
            continue; % skip to next manipulation
            end
    end
    
   
    current_flow_row_1 = Control_match.row_nums_1(i);
    current_flow_row_2 = Control_match.row_nums_2(i);
    
    % calculate new row in A.mean_values according to rule
    Model.matrices.F.mean_values(current_flow_row_1,:) = ...
        Model.matrices.F.mean_values(current_flow_row_1,:) + Model.matrices.F.mean_values(current_flow_row_2,:) * Control_match.conv_factors(i);
 
    % check for different metadata from factor requierment to be deleted and save it
    for j = 1:size(Model.meta_data_factor_requirements,2)
        deleted_metadata{i+1,1} = Control_match.names_2{i}; % save name of deleted factor requierment 
        
        if(j == 1 || j == 2 || j == 3 ) % for string format cells
            if(~strcmp(Model.meta_data_factor_requirements{current_flow_row_1 + 1,j},Model.meta_data_factor_requirements{current_flow_row_2 + 1,j})) % compare metadata field per field
                deleted_metadata{i+1,j} = Model.meta_data_factor_requirements{current_flow_row_2+1,j}; % save if not equal
            end
            
        else % for double type format cells
            if(Model.meta_data_factor_requirements{current_flow_row_1 + 1,j} ~= Model.meta_data_factor_requirements{current_flow_row_2 + 1,j}) % compare metadata field per field
                deleted_metadata{i+1,j} = Model.meta_data_factor_requirements{current_flow_row_2+1,j}; % save if not equal
            end
        end
        
    end
    
    clear j
    
    % delete row of second factor requierment in all relevant F submatrices
    Model.matrices.F.mean_values(current_flow_row_2,:) = [];
    
    % delete row of second factor requierment in meta data
    Model.meta_data_factor_requirements(current_flow_row_2+1,:) = []; % +1 because of header
    
end

clear i

if ~isempty(Control_match.log.main_flows{1,1})
Control_match.log.main_flows = Control_match.log.main_flows(~cellfun(@isempty, Control_match.log.main_flows));
Control_match.log.main_flows = unique(Control_match.log.main_flows(~cellfun(@isempty, Control_match.log.main_flows)));
end

%% save in model
Model.Control_match_costs.deleted_metadata=deleted_metadata;
Model.Control_match_costs.log.skipped_rows=Control_match.log.skipped_rows;
Model.Control_match_costs.log.created_flows=Control_match.log.created_flows;
Model.Control_match_costs.log.main_flows=(Control_match.log.main_flows);

end

