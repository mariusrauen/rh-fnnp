function [ Model ] = matching_flows_layer_1_2_3( Model , Control_match )
%FLOW_MATCHING Flow matching for chemistry database building

%% DISCLAIMER
% make sure to close the control sheet before executing the function!
Model.Control_match.deleted_metadata={[]};
Model.Control_match.log.skipped_rows={[]};
Model.Control_match.log.created_flows={[]};
Model.Control_match.log.main_flows={[]};
deleted_metadata={[]};
Control_match.log.skipped_rows={[]};
Control_match.log.created_flows={[]};
Control_match.log.main_flows={'1'};

%% MANIPULATE MATRICES
% prepare matrix containing information about deleted metadata with headers
deleted_metadata=cell(length(Control_match.names_1)+1,14); 
deleted_metadata(1,:) = [ {'name'} {'category'} {'concentration/purity'} {'CAS-Nr.'} {'unit(choice)'} {'[unit choice]'} {'Market price'} ...
    {'LHV'} {'HHV'} {'chemical formular[C2C4 format]'} {'location(choice)'} {'exact location'} {'comments'} {'SMILES CODE'}]; 

k = 1; % write index for skipped rows
l = 1; % write index for created flows

% execute the manipulations
for i = 1:length(Control_match.names_1) 
    flag = false; % reset flow 2 not found flag 
    
    % find row number of second flow
    for j = 2:size(Model.meta_data_flows,1)
        
       if(strcmp(Control_match.names_2(i) , Model.meta_data_flows{j,1} )) % compare names
           if (Control_match.cats_2(i) == Model.meta_data_flows{j,2}) % compare categories
               if isequal(Control_match.units_2{i},Model.meta_data_flows{j,6}) % compare unit
               Control_match.row_nums_2(i) = j - 1; % save if found. meta_data has titles, therefore j - 1
               break; % stop searching
               end
           end          
       end
       
       if(j == size(Model.meta_data_flows,1)) % flow was not found
           flag = true;
           break; % stop searching
       end
       
    end
    
    % log if row was skipped because of non-existance
    if flag == true;
        Control_match.log.skipped_rows{k,1} = i;
        Control_match.log.skipped_rows{k,2} = 'Flow 2 was not found.';
        k = k + 1;
        continue; % skip row (go to next manipulation)
    end
    
    % find row number of first flow
    for j = 2:size(Model.meta_data_flows,1)
        
       if(strcmp(Control_match.names_1{i} , Model.meta_data_flows{j,1} ))
           if (Control_match.cats_1(i) == Model.meta_data_flows{j,2})
               if isequal(Control_match.units_1{i},Model.meta_data_flows{j,6})
               Control_match.row_nums_1(i) = j - 1; % save row number. meta_data_flows has titles therefore j - 1
               break;
               end
           end          
       end
       
       if(j == size(Model.meta_data_flows,1)) % flow was not found
           % log created flow
           Control_match.log.created_flows{l,1} = i;
           Control_match.log.created_flows{l,2} = Control_match.names_1{i};
           Control_match.log.created_flows{l,3} = Control_match.cats_1(i);
           Control_match.log.created_flows{l,4} = Control_match.units_1{i};
           l = l + 1;
           
           % create new empty flow row in all relevant A submatrices
           Model.matrices.A.mean_values = [Model.matrices.A.mean_values;zeros(1,size(Model.matrices.A.mean_values,2))];
         
           % create new flow in meta_data_flows matrix
           new_metadata = cell(1,size(Model.meta_data_flows,2)); %empty cell line
           new_metadata{1} = Control_match.names_1{i}; % assign name
           new_metadata{2} = Control_match.cats_1(i);  % assign category
           new_metadata{6} = Control_match.units_1{i}; % assign unit
          
           if (strcmp(Control_match.units_1{i},'kg'))  % assign unit category
               new_metadata{5} = 'Mass';
           end
           if (strcmp(Control_match.units_1{i},'Nm3'))
               new_metadata{5} = 'Volumen';
           end
           if (strcmp(Control_match.units_1{i},'MJ'))
               new_metadata{5} = 'Energy';
           end
           
           % assign new metadata 
           Model.meta_data_flows = [Model.meta_data_flows;new_metadata];
           % assign index
           Control_match.row_nums_1(i) = size(Model.matrices.A.mean_values,1);
       end
    end
    
    % check if names, categorie and unitss of flow 1 and 2 equal 
    if( Control_match.row_nums_1(i) == Control_match.row_nums_2(i) )
        if(Control_match.cats_1(i) == Control_match.cats_2(i))
            if isequal(Control_match.units_1(i),Control_match.units_2(i))
            % log if equal
            Control_match.log.skipped_rows{k,1} = i;
            Control_match.log.skipped_rows{k,2} = 'Flow 1 and flow 2 are equal.';
            k = k + 1;
            continue; % skip to next manipulation
            end
        end
    end
    
   
    current_flow_row_1 = Control_match.row_nums_1(i);
    current_flow_row_2 = Control_match.row_nums_2(i);
    
    % calculate new row in A.mean_values according to rule
    Model.matrices.A.mean_values(current_flow_row_1,:) = ...
        Model.matrices.A.mean_values(current_flow_row_1,:) + Model.matrices.A.mean_values(current_flow_row_2,:) * Control_match.conv_factors(i);
 
    % check for different metadata from flow to be deleted and save it
    for j = 1:size(Model.meta_data_flows,2)
        deleted_metadata{i+1,1} = Control_match.names_2{i}; % save name of deleted flow 
        
        if(j == 1 || j == 4 || j == 5 || j == 6 || j == 10 || j == 11 || j == 12 || j == 13 || j == 14 ) % for string format cells
            if(~strcmp(Model.meta_data_flows{current_flow_row_1 + 1,j},Model.meta_data_flows{current_flow_row_2 + 1,j})) % compare metadata field per field
                deleted_metadata{i+1,j} = Model.meta_data_flows{current_flow_row_2+1,j}; % save if not equal
            end
            
        else % for double type format cells
            if(Model.meta_data_flows{current_flow_row_1 + 1,j} ~= Model.meta_data_flows{current_flow_row_2 + 1,j}) % compare metadata field per field
                deleted_metadata{i+1,j} = Model.meta_data_flows{current_flow_row_2+1,j}; % save if not equal
            end
        end
        
    end
    
    % delete row of second flow in all relevant A submatrices
    Model.matrices.A.mean_values(current_flow_row_2,:) = [];
    
    % delete row of second flow in meta data
    Model.meta_data_flows(current_flow_row_2+1,:) = []; % +1 because of header
    
    %% revise main flow data
    for j = 2:size(Model.meta_data_processes,2)
        if (strcmp(Control_match.names_2(i) , Model.meta_data_processes{4,j} )) % find flows with same name
            Model.meta_data_processes(4,j)=Control_match.names_1(i);
            Control_match.log.main_flows{j,1} = Control_match.names_2{i};
        end
    end
    

    
end

if ~isempty(Control_match.log.main_flows{1,1});
Control_match.log.main_flows = Control_match.log.main_flows(~cellfun(@isempty, Control_match.log.main_flows));
Control_match.log.main_flows = unique(Control_match.log.main_flows(~cellfun(@isempty, Control_match.log.main_flows)));
end

%% save in model
Model.Control_match.deleted_metadata=deleted_metadata;
Model.Control_match.log.skipped_rows=Control_match.log.skipped_rows;
Model.Control_match.log.created_flows=Control_match.log.created_flows;
Model.Control_match.log.main_flows=(Control_match.log.main_flows);

end

