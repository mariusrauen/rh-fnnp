function [ Model ] = get_flow_splitting_Model( Model , Control_splitting )
%FLOW_SPLITTING_FUNCTION Flow splitting for chemistry database building


%% DISCLAIMER
% make sure to close the control sheet before executing the function!

%% MANIPULATE MATRICES

k = 1; % write index for skipped rows
l = 1; % write index for created flows

% execute the manipulations
for i = 2:length(Control_splitting.actual.processes.names)  % index of manipulation row: i, skip BEGINNING ROW
    flag_process = false; % reset actual process not found flag 
    flag_flow = false; % reset actual flow not found flag 
    
    % check if manipulation is empty
    if (strcmp(Control_splitting.actual.processes.names{i,1},''))
        continue; % and skip this manipulation        
    end
    
    % check if END is reached
    if (strcmp(Control_splitting.actual.processes.names{i,1},'END'))
        continue; % and skip this manipulation        
    end
    
    
    % check if mass percentages sum up to 1
    % reset mass percentages
    x_1 = 0;
    x_2 = 0;
    x_3 = 0;
    x_4 = 0;
    
    if(  ~isnan(  Control_splitting.target.flows_1.x(i)  )   )   % check if mass percentage exists (is not a NaN) 
        x_1 = Control_splitting.target.flows_1.x(i);        
    end
    
    if(  ~isnan(  Control_splitting.target.flows_2.x(i)  )   )
        x_2 = Control_splitting.target.flows_2.x(i);        
    end
    
    if(  ~isnan(  Control_splitting.target.flows_3.x(i)  )   )
        x_3 = Control_splitting.target.flows_3.x(i);        
    end
    
    if(  ~isnan(  Control_splitting.target.flows_4.x(i)  )    )
        x_4 = Control_splitting.target.flows_4.x(i);       
    end
    
    x_sum = x_1 + x_2 + x_3 + x_4;
    
%     if(x_sum ~= 1)
%         Control_splitting.log.skipped_rows{k,1} = i;
%         Control_splitting.log.skipped_rows{k,2} = 'Sum of mass percentages does not equal 1.';
%         k = k + 1;
%         continue; % skip row (go to next manipulation)
%     end
    
    % find column number of actual process
    for j = 2:size(Model.meta_data_processes,2)
        
       if(strcmp(Control_splitting.actual.processes.names(i) , Model.meta_data_processes{1,j} )) % compare names
           Control_splitting.actual.processes.col_nums(i) = j - 1; % save if found. meta_data has titles, therefore j - 1
           break; % stop searching         
       end
       
       if(j == size(Model.meta_data_processes,2)) % process was not found
           flag_process = true;
           break; % stop searching
       end
       
    end
    
    % log if row was skipped because of non-existance
    if flag_process == true;
        Control_splitting.log.skipped_rows{k,1} = i;
        Control_splitting.log.skipped_rows{k,2} = 'Actual process was not found.';
        k = k + 1;
        continue; % skip row (go to next manipulation)
    end
    
    % find row number of actual flow
    for j = 2:size(Model.meta_data_flows,1)
        
       if(strcmp(Control_splitting.actual.flows.names(i) , Model.meta_data_flows{j,1} )) % compare name
           if (Control_splitting.actual.flows.cats(i) == Model.meta_data_flows{j,2})
                Control_splitting.actual.flows.row_nums(i) = j - 1; % save if found. meta_data has titles, therefore j - 1
                break; % stop searching
           end
       end
       
       if(j == size(Model.meta_data_flows,1)) % flow was not found
           flag_flow = true;
           break; % stop searching
       end
       
    end
    
    % log if row was skipped because of non-existance
    if flag_flow == true;
        Control_splitting.log.skipped_rows{k,1} = i;
        Control_splitting.log.skipped_rows{k,2} = 'Actual flow was not found.';
        k = k + 1;
        continue; % skip row (go to next manipulation)
    end
    
    
    % find row number of all target flows
    for w = 1:4
        flow_w = genvarname(strcat('flows_',num2str(w)));
        
        % check if target flow is used
        if(  strcmp(  Control_splitting.target.(flow_w).names{i} , ''  )  )
           continue;  
        end

        % search for flow in meta data sheet
        for j = 2:size(Model.meta_data_flows,1)
           if(strcmp(Control_splitting.target.(flow_w).names{i} , Model.meta_data_flows{j,1} ))
               if (Control_splitting.target.(flow_w).cats(i) == Model.meta_data_flows{j,2})
                   Control_splitting.target.(flow_w).row_nums(i) = j - 1; % save row number. meta_data_flows has titles therefore j - 1
                   break;
               end          
           end

           if(j == size(Model.meta_data_flows,1)) % flow was not found
               % log created flow
               Control_splitting.log.created_flows{l,1} = i;
               Control_splitting.log.created_flows{l,2} = Control_splitting.target.(flow_w).names{i};
               Control_splitting.log.created_flows{l,3} = Control_splitting.target.(flow_w).cats(i);
               Control_splitting.log.created_flows{l,4} = Control_splitting.target.(flow_w).units{i};
               l = l + 1;

               % create new empty flow row in all relevant A submatrices
               Model.matrices.A.mean_values = [Model.matrices.A.mean_values;zeros(1,size(Model.matrices.A.mean_values,2))];

               % create new flow in meta_data_flows matrix
               new_metadata = cell(1,size(Model.meta_data_flows,2)); %empty cell line
               new_metadata{1} = Control_splitting.target.(flow_w).names{i}; % assign name
               new_metadata{2} = Control_splitting.target.(flow_w).cats(i);  % assign category
               new_metadata{6} = Control_splitting.target.(flow_w).units{i}; % assign unit
               if (strcmp(Control_splitting.target.(flow_w).units{i},'kg'))  % assign unit category
                   new_metadata{5} = 'Mass';
               end
               if (strcmp(Control_splitting.target.(flow_w).units{i},'Nm3'))
                   new_metadata{5} = 'Volumen';
               end
               if (strcmp(Control_splitting.target.(flow_w).units{i},'MJ'))
                   new_metadata{5} = 'Energy';
               end

               % assign new metadata 
               Model.meta_data_flows = [Model.meta_data_flows;new_metadata];
               % assign index
               Control_splitting.target.(flow_w).row_nums(i) = size(Model.matrices.A.mean_values,1);
           end
        end 
    end
          
    
    % get actual values in A, calculate and assign new values 
    Control_splitting.actual.values(i) = Model.matrices.A.mean_values(Control_splitting.actual.flows.row_nums(i),Control_splitting.actual.processes.col_nums(i));
    v = Control_splitting.actual.values(i);
    
    Model.matrices.A.mean_values(Control_splitting.actual.flows.row_nums(i),Control_splitting.actual.processes.col_nums(i)) = 0; % delete actual flow in process
    
    if(isfield(Control_splitting.target.flows_1,'row_nums'))             % row nums field exists
        if(i <= length(Control_splitting.target.flows_1.row_nums))    % target flow row number exists
            if(  ~isnan(  Control_splitting.target.flows_1.x(i)  )   )   % check if mass percentage exists (is not a NaN) 
                a = Model.matrices.A.mean_values(Control_splitting.target.flows_1.row_nums(i),Control_splitting.actual.processes.col_nums(i));
                x_1 = Control_splitting.target.flows_1.x(i);
                % calculation rule
                a = a + v * x_1;
                Model.matrices.A.mean_values(Control_splitting.target.flows_1.row_nums(i),Control_splitting.actual.processes.col_nums(i)) = a;
            end
        end
    end
   
    
    if(isfield(Control_splitting.target.flows_2,'row_nums'))
        if(i <= length(Control_splitting.target.flows_2.row_nums))
            if(  ~isnan(  Control_splitting.target.flows_2.x(i)  )   )  
                b = Model.matrices.A.mean_values(Control_splitting.target.flows_2.row_nums(i),Control_splitting.actual.processes.col_nums(i));
                x_2 = Control_splitting.target.flows_2.x(i);
                b = b + v * x_2;
                Model.matrices.A.mean_values(Control_splitting.target.flows_2.row_nums(i),Control_splitting.actual.processes.col_nums(i)) = b;
            end
        end
    end
    
    if(isfield(Control_splitting.target.flows_3,'row_nums'))
        if(i <= length(Control_splitting.target.flows_3.row_nums))
            if(  ~isnan(  Control_splitting.target.flows_3.x(i)  )   )
                c = Model.matrices.A.mean_values(Control_splitting.target.flows_3.row_nums(i),Control_splitting.actual.processes.col_nums(i));
                x_3 = Control_splitting.target.flows_3.x(i);
                c = c + v * x_3;
                Model.matrices.A.mean_values(Control_splitting.target.flows_3.row_nums(i),Control_splitting.actual.processes.col_nums(i)) = c;
            end
        end
    end
    
    if(isfield(Control_splitting.target.flows_4,'row_nums'))
        if(i <= length(Control_splitting.target.flows_4.row_nums))
            if(  ~isnan(  Control_splitting.target.flows_4.x(i)  )   )
                d = Model.matrices.A.mean_values(Control_splitting.target.flows_4.row_nums(i),Control_splitting.actual.processes.col_nums(i));
                x_4 = Control_splitting.target.flows_4.x(i);
                d = d + v * x_4;
                Model.matrices.A.mean_values(Control_splitting.target.flows_4.row_nums(i),Control_splitting.actual.processes.col_nums(i)) = d;
            end
        end    
    end 
    
end

%% save to model
Model.Control_splitting=Control_splitting;

end

