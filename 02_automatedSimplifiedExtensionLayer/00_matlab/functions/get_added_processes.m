function [ Model ] = get_added_processes( Model , process_adding )

%PROCESS_ADDING_FUNCTION Flow adding for chemistry database building

%% MANIPULATE MATRICES
a = size(process_adding.A.mean_values,2);
b = size(process_adding.B.mean_values,2);
f = size(process_adding.F.mean_values,2);
pd = size(process_adding.meta_data_processes,2)-1;
% catch problems
if isempty(process_adding.A.mean_values) ||...
   isequal(size(process_adding.meta_data_processes,2),1) || ...
   and(~isequal(a,b),b>0) || ...
   and(~isequal(a,f),f>0) || ...
   and(~isequal(a,pd),pd>0)
   disp('Process adding is cancelled, because provided data is either empty or sizes of matrices do not fit.');
   disp('Check data given. If no processes should be added, ignore this message.');
   Model=Model;
   return
end

% find row numbers of flows in A

for i = 1:size(process_adding.A.mean_values ,1)  
    for j = 1:size(Model.meta_data_flows(2:end,:),1)
       if strcmp(process_adding.meta_data_flows{i,1} , Model.meta_data_flows{j+1,1}) && ...        % compare name
          isequal(process_adding.meta_data_flows{i,2} , Model.meta_data_flows{j+1,2}) && ...       
          strcmp(process_adding.meta_data_flows{i,3},   Model.meta_data_flows{j+1,6})
      
                   process_adding.A.flows.row_indices(i) = j; % save if found. meta_data has titles, therefore j - 1
                   break; % stop searching

       end

       if(j == size(Model.meta_data_flows(2:end,:),1)) % flow was not found
           disp('The following non-existent flow in supplementary matrix A was created:');
           disp([process_adding.meta_data_flows(i,1)]);
           % create missing flow meta data
           if strcmp(process_adding.meta_data_flows(i,3) , 'kg')
               unit_cat = 'Mass';
           end
           if strcmp(process_adding.meta_data_flows(i,3) , 'Nm3')
               unit_cat = 'Volumen';
           end
           if strcmp(process_adding.meta_data_flows(i,3) , 'MJ')
               unit_cat = 'Energy';
           end  
           if strcmp(process_adding.meta_data_flows(i,3) , 'tkm')
               unit_cat = 'ton-kilometer';
           end     

           new_line_meta_data_flows = [process_adding.meta_data_flows(i,1) process_adding.meta_data_flows(i,2)  {char(1)} {char(1)} ...
               unit_cat process_adding.meta_data_flows(i,3) {[]} {[]} {[]} {[]} {[]} {[]} {[]} {[]} {[]}];
           Model.meta_data_flows = [Model.meta_data_flows; new_line_meta_data_flows];
           
           %create new flow line in A
           Model.matrices.A.mean_values = [Model.matrices.A.mean_values; zeros(1,size(Model.matrices.A.mean_values , 2))];
           process_adding.A.flows.row_indices(i) = size(Model.matrices.A.mean_values, 1); % save index of new flow
       end   
    end
end

% find row numbers of flows in B
for i = 1:size(process_adding.B.mean_values ,1)    
    for j = 1:size(Model.meta_data_elementary_flows(2:end,:),1)      
       if strcmp(process_adding.meta_data_elementary_flows{i,1} , Model.meta_data_elementary_flows{j+1,1} ) && ...                   % compare name
          strcmp(process_adding.meta_data_elementary_flows{i,2} , Model.meta_data_elementary_flows{j+1,2} ) && ...        % compare compartment
          strcmp(process_adding.meta_data_elementary_flows{i,3} , Model.meta_data_elementary_flows{j+1,3} ) % compare subcompartment
                    
        process_adding.B.flows.row_indices(i) = j;         % save if found. meta_data has titles, therefore j - 1
        break; % stop searching
        
       end

       if(j == size(Model.meta_data_elementary_flows(2:end,:),1)) % flow was not found
           disp('The following non-existent flow in supplementary matrix B was created:');
           disp([process_adding.meta_data_elementary_flows(i,1)]);
           % create missing flow meta data
           new_line_meta_data_elementary_flows = [process_adding.meta_data_elementary_flows(i,1) process_adding.meta_data_elementary_flows(i,2) ...
               process_adding.meta_data_elementary_flows(i,3) {char(1)} {''} {''}];
           Model.meta_data_elementary_flows = [Model.meta_data_flows; new_line_meta_data_elementary_flows];
           
           %create new flow line in B
           Model.matrices.B.mean_values = [Model.matrices.B.mean_values; zeros(1,size(Model.matrices.B.mean_values , 2))];
           process_adding.B.flows.row_indices(i) = size(Model.matrices.B.mean_values, 1); % save index of new flow
       end
    end  
end


% find row numbers of flows in F
for i = 1:size(process_adding.F.mean_values ,1)    
    for j = 1:size(Model.meta_data_factor_requirements(2:end,:),1)       
       if strcmp(process_adding.meta_data_factor_requirements{i,1} , Model.meta_data_factor_requirements(j+1,1)) && ...
          strcmp(process_adding.meta_data_factor_requirements{i,2} , Model.meta_data_factor_requirements(j+1,2)) 
            process_adding.F.flows.row_indices(i) = j ; % save if found. 
            break; % stop searching
       end
       
       if(j == size(Model.meta_data_factor_requirements(2:end,:),1)) % flow was not found
           disp('The following non-existent factor requirement in supplementary matrix F was created:');
           disp([process_adding.meta_data_factor_requirements(i,1)]);
           % create missing flow meta data

           new_line_meta_data_factor_requirements = [process_adding.meta_data_factor_requirements(i,1) process_adding.meta_data_factor_requirements(i,2),{''}];
           Model.meta_data_factor_requirements = [Model.meta_data_factor_requirements; new_line_meta_data_factor_requirements];

           %create new flow line in F
           Model.matrices.F.mean_values = [Model.matrices.F.mean_values; zeros(1,size(Model.matrices.F.mean_values , 2))];
           process_adding.F.flows.row_indices(i) = size(Model.matrices.F.mean_values, 1); % save index of new flow
       end       


    end    
end
clear i j

% this part is only executed if all flows in all matrices exist.
for i = 1:size(process_adding.meta_data_processes(:,2:end),2) % i = Index of process

    % check if process is empty in A, B and F
    if (process_adding.A.mean_values(:,i) ==  0)
        if (process_adding.B.mean_values(:,i) ==  0)
            if(process_adding.F.mean_values(:,i) ==  0)
                continue; % and skip this manipulation 
            end
        end
    end

    % create new column in all relevant A matrices
    Model.matrices.A.mean_values = ...
        [Model.matrices.A.mean_values zeros(size(Model.matrices.A.mean_values,1) , 1)];
%     Model.matrices.A.std_dev     = [Model.matrices.A.std_dev zeros(size(Model.matrices.A.std_dev,1) , 1)];
%     Model.matrices.A.distr_type  = [Model.matrices.A.distr_type zeros(size(Model.matrices.A.distr_type,1) , 1)];
%     Model.matrices.A.col_names   = [Model.matrices.A.col_names process_adding.processes.names(i)];

    % create new column in all relevant B matrices
    Model.matrices.B.mean_values = ...
        [Model.matrices.B.mean_values zeros(size(Model.matrices.B.mean_values,1) , 1)];
%     Model.matrices.B.std_dev     = [Model.matrices.B.std_dev zeros(size(Model.matrices.B.std_dev,1) , 1)];
%     Model.matrices.B.distr_type  = [Model.matrices.B.distr_type zeros(size(Model.matrices.B.distr_type,1) , 1)];
%     Model.matrices.B.col_names   = [Model.matrices.B.col_names process_adding.processes.names(i)];
    
    % create new column in all relevant F matrices
    Model.matrices.F.mean_values = ...
        [Model.matrices.F.mean_values zeros(size(Model.matrices.F.mean_values,1) , 1)];
%     Model.matrices.F.std_dev     = [Model.matrices.F.std_dev zeros(size(Model.matrices.F.std_dev,1) , 1)];
%     Model.matrices.F.distr_type  = [Model.matrices.F.distr_type zeros(size(Model.matrices.F.distr_type,1) , 1)];
%     Model.matrices.F.col_names   = [Model.matrices.F.col_names process_adding.processes.names(i)];   

    % write meta data of added process
    Model.meta_data_processes(:, end+1) = process_adding.meta_data_processes(:,i+1);

    % write data in A.mean_values
    for j = 1:size(process_adding.A.mean_values,1)
        if (process_adding.A.mean_values(j,i) == 0)
           continue; % skip empty entries 
        end
        % write data
        Model.matrices.A.mean_values(...
            process_adding.A.flows.row_indices(j),end)...
            = process_adding.A.mean_values(j,i);
    end
    
    % write data in B.mean_values
    for j = 1:size(process_adding.B.mean_values,1)
        if (process_adding.B.mean_values(j,i) == 0)
           continue; % skip empty entries 
        end
        % write data
        Model.matrices.B.mean_values(...
            process_adding.B.flows.row_indices(j),end)...
            = process_adding.B.mean_values(j,i);
    end
    
    % write data in F.mean_values
    for j = 1:size(process_adding.F.mean_values,1)
        if (process_adding.F.mean_values(j,i) == 0)
           continue; % skip empty entries 
        end
        % write data
        Model.matrices.F.mean_values(...
            process_adding.F.flows.row_indices(j),end)...
            = process_adding.F.mean_values(j,i);
    end
    

    
end

end

