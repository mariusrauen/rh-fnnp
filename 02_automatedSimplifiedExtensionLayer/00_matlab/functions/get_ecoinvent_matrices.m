function [B_eco_mean, A_eco_mean, F_eco_mean, eco_process_meta] = get_ecoinvent_matrices(Model,Ecoinvent_data,match,regions)

%% get mean values from ecoinvent list

counter = 0;
B_eco_mean = [];
A_eco_mean = zeros(size(Model.matrices.A.mean_values,1),size(match,1)-1);
F_eco_mean = eye(size(match,1)-1);

match{1,8} = 'Ecoinvent_process_column'; 
match{1,9} = 'region'; 

%% find columns of Ecoinvent processes
for i = 2 : size(match,1)
    found = 0;
    for j = 1 : size(Ecoinvent_data.col_names,1)
        for k = 1:length(regions)
            
            if  strcmp(match(i,4),Ecoinvent_data.col_names(j,2))&&...
                strcmp(match(i,5),Ecoinvent_data.col_names(j,1))&&...
                strcmp(regions(k),Ecoinvent_data.col_names(j,3))&&...
                ~found

                match{i,8} = j;
                match{i,9} = regions(k);
                
                found = 1;
            end
        end
    end
    
    if ~found
        display('WARNING, the following flow was not found:')
        display(match(i,1))
        display(match(i,4))
        display(match(i,5))
        match{i,8} = {[]};
    end
    
end

%% make B_eco_mean, A_eco_mean, F_eco_mean
counter_delete_columns = 0;
for i = 2 : size(match,1)
    
    counter = counter + 1;
    
    % normal ecoinvent processes
    if match{i,7} >= 0
        % B
        
        B_eco_mean(:,counter) = Ecoinvent_data.mean_values(:,match{i,8}) * match{i,7};
        
        % A
        row_flow = find(strcmp...
            (Model.meta_data_flows(:,1),match{i,1})); % find row in meta data
    
        if isempty(row_flow) % next if no match is found
        continue
        end
        
        for k = [row_flow'] % go through all found rows
        
        if isequal(Model.meta_data_flows{k,2},match{i,2})&&... %% unit is equal
                isequal(Model.meta_data_flows{k,6},match{i,3}) %% type is equal
           
            A_eco_mean(k-1,i-1) = 1; % overwrite line in A with one. -1 because row are changed by one if rows from flow meta data is used
            
            price = Model.meta_data_flows{k,7};
            
            if isempty(price)
                price=[0];
            end
                           
            F_eco_mean(i-1,:) = F_eco_mean(i-1,:)*price;
            
        end
        
        end
    
    % avoided burden
    
    elseif match{i,7} < 0
        % B
        
        B_eco_mean(:,counter) = Ecoinvent_data.mean_values(:,match{i,8}) * -match{i,7};
        
        % A
        row_flow = find(strcmp...
            (Model.meta_data_flows(:,1),match{i,1})); % find row in meta data
    
        if isempty(row_flow) % next if no match is found
        continue
        end
        
        for k = [row_flow'] % go through all found rows
        
        if isequal(Model.meta_data_flows{k,2},match{i,2})&&... %% unit is equal
                isequal(Model.meta_data_flows{k,6},match{i,3}) %% type is equal
           
            A_eco_mean(k-1,i-1) = -1; % overwrite line in A with one. -1 because row are changed by one if rows from flow meta data is used
            
            price = Model.meta_data_flows{k,7};
            
            if isempty(price)
                price=[0];
            end
                           
            F_eco_mean(i-1,:) = F_eco_mean(i-1,:) * -price;
            
        end
        
        end
    
    end
    
end


% for i = 1:length(list)
%     
%     counter=counter+1;
% 
%     elementary_flows = get_elementary(list(i).file,allocation_ecoinvent);
%         
%     B_eco_mean(:,counter)= extractfield(elementary_flows,'amount') * list(i).con_fac;
%     
%     % A_mean & F_mean
% 
%     row_flow=find(strcmp...
%         (Model.meta_data_flows(:,1),list(i).name)); % find row in meta data
%     
%     if isempty(row_flow) % next if no match is found
%        continue
%     end
%     
%     for k=[row_flow'] % go through all found rows
%         
%         if isequal(Model.meta_data_flows{k,6},list(i).unit)&&... %% unit is equal
%                 isequal(Model.meta_data_flows{k,2},list(i).type) %% type is equal
%            
%             A_eco_mean(k-1,i) = 1; % overwrite line in A with one. -1 because row are changed by one if rows from flow meta data is used
%             price=Model.meta_data_flows{k,7};
%             
%             if isempty(price)
%                 price=[0];
%             end
%                            
%             F_eco_mean(i,:)=F_eco_mean(i,:)*price;
%             
%         end
%     end
% end
% 
% clear elementary_flows counter i
% 
% 
%% make meta data ecoinvent processes
eco_process_meta(1,:) = match(2:end,4);
eco_process_meta(2,:) = {[]};
eco_process_meta(3,:) = {[]};
eco_process_meta(4,:) = match(2:end,1);
eco_process_meta(5,:) = match(2:end,9);
eco_process_meta(6,:) = {[]};
eco_process_meta(7,:) = {[]};
eco_process_meta(8,:) = {[]};
eco_process_meta(9,:) = {[]};
eco_process_meta(10,:) = {4}; %%%% ONLY ADDED IN THIS SPECIFIC CASE

% delete columns with only zeros

delete_columns = ~any(A_eco_mean,1);

A_eco_mean(:,delete_columns) = [];
B_eco_mean(:,delete_columns) = [];
F_eco_mean(:,delete_columns) = [];

eco_process_meta(:,delete_columns) = [];

delete_rows_F = ~any(F_eco_mean,2);
F_eco_mean(delete_rows_F,:) = [];

end 