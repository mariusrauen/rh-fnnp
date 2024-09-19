
function [results]=get_missing_data(all_flows_meta_data, data)

for i=2:(size(all_flows_meta_data,1))
    
    % get row number in missing meta data filled table 
    
    row_flow=find(strcmp...
    (data(:,1),all_flows_meta_data(i,1))); % find row
    
    if isempty(row_flow) % next if no match is found
       continue
    end

    for k=[row_flow'] % go through all found rows
        
        if isequal(all_flows_meta_data(i,6),data(k,6))&&... %% unit is equal
                isequal(all_flows_meta_data(i,2),data(k,2)) %% type is equal

           all_flows_meta_data(i,:)=data(k,:); % overwrite flow until price
%            all_flows_meta_data(i,1:6)=data(k,1:6); % overwrite flow until price
%            all_flows_meta_data(i,8:10)=data(k,8:10); % overwrite LLV, HHV, formular
%            all_flows_meta_data(i,13:14)=data(k,13:14); % overwrite comment and smiles
%            
%            if isempty(all_flows_meta_data{i,7})
%                all_flows_meta_data(i,7)=data(k,7); % overwrite price if is empty
%            end
        end
        
    end
    
end

results=all_flows_meta_data;
