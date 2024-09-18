function [meta_data_to_update]=get_meta_data_to_update(data)

counter=0;

for i=2:size(data,1);

    if isempty(data{i,4})||...    % CAS
       isequal(data{i,7},1)||... % price
       isequal(data{i,7},0)||...
       isempty(data{i,7})||...
       isempty(data{i,9})||...  % HHV
       isempty(data{i,10})||... % formular
       isempty(data{i,15})% molecular mass
       
       counter=counter+1;
       
       meta_data_to_update(counter+1,:)=data(i,:);
       
    else
        continue
    end
    
    
end

       meta_data_to_update(1,:) = data(1,:);
       
       if counter > 0
           
           disp('FLOW REVISION IS NEEDED. CHECK MISSING META DATA.');
           
       end
       
end