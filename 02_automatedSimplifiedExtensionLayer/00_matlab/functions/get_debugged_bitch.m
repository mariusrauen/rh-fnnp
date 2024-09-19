function [flows_opt] = get_debugged_bitch(Model)


A = Model.matrices.A.mean_values;

meta_p = Model.meta_data_processes;

meta_f = Model.meta_data_flows;

process_multi = cell(0,100);

flows_opt = cell(0,100);

counter = 0;

for i = 1:size(A,1)
    
    col = find(A(i,:)>0);
    if length(col)>1
    
        counter = counter +1;
        flows_opt(counter,1) = meta_f(i+1,1);
        flows_opt(counter,2) = meta_f(i+1,11);

        for j= 3:length(col)+2

            flows_opt(counter,j) = meta_p(1,col(j-2)+1);

        end
    end
end

counter = 0;
% 
% for i = 1:size(A,2)
%     
%     row = find(A(:,i)>0);
%     
%     if length(row)>1
%         
%         counter = counter +1;
%         process_multi(counter,1) =  meta_p(1,i+1);
%         process_multi(counter,2) = meta_p(i+1,5);
% 
%         for j= 3:length(row)+2
% 
%             process_multi(counter,j) = meta_f(row(j-2)+1,1);
% 
%         end
%     end
% end

end
