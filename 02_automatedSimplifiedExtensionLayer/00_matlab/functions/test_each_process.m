function [process_errors,impact_results_process,s,y] = test_each_process(Model)

A_ineq = -Model.matrices.A.mean_values;
objective = Model.matrices.objective.mean_values;

columns_supply = size(A_ineq,2);
row_supply = size(A_ineq,1);

columns_numbers = columns_supply+1:1:(columns_supply+row_supply);

products_supply = Model.meta_data_flows(2:end,1);

add_supply = -eye(row_supply,row_supply);
objective_supply = ones(1,row_supply)*1e10;

A_ineq_supply = [A_ineq add_supply];
objective_supply = [objective objective_supply];


b_ineq = zeros(size(A_ineq,1),1);
b_ineq_supply = zeros(size(A_ineq_supply,1),1);

ub = ones(size(A_ineq,2),1)*1e10;
lb = zeros(size(A_ineq,2),1);

ub_supply = ones(size(A_ineq_supply,2),1)*1e10;
lb_supply = zeros(size(A_ineq_supply,2),1);

% %% scaling if needed
% if cond(A_ineq)>1e2
%     
% maximum = max(A_ineq);
% maximum(maximum == 0) = 1;
% 
% maximum_supply = max(A_ineq_supply);
% maximum_supply(maximum_supply == 0) = 1;
% 
% objective = objective./maximum;
% objective_supply = objective_supply./maximum_supply;
% 
% scale_A = repmat(maximum,size(A_ineq,1),1);
% scale_A_supply = repmat(maximum_supply,size(A_ineq_supply,1),1);
% 
% A_ineq = A_ineq./scale_A;
% A_ineq_supply = A_ineq_supply./scale_A_supply;
% 
% end
% 
% %% find waste flows
% 
% types = cell2mat(Model.meta_data_flows(2:end,2))==3;
% 
% rows = find(types);
% 
% if ~isempty(rows)
%     
%     A_ineq(rows,:) = [];
%     b_ineq(rows,:) = [];
%     
%     A_ineq_supply(rows,:) = [];
%     b_ineq_supply(rows,:) = [];
%     
% end

%% optimization
options = optimoptions('linprog',...
            'Algorithm','dual-simplex',...
            'OptimalityTolerance',1e-10,...
            'MaxIterations',50000,...
            'Display','off');

counter = 0;
counter_1 = 0;
process_errors = {[]};

for i = 1:length(lb)
        counter_1=counter_1 + 1;
        lb(i) = 1;
        lb_supply(i) = 1;
        
        [scaling_vector,impact,exit_flag]=...
            linprog(objective,...
                    A_ineq,...
                    b_ineq,...
                    [],...
                    [],...
                    lb,...
                    ub,...
                    options);
                
         lb(i) = 0;
         
         impact_results_process{counter_1,1} = impact;       
         impact_results_process{counter_1,2} = Model.meta_data_processes(1,i+1);
         impact_results_process{counter_1,3} = Model.meta_data_processes(4,i+1);
    
    if ~isempty(scaling_vector)     
         s(:,counter_1) = scaling_vector;
         y(:,counter_1) = -A_ineq * scaling_vector;
    end
    
    if isempty(scaling_vector)
                
        
        s(:,counter_1) = zeros(size(A_ineq,2),1);
        y(:,counter_1) = zeros(size(A_ineq,1),1);
        disp('Process unsuccesfull: ')
        disp(Model.meta_data_processes(1,i+1));
        counter = counter + 1;
        process_errors(counter,1) = Model.meta_data_processes(1,i+1);
        
        [scaling_vector_supply,~,~]=...
            linprog(objective_supply,...
                    A_ineq_supply,...
                    b_ineq_supply,...
                    [],...
                    [],...
                    lb_supply,...
                    ub_supply,...
                    options);
        
        used_supplies = find(scaling_vector_supply(columns_numbers)>0);
        
        flows_not_producable = products_supply(used_supplies);
        process_errors{counter,2} = flows_not_producable;        
        disp('Due to flow: ')
        disp(flows_not_producable);
        lb_supply(i) = 0;
        
        continue;
        
    end
        
end

%% check
if isempty(process_errors)
   disp('all processes usable');
end

end