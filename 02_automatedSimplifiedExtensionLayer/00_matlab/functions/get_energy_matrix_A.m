function [ Amatrixforallocationfactors , possible ] = get_energy_matrix_A(Model)

%% get heating values

possible = 1;

conversion_factors = Model.meta_data_flows(2:end,9); 

missing_conversion_factors = find(strcmp(conversion_factors,'missing'));

% set missing factors to zero
if ~isempty(missing_conversion_factors)
conversion_factors(missing_conversion_factors) = {0};
end

%% set conversion factors of flows already in MJ to 1

energy_flows = find(strcmp(Model.meta_data_flows(2:end,6),'MJ'));

if ~isempty(energy_flows)
conversion_factors(energy_flows) = {1};    
end


%% Get matrix Amatrixforallocationfactors

energy_vector = cell2mat(conversion_factors);

if size(energy_vector,1) < size(Model.meta_data_flows(2:end,9),1)
   possible = 0;
   error('No energy allocation possible, because not all prices are given or one price is zero.')
   return
end
A = Model.matrices.A.mean_values;
A(A<0) = 0;

energy_matrix = repmat(energy_vector,1,size(A,2));

Amatrixforallocationfactors = A.*energy_matrix;

end