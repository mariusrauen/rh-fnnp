function [ Amatrixforallocationfactors , possible ] = get_mass_matrix_A(Model)

possible=1;

%% get heating values
conversion_factors = Model.meta_data_flows(2:end,9); 

missing_conversion_factors = find(strcmp(conversion_factors,'missing'));

% set missing factors to zero
if ~isempty(missing_conversion_factors)
conversion_factors(missing_conversion_factors) = {0};
end

%% set conversion factors of flows already in kg to 1

mass_flows = find(strcmp(Model.meta_data_flows(2:end,6),'kg'));

if ~isempty(mass_flows)
conversion_factors(mass_flows) = {1};    
end

%% Get matrix Amatrixforallocationfactors

mass_vector = cell2mat(conversion_factors);

if size(mass_vector,1) < size(Model.meta_data_flows(2:end,9),1)
   possible=0;
   display('No mass allocation possible, because not all prices are given or one price is zero.')
   return
end

mass_matrix = repmat(mass_vector,1,size(Model.matrices.A.mean_values,2));

Amatrixforallocationfactors = Model.matrices.A.mean_values.*mass_matrix;

end