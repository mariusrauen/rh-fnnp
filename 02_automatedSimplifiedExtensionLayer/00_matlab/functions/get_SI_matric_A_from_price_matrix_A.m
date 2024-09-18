function [ Model ] = get_SI_matric_A_from_price_matrix_A(Model);

price_vector = cell2mat(Model.meta_data_flows(2:end,7));

price_vector_reverse = ones(size(price_vector))./price_vector;

price_matrix_reverse = repmat(price_vector_reverse,1,size(Model.matrices.A.mean_values,2));

Model.matrices.A.mean_values = Model.matrices.A.mean_values.*price_matrix_reverse;

end