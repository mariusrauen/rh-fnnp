function [ Model_wastes ] = make_avoided_burden_steam( Model_wastes )


meta_data =  Model_wastes.meta_data_flows;
A = Model_wastes.matrices.A.mean_values;


row_natural_gas = find(strcmp(meta_data(2:end,1),'NATURAL GAS')...
    & strcmp(meta_data(2:end,6),'MJ')...
    & (cell2mat(meta_data(2:end,2)) == 2));

row_avoided_burden = find(strcmp(meta_data(2:end,1),'NATURAL GAS AVOIDED BURDEN')...
    & strcmp(meta_data(2:end,6),'MJ')...
    & (cell2mat(meta_data(2:end,2)) == 2));

if ~isempty(row_natural_gas)
    
    rows_positive = find(A(row_natural_gas,:)>0);
    
    if ~isempty(rows_positive)
       
        A_new = zeros(1,size(A,2));
        
    end
    
end

end