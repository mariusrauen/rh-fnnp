function [ Model_allocated ] = make_allocation(Model, allocation_onoff, allocation_type)
%% make data for calculations

if allocation_onoff
    possible = 0;
    
    if strcmp(allocation_type,'mass')
        
        [ Amatrixforallocationfactors, possible ] = get_mass_matrix_A(Model);
                
    elseif strcmp(allocation_type,'energy')
        
        [ Amatrixforallocationfactors, possible ] = get_energy_matrix_A(Model); 
    
    elseif strcmp(allocation_type,'price')
        
        [ Amatrixforallocationfactors, possible ] = get_price_matrix_A(Model);
    
    end
    
    if ~possible % only if all prices are given are non zero
        return
    end
    
    [ Model_allocated ] = perform_allocation_RMe(Model, Amatrixforallocationfactors);
    

end

end