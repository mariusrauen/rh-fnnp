function [match] = get_match_table(path_Eco_list,matchingTableName)

[~,~,match] = xlsread(fullfile(path_Eco_list,matchingTableName));

    match(find(cellfun(@(C)...
        any(isnan(C(:))), match)))={[]};  

end