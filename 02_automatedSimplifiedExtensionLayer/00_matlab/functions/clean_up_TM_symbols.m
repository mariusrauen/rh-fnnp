function [Model] = clean_up_TM_symbols(Model)

codedstring_1 = '\u0099'; % corrupted (TM)-symbol
invalid_character_1 = sprintf(strrep(codedstring_1, '\u', '\x'));

codedstring_2 = '\u2122'; % correct (TM)-symbol
invalid_character_2 = sprintf(strrep(codedstring_2, '\u', '\x'));

for i=2:size(Model.meta_data_processes,2)
    if ~isempty(strfind(Model.meta_data_processes{1,i},invalid_character_1))
        Model.meta_data_processes{1,i} = regexprep(Model.meta_data_processes{1,i},invalid_character_1,'');
        Model.meta_data_processes{2,i} = regexprep(Model.meta_data_processes{2,i},invalid_character_1,'');
    end
    if ~isempty(strfind(Model.meta_data_processes{1,i},invalid_character_2))
        Model.meta_data_processes{1,i} = regexprep(Model.meta_data_processes{1,i},invalid_character_2,'');
        Model.meta_data_processes{2,i} = regexprep(Model.meta_data_processes{2,i},invalid_character_2,'');
    end
end

end