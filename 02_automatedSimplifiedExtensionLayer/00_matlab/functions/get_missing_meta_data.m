function [missing_meta_data] = get_missing_meta_data(path_own_scenario)

        file_name=fullfile(path_own_scenario,'meta_data_flows.xlsx');
        
        [~,~, missing_meta_data]=xlsread(file_name);
        
         missing_meta_data(find(cellfun(@(C)...
             any(isnan(C(:))), missing_meta_data)))={[]}; 
   
end