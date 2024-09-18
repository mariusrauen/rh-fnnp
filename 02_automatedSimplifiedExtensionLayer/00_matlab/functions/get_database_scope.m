function [database_scope, IHS_processes_main_flows] = get_database_scope(path_data_base_scope)

   
database_scope={[]};
IHS_processes_main_flows ={[]};
      
        file_name=fullfile(path_data_base_scope,'\database_scope.xlsx');
        
        [~,~, database_scope] = xlsread(file_name);
        
         database_scope(find(cellfun(@(C)...
             any(isnan(C(:))), database_scope)))={[]};  
         
         IHS_processes_main_flows = readtable(file_name,'Sheet','Processes IHS');

end