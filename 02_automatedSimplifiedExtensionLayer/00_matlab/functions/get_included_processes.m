function [ included_processes ] = get_included_processes(path_own_scenario)

av_files=extractfield(dir(path_own_scenario),...
        'name')';

included_processes = {'none'};
    
%%  
   if ismember('included_processes',av_files)
       
       adding_files=extractfield(dir([path_own_scenario,'\included_processes']),...
           'name')';
       
       counter=0;
       
       for i = 3:length(adding_files)
           
           filename=fullfile([path_own_scenario,'\included_processes'],adding_files{i});
                       
           [~,~, included_processes_add] = xlsread(filename);
           
           included_processes_add(find(cellfun(@(C)...
                any(isnan(C(:))), included_processes_add)))={};
            
           included_processes = [included_processes ; included_processes_add];
       end
         
   end
   
 included_processes(1) = [];
end