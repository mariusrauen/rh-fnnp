%% *SCRIPT TO BUILT THE TECHNOLOGY DATA INPUTS*
% make sure you are in the right folder where also this code is stored 

clear
clc


%% Add paths
pathRevisedModels = ('');


%%
adding_files = struct2cell(dir([pathRevisedModels]));
adding_files = adding_files(1,:)';

%%
for i=3:length(adding_files)
    disp(num2str(i))
    filename=fullfile(pathRevisedModels,adding_files{i});

    % process names
    [~,~,process_adding.meta_data_processes] = xlsread(filename,'Process_meta_data');
    process_adding.meta_data_processes(find(cellfun(@(C)...
        any(isnan(C(:))), process_adding.meta_data_processes)))={[]};
    
    if ~strcmp(process_adding.meta_data_processes{end,1},'EFnomenclature')
        process_adding.meta_data_processes{end+1,1} = 'EFnomenclature';
        process_adding.meta_data_processes(end,2:end) = cellstr(repmat("3.8_2021",1,size(process_adding.meta_data_processes,2)-1));
        
        writecell(process_adding.meta_data_processes,filename,'Sheet','Process_meta_data','WriteMode','overwrite');
    end
    
end        
