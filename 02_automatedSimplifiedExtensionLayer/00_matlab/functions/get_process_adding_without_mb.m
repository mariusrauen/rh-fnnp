function [ add_manual_processes ] = get_process_adding_without_mb(path_alignments)

% get availible data files in scenario folder
    av_files=extractfield(dir(path_alignments),...
        'name')';
    
    if ismember('manual_adding_processes_no_mass_balance',av_files)
        
        adding_files=extractfield(dir([path_alignments,'\manual_adding_processes_no_mass_balance']),...
        'name')';
    
        counter=0;
        for i=3:length(adding_files)
            
            filename=fullfile([path_alignments,'\manual_adding_processes_no_mass_balance'],adding_files{i});
       
            % process names
            [~,~,process_adding.meta_data_processes] = xlsread(filename,1);
            process_adding.meta_data_processes(find(cellfun(@(C)...
                any(isnan(C(:))), process_adding.meta_data_processes)))={[]};

            % A
            [process_adding.A.mean_values,~,~] = xlsread(filename,2,'D2:ZZ3000');
            [types,process_adding.meta_data_flows,~] = xlsread(filename,2,'A2:C3000');

            if ~isempty(process_adding.A.mean_values)
            process_adding.A.mean_values(isnan(process_adding.A.mean_values))=0;
            process_adding.A.mean_values(isempty(process_adding.A.mean_values))=0;
            end

            if ~isempty(types)
            process_adding.meta_data_flows(:,2)=num2cell(types);
            end

            % B
            [process_adding.B.mean_values,~,~] = xlsread(filename,3,'D2:ZZ3000');
            [~,process_adding.meta_data_elementary_flows,~] = xlsread(filename,3,'A2:C3000');

            if ~isempty(process_adding.B.mean_values)
            process_adding.B.mean_values(isnan(process_adding.B.mean_values))=0;
            process_adding.B.mean_values(isempty(process_adding.B.mean_values))=0;
            end

            % F
            [process_adding.F.mean_values,~,~] = xlsread(filename,4,'C2:ZZ3000');
            [~,process_adding.meta_data_factor_requirements,~] = xlsread(filename,4,'A2:C3000');

            if ~isempty(process_adding.F.mean_values)
            process_adding.F.mean_values(isnan(process_adding.F.mean_values))=0;
            process_adding.F.mean_values(isempty(process_adding.F.mean_values))=0;
            end
            
            counter=counter+1;
            add_manual_processes(counter)=process_adding;

        end
        
    else
        
        display('No process adding was performed.');
        
        process_adding.A.mean_values=[];
        
    end

end