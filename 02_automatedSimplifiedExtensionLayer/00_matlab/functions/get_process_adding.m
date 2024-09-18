function [ add_manual_processes,adding_files ] = get_process_adding(path_alignments)

adding_files = struct2cell(dir([path_alignments]));
adding_files = adding_files(1,:)';



        counter=0;
        for i=3:length(adding_files)
            disp(adding_files{i})
            filename=fullfile(path_alignments,adding_files{i});
       
            % process names
            [~,~,process_adding.meta_data_processes] = xlsread(filename,'Process_meta_data');
            process_adding.meta_data_processes(find(cellfun(@(C)...
                any(isnan(C(:))), process_adding.meta_data_processes)))={[]};

            % A
            [process_adding.A.mean_values,~,~] = xlsread(filename,'SUMMARY A','D2:IA200');
            [types,process_adding.meta_data_flows,~] = xlsread(filename,'SUMMARY A','A2:C200');

            if ~isempty(process_adding.A.mean_values)
            process_adding.A.mean_values(isnan(process_adding.A.mean_values))=0;
            process_adding.A.mean_values(isempty(process_adding.A.mean_values))=0;
            end

            if ~isempty(types)
            process_adding.meta_data_flows(:,2)=num2cell(types);
            end

            % B
            [process_adding.B.mean_values,~,~] = xlsread(filename,'SUMMARY B','D2:IA500');
            [~,process_adding.meta_data_elementary_flows,~] = xlsread(filename,'SUMMARY B','A2:C500');

            if ~isempty(process_adding.B.mean_values)
            process_adding.B.mean_values(isnan(process_adding.B.mean_values))=0;
            process_adding.B.mean_values(isempty(process_adding.B.mean_values))=0;
            end

            % F
            [process_adding.F.mean_values,~,~] = xlsread(filename,'SUMMARY F','C2:IA200');
            [~,process_adding.meta_data_factor_requirements,~] = xlsread(filename,'SUMMARY F','A2:C200');

            if ~isempty(process_adding.F.mean_values)
            process_adding.F.mean_values(isnan(process_adding.F.mean_values))=0;
            process_adding.F.mean_values(isempty(process_adding.F.mean_values))=0;
            end
            
            counter=counter+1;
            add_manual_processes(counter)=process_adding;

        end
        
        
end