function [] = output_proportions(Model,path)

Model_allocated = Model;

%%%%%%%%%%%%%%%%%%No Utilities or other unit than kg%%%%%%%%%%

Model_modified=Model_allocated.matrices.A.mean_values;

for j=1:size(Model_modified,1)
    for i=1:size(Model_modified,2)
        if cell2mat(Model_allocated.meta_data_flows(j+1,2))>1
            Model_modified(j,i)=0;
        end
        if strcmp(Model_allocated.meta_data_flows(j+1,6),'kg')==0
            Model_modified(j,i)=0;
        end
    end
end

%%%%%%%%%%%OUTPUTS%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

A_outputs=Model_modified;
A_outputs(A_outputs<0)=0;
summe_outputs=sum(A_outputs,1);
for row=1:size(A_outputs,2)
    for col=1:size(A_outputs,1)
        A_outputs(col,row)=A_outputs(col,row)./summe_outputs(1,row)*100;
    end
end

matrix_outputs=Model_allocated.meta_data_processes(1,2:size(Model_allocated.meta_data_processes,2));

for row=1:size(A_outputs,2)
    k=2;
    for col=1:size(A_outputs,1)
        if A_outputs(col,row)>0
      
            flow =   char(Model_allocated.meta_data_flows(col+1,1));
            title =   'Percentage Outputs: ';
            value = num2str(A_outputs(col,row));
            value2= num2str(Model_allocated.matrices.A.mean_values(col,row));
            
            fullName = [flow ', ' title value ' % , Output Value: ' value2 ' kg'];
            matrix_outputs(k,row)=cellstr(fullName);
            
            k=k+1;
        end
    end
end


%%%%%%%%%%%INPUTS%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

A_inputs=Model_modified;
A_inputs(A_inputs>0)=0;
summe_inputs=sum(A_inputs,1);
for row=1:size(A_inputs,2)
    for col=1:size(A_inputs,1)
        A_inputs(col,row)=A_inputs(col,row)./summe_inputs(1,row)*100;
    end
end

matrix_inputs=Model_allocated.meta_data_processes(1,2:size(Model_allocated.meta_data_processes,2));

for row=1:size(A_inputs,2)
    k=2;
    for col=1:size(A_inputs,1)

        if A_inputs(col,row)>0
      
            flow =   char(Model_allocated.meta_data_flows(col+1,1));
            title =   'Percentage Inputs: ';
            value = num2str(A_inputs(col,row));
            value2= num2str(Model_allocated.matrices.A.mean_values(col,row));
            
            fullName = [flow ', ' title value ' % , Input Value: ' value2 ' kg'];
            matrix_inputs(k,row)=cellstr(fullName);
            
            k=k+1;
                  
        end
    end

end

Model_utilities_nokg=Model_allocated.matrices.A.mean_values;
for j=1:size(Model_utilities_nokg,1)
    for i=1:size(Model_utilities_nokg,2)
        if cell2mat(Model_allocated.meta_data_flows(j+1,2))==1 && strcmp(Model_allocated.meta_data_flows(j+1,6),'kg')==1
            Model_utilities_nokg(j,i)=0;
        end
    end
end

matrix_utilities_nokg=Model_allocated.meta_data_processes(1,2:size(Model_allocated.meta_data_processes,2));

for row=1:size(Model_utilities_nokg,2)
    k=2;
    for col=1:size(Model_utilities_nokg,1)

        if Model_utilities_nokg(col,row)~=0
            if Model_utilities_nokg(col,row) <0
                InOut= 'Input';
            elseif Model_utilities_nokg(col,row) >0
                InOut = 'Output';
            end
            flow =   char(Model_allocated.meta_data_flows(col+1,1));
            title =   'Value: ';
            value = num2str(Model_utilities_nokg(col,row));
            unit =  char(Model_allocated.meta_data_flows(col+1,6));
            fullName = [InOut ' ' flow ', ' title value ' ' unit];
            matrix_utilities_nokg(k,row)=cellstr(fullName);
            
            k=k+1;
        end
    end

end

process_name = Model_allocated.meta_data_processes(1,:)';
process_description = Model_allocated.meta_data_processes(3,:)';
descriptions=horzcat(process_name,process_description);
matrix_inputs=matrix_inputs(2:end,:);
matrix_outputs=matrix_outputs(2:end,:);
matrix_utilities_nokg=matrix_utilities_nokg(2:end,:);

matrix_in=vertcat(num2cell(zeros(1,size(matrix_inputs,1))),matrix_inputs');
matrix_in(1,1)={'Inputs'};
matrix_out=vertcat(num2cell(zeros(1,size(matrix_outputs,1))),matrix_outputs');
matrix_out(1,1)={'Outputs'};
matrix_utilities=vertcat(num2cell(zeros(1,size(matrix_utilities_nokg,1))),matrix_utilities_nokg');
matrix_utilities(1,1)={'Utilities'};

full=horzcat(process_name,Model_allocated.meta_data_processes(4,:)',matrix_in,matrix_out,matrix_utilities,process_description);
xlswrite([path,'/','processes_inputs_outputs_utilities.xlsx'],full);
