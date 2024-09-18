function [Model_out] = get_waste_and_elementary_flows_direct_emissions(Model,elements,M_e,raw)
%% 1) Create new matrices for chemical formula - stored in waste_code struct

RowsofUtilities=[];

for i=1:size(Model.meta_data_flows,1)-1
    if cell2mat(Model.meta_data_flows(i+1,2)) == 2
        RowsofUtilities(end+1,1)=i;
    end
end
clear i

% delete all utility flows in meta_data for waste code
meta_data_flows_without_utilities=Model.meta_data_flows;
meta_data_flows_without_utilities (RowsofUtilities+1,:)=[];

% delete all utility flows in A for waste code
A_without_utilities = Model.matrices.A.mean_values;
A_without_utilities (RowsofUtilities,:) = [];

% write inputs to waste_code struct
waste_code.chemical_formula.flows=meta_data_flows_without_utilities(2:end,1);  % flow names from meta data
waste_code.chemical_formula.chemical_formulas=meta_data_flows_without_utilities(2:end,10); % all chemical formulas from meta data
waste_code.chemical_formula.elements=elements;
waste_code.M_e=M_e;
waste_code.chemical_formula.values=zeros(size(waste_code.chemical_formula.flows,1),size(waste_code.chemical_formula.elements,1));
waste_code.chemical_formula.elements_missing = {};

clear elements M_e clear meta_data_flows_without_utilities RowsofUtilities;

%% 2) Create matrix containing elements of each flow in A (without utilities)

chemical_formulas_new = waste_code.chemical_formula.chemical_formulas;

for i=1:size(waste_code.chemical_formula.flows,1)
    if iscellstr(waste_code.chemical_formula.chemical_formulas(i,1))==1 % if chemical formula is available
        
        % modify chemical formula (e.g. CH3COOH --> C1H3C1O1O1H1)
        stringIn = char(waste_code.chemical_formula.chemical_formulas(i,1)); 
        stringOut = regexprep(stringIn, '([A-Z])', ' 1$1');
        stringOut(regexp(stringOut,'\d 1')+2)=[];
        stringOut=regexprep(stringOut,'\s','');
        stringOut=stringOut(2:end);
        chemical_formulas_new(i,1)=cellstr(stringOut);   
        
        % seperate numbers and letters
        letters = strsplit(regexprep(stringOut, '\d*', ' '));
        numbers = strsplit(regexprep(stringOut, '\D*', ' '));
        numbers = str2num(char(numbers(1,2:end)'))';
        
        if size(numbers,2) < size(letters,2)
            numbers(1,end+1)=1;
        end
        
        % fill matrix "waste_code.chemical_formula.values" (rows: flows, colums: elements)
        for j=1:size(letters,2)
            for idx=1:size(waste_code.chemical_formula.elements,1)
                if isequal(letters(1,j),waste_code.chemical_formula.elements(idx,1)) == 1
                    waste_code.chemical_formula.values(i,idx) = numbers(1,j);
                end
            end
            
        % find elements not existing in "waste_code.chemical_formula.elements"
            if ismember(letters(1,j),waste_code.chemical_formula.elements) == 0 && isempty(letters{1,j}) == 0
                waste_code.chemical_formula.elements_missing(end+1,1) = letters(1,j);
            end
        end           
    end
    clear letters numbers 
end

if isempty(waste_code.chemical_formula.elements_missing) == 0
    display('WARNING: See missing elements in "Model.waste_code.chemical_formula.elements_missing" and update in excel file "elements_and_molar_mass"!');
end

clear i j idx stringIn stringOut chemical_formulas_new

%% 3) Create in-and output mass flows of each element in each process

% get elements of each flow in A and store in createE struct
waste_code.createE.Z = waste_code.chemical_formula.values; % elements in flows
waste_code.createE.s = size(waste_code.createE.Z,1); % row amount
waste_code.createE.e = size(waste_code.createE.Z,2); % column amount

% molar masses of elements
waste_code.createE.M_e_matrix = meshgrid(waste_code.M_e, 1:waste_code.createE.s)./1000;

% scale to processes to one
waste_code.createE.h = ones(waste_code.createE.s,1);

% calculate molar masses of each flow in g/mol (M_s)
waste_code.createE.M_s = waste_code.createE.Z*waste_code.M_e;

% calculate spezific molar weight in kg/mol (n_s)
waste_code.createE.n_s = zeros(waste_code.createE.s,1);
for i = 1:waste_code.createE.s
    if waste_code.createE.M_s(i)~=0     % molar mass of flow is not zero
        waste_code.createE.n_s(i) = waste_code.createE.h(i)/waste_code.createE.M_s(i)*1000;
    else             % molar mass of flow is zero
        waste_code.createE.n_s(i) = 0;
    end
end

clear i

% transform to matrix with size s x e
waste_code.createE.N_s = meshgrid(waste_code.createE.n_s, 1:waste_code.createE.e)';

% calculate amount (mol) of substance per kg of flow
waste_code.createE.E_n = waste_code.createE.N_s.*waste_code.createE.Z;

% calculate masses of each element in each flow and transpose
waste_code.createE.E = (waste_code.createE.E_n.*waste_code.createE.M_e_matrix)';



%% 4) Complete Model
% this function closes atom balances and introduces X-to-waste processes into A
% matrix as well as completes the other matrices regarding their size

% derive mass balance of each element in every process
waste_code.completeModel.M.mean_values = waste_code.createE.E*A_without_utilities; % M contains balance of each element in each process (thus its size is #elements x #processes)


%% find all elementary waste in every process (output waste)
waste_code.completeModel.elementstoWaste.mean_values = waste_code.completeModel.M.mean_values;
waste_code.completeModel.elementstoWaste.mean_values(waste_code.completeModel.elementstoWaste.mean_values >= 0) = 0;
waste_code.completeModel.elementstoWaste.mean_values=-waste_code.completeModel.elementstoWaste.mean_values;

%% find all missing elementary inputs (elementary inputs)

waste_code.completeModel.tooManyOutputs = waste_code.completeModel.M.mean_values;
waste_code.completeModel.tooManyOutputs(waste_code.completeModel.tooManyOutputs <= 0) = 0;

waste_code.completeModel.zerorows =[];
for i=1:size(waste_code.completeModel.tooManyOutputs,1)
    if waste_code.completeModel.tooManyOutputs(i,:) == 0
        waste_code.completeModel.zerorows(end+1,1)=i;
    end
end

waste_code.completeModel.zerocols =[];
for i=1:size(waste_code.completeModel.tooManyOutputs,2)
    if waste_code.completeModel.tooManyOutputs(:,i) == 0
        waste_code.completeModel.zerocols(end+1,1)=i;
    end
end

waste_code.completeModel.tooManyOutputs(waste_code.completeModel.zerorows,:)=[];
waste_code.completeModel.tooManyOutputs(:,waste_code.completeModel.zerocols)=[];
waste_code.completeModel.elements_tooManyOutputs=waste_code.chemical_formula.elements;
waste_code.completeModel.elements_tooManyOutputs(waste_code.completeModel.zerorows,:)=[];
waste_code.completeModel.processes_tooManyOutputs=Model.meta_data_processes(1,2:end);
waste_code.completeModel.processes_tooManyOutputs(:,waste_code.completeModel.zerocols)=[];
Matrix_tooManyOutputs=vertcat(horzcat({'elements/processes'},waste_code.completeModel.processes_tooManyOutputs) ,horzcat(waste_code.completeModel.elements_tooManyOutputs, num2cell(waste_code.completeModel.tooManyOutputs)));

%% overwrite
raw_A.A = raw.rawA; raw_B.B = raw.rawB; clear raw

%% delete flows raw_A that have unit 'unit' or unit 'tkm' ir 'Kg' --> delete all flows
delete_flows={};
for i=2:size(raw_A.A,1)
    if isequal(raw_A.A{i,2},'unit') == 1 ...
        || isequal(raw_A.A{i,2},'tkm')...
%         || isequal(raw_A.A{i,2},'kg') == 1 ...
%         || isequal(raw_A.A{i,2},'MJ') == 1
        delete_flows(end+1,1)=raw_A.A(i,1);
    end
end
raw_A.A(ismember(raw_A.A(:,1),delete_flows),:)=[];
clear i delete flows
%% convert all isnan to 0

raw_A.A(find(cellfun(@(C)any(isnan(C(:))), raw_A.A)))={0}; raw_B.B(cellfun(@(C)any(isnan(C(:))), raw_B.B))={0};

%% add elements (processes) if not existing in raw_A (p.e. Argon)
for i=1:size(waste_code.chemical_formula.elements)
    if ismember(waste_code.chemical_formula.elements(i,1),raw_A.A(1,:)) == 0
        % create new column in raw_A.A if chmicals are missing
        raw_A.A(1,end+1) = waste_code.chemical_formula.elements(i,1); raw_A.A(2:end,end) ={0};
    end 
end
clear i

%% add elements (processes) if not existing in raw_B (p.e. Argon)
for i=1:size(waste_code.chemical_formula.elements)
    if ismember(waste_code.chemical_formula.elements(i,1),raw_B.B(1,:)) == 0
        % create new column in raw_BB if chemicals are missing
        raw_B.B(1,end+1) = waste_code.chemical_formula.elements(i,1); raw_B.B(2:end,end) ={0}; 
%       raw_B.B(end+1,1) = cellstr(sprintf('Emissions of %s',waste_code.chemical_formula.elements{i,1})); raw_B.B(end,2:end-1) = {0}; raw_B.B(end,end) = {1};  raw_B.B(end,4) ={'kg'};
    end 
end    
clear i

%% get direct emissions and utilities for wastes treatment

elements_to_waste = waste_code.completeModel.elementstoWaste.mean_values;
elements_to_wastes_names = waste_code.chemical_formula.elements;

names_rawB=raw_B.B(1,5:end); % row names of B

values_rawB=cell2mat(raw_B.B(2:end,5:end)); % values for elementary flows
values_rawA=cell2mat(raw_A.A(2:end,4:end)); % values for utilities

% generate emissions and waste treatment utilities of each process
for i=1:size(elements_to_wastes_names,1)
    for j=1:size(names_rawB,2)
        if isequal(elements_to_wastes_names(i),names_rawB(1,j)) == 1
           B_inc(:,i) = values_rawB(:,j);
           A_inc(:,i) = values_rawA(:,j);
        end
    end
end

waste_code.completeModel.EmissionsPerProcess=B_inc*elements_to_waste;
waste_code.completeModel.WasteUtilitiesPerProcess=A_inc*elements_to_waste;

clear A_inc B_inc i j elements_to_waste elements_to_wastes_names names_rawB values_rawB values_rawA
%% Include flows in Model

% A matrix and meta data flows manipulation
Model_waste = Model;
end_meta_data_flows=size(Model_waste.meta_data_flows,1);

delete = [];

for i = 1:size(raw_A.A,1)-1 %% add existing flows to model
    
    RowModel = strcmp(Model_waste.meta_data_flows(2:end,1),raw_A.A(i+1,1));
    
    if any(RowModel)
        
    delete = [delete,i];
    
    Model_waste.matrices.A.mean_values(RowModel,:) = Model_waste.matrices.A.mean_values(RowModel,:) + waste_code.completeModel.WasteUtilitiesPerProcess(i,:);
    
    end
    
end

% delete already included flows
raw_A.A(delete+1,:) = [];
waste_code.completeModel.WasteUtilitiesPerProcess(delete,:) = [];

Model_waste.meta_data_flows(end_meta_data_flows+1:end_meta_data_flows+size(raw_A.A(2:end,1:3),1),1)=raw_A.A(2:end,1);
Model_waste.meta_data_flows(end_meta_data_flows+1:end_meta_data_flows+size(raw_A.A(2:end,1:3),1),6)=raw_A.A(2:end,2);
Model_waste.meta_data_flows(end_meta_data_flows+1:end_meta_data_flows+size(raw_A.A(2:end,1:3),1),2)=raw_A.A(2:end,3);

Model_waste.matrices.A.mean_values = vertcat(Model_waste.matrices.A.mean_values,waste_code.completeModel.WasteUtilitiesPerProcess);

clear end_meta_data_flows

% B matrix and meta data elementary flow manipulation

name_model = Model_waste.meta_data_elementary_flows(2:end,1);
medium_model = Model_waste.meta_data_elementary_flows(2:end,2);
location_model = Model_waste.meta_data_elementary_flows(2:end,3);

name_waste = raw_B.B(2:end,1);
medium_waste = raw_B.B(2:end,2);
location_waste = raw_B.B(2:end,3);

for i=1:size(name_waste,1)

    for j=1:size(name_model,1)
        
        if isequal(name_waste(i),name_model(j)) &&...
           isequal(medium_waste(i),medium_model(j)) &&...
           isequal(location_waste(i),location_model(j))
            
        Model_waste.matrices.B.mean_values(j,:)=...
            Model_waste.matrices.B.mean_values(j,:)+...
            waste_code.completeModel.EmissionsPerProcess(i,:);
        
        break
        
        elseif j == size(name_model,1) 
            if waste_code.completeModel.EmissionsPerProcess(i,:) ~= 0
                error(["Elementary flow " name_waste(i) "from waste model not found in elementary flow list of defined ecoinvent version. To change the ecoinvent version, go to line 27 in main_generate_initail_model.m. You need to re-run the 'main_run_all.m' code afterwards!"])
            end 
            
        end
        
    end
    
end
 
clear i j name_model medium_model location_model  name_waste medium_waste location_waste

Model_waste.waste_code=waste_code;

Model_waste.waste_code.Matrix_tooManyOutputs = Matrix_tooManyOutputs;

Model_out=Model_waste;
end