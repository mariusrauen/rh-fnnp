function [carbon_contents, names_of_flows] = get_carbon_content(Model,elements,M_e,raw)

%% get data for calculation

% get meta_data
meta_data_flows=Model.meta_data_flows;

% get A matrix
% A_without_utilities = Model.matrices.A.mean_values;

% write inputs to waste_code struct
waste_code.chemical_formula.flows = meta_data_flows(2:end,1);  % flow names from meta data
waste_code.chemical_formula.chemical_formulas = meta_data_flows(2:end,10); % all chemical formulas from meta data
waste_code.chemical_formula.elements = elements;
waste_code.M_e = M_e;
waste_code.chemical_formula.values = zeros(size(waste_code.chemical_formula.flows,1),size(waste_code.chemical_formula.elements,1));
waste_code.chemical_formula.elements_missing = {};

% delete utility chemical formulas

utility_flows = cell2mat(meta_data_flows(2:end,2))==2;

waste_code.chemical_formula.chemical_formulas(utility_flows) = {[]};
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

%%
carbon_contents = waste_code.createE.E(8,:);
names_of_flows = waste_code.chemical_formula.flows';

end