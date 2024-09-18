%% Load everything
clear
clc

load('G:\Geteilte Ablagen\09a_NEW_cm_chemicals_data\01_final_technology_datasets\00b_alignments\M_e.mat')
load('G:\Geteilte Ablagen\09a_NEW_cm_chemicals_data\01_final_technology_datasets\00b_alignments\elements.mat')

file = "G:\Geteilte Ablagen\09a_NEW_cm_chemicals_data\01_final_technology_datasets\00b_alignments\meta_data_flows.xlsx";
opts = spreadsheetImportOptions("NumVariables", 17);
% Specify sheet and range
opts.Sheet = "Tabelle1";
opts.DataRange = "A2";
% Specify column names and types
opts.VariableNames = ["name", "category", "concentrationpurity", "CASNr", "unitchoice", "unitchoice1", "Marketprice", "HHV", "LHV", "chemicalformula", "locationchoice", "exactlocation", "comments", "SMILESCODE", "molecularmass", "flowCategory", "flowSubcategory"];
opts.VariableTypes = ["string", "double", "string", "string", "categorical", "categorical", "double", "string", "double", "string", "string", "string", "string", "categorical", "double", "categorical", "categorical"];
% Specify variable properties
opts = setvaropts(opts, ["name", "concentrationpurity", "CASNr", "HHV", "chemicalformula", "locationchoice", "exactlocation", "comments"], "WhitespaceRule", "preserve");
opts = setvaropts(opts, ["name", "concentrationpurity", "CASNr", "unitchoice", "unitchoice1", "HHV", "chemicalformula", "locationchoice", "exactlocation", "comments", "SMILESCODE", "flowCategory", "flowSubcategory"], "EmptyFieldRule", "auto");
% Import the data
meta_data_flows = readtable(file, opts, "UseExcel", false);
meta_data_flows(meta_data_flows.category == 2,:) = [];
meta_data_flows = removevars(meta_data_flows, {'category','concentrationpurity','CASNr','unitchoice','unitchoice1','Marketprice','HHV','LHV','locationchoice','exactlocation','comments','SMILESCODE','molecularmass','flowCategory','flowSubcategory'});
clear opts

%% calculate carbon_contents
% write inputs to carbon_content struct
carbon_content.flows=meta_data_flows.name;  % flow names from meta data
carbon_content.chemical_formulas=meta_data_flows.chemicalformula; % all chemical formulas from meta data
carbon_content.elements=elements;
carbon_content.M_e=M_e;
carbon_content.values=zeros(size(carbon_content.flows,1),size(carbon_content.elements,1));
carbon_content.elements_missing = {};

clear elements M_e meta_data_flows;

%% 2) Create matrix containing elements of each flow in A (without utilities)

chemical_formulas_new = carbon_content.chemical_formulas;

for i=1:size(carbon_content.flows,1)
        % modify chemical formula (e.g. CH3COOH --> C1H3C1O1O1H1)
        stringIn = char(carbon_content.chemical_formulas(i,1)); 
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
        
        % fill matrix "carbon_content.values" (rows: flows, colums: elements)
        for j=1:size(letters,2)
            for idx=1:size(carbon_content.elements,1)
                if isequal(letters(1,j),carbon_content.elements(idx,1)) == 1
                    carbon_content.values(i,idx) = numbers(1,j);
                end
            end
            
        % find elements not existing in "carbon_content.elements"
            if ismember(letters(1,j),carbon_content.elements) == 0 && isempty(letters{1,j}) == 0
                carbon_content.elements_missing(end+1,1) = letters(1,j);
            end
        end           

    clear letters numbers 
end

if isempty(carbon_content.elements_missing) == 0
    display('WARNING: See missing elements in "Model.carbon_content.elements_missing" and update in excel file "elements_and_molar_mass"!');
end

clear i j idx stringIn stringOut chemical_formulas_new


%% calculate carbon content
carbon_content.elements_table = table(string(carbon_content.elements),carbon_content.M_e,'VariableNames',{'elements','M_e'});
index_C = find(carbon_content.elements_table.elements == "C");
carbon_content.mass_carbon = carbon_content.values(:,index_C).*carbon_content.elements_table.M_e(index_C);
carbon_content.mass_sum = carbon_content.values * carbon_content.elements_table.M_e;
carbon_content.carbon_content = carbon_content.mass_carbon./carbon_content.mass_sum;
