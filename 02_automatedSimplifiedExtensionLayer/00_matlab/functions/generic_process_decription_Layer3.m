function [Model] = generic_process_decription_Layer3(Model)

%% Define generic process description
AmountProcesses = size(Model.meta_data_processes(1,1:end),2)-1;

TextProcess1 = "The ";
TextProcess2 = " produces the main product ";
TextProcess3 = ". ";

RawMaterials1 = "For this purpose, the process consumes ";
RawMaterials2 = ". ";

Utilities1_plural = "Utilities consumed comprise ";
Utilities2_plural = ". ";
Utilities_singular = " is consumed as utility. ";

ByProducts1_plural = "By-products comprise ";
ByProducts2_plural = ". ";
ByProducts_singular = " is produced as by-product. ";

EnergyProducts1 = "Additionally, the process produces ";
EnergyProducts2_plural = " as utility by-products.";
EnergyProducts2_singular = " as a utility by-product.";
EnergyProducts3_plural = " The multifunctionality problem for utility by-products is solved by system expansion via the avoided burden approach.";
EnergyProducts3_singular = " The multifunctionality problem for the utility by-product is solved by system expansion via the avoided burden approach.";

GendorfInformation = "Utility demands in form of thermal energy and electricity are estimated according to the chemical park in Gendorf. For each kg of chemical product, 1.2 MJ electricity and 2 MJ thermal energy are required. A conversion rate of 95% is assumed. Process wastes are treated in municipal waste incinerators.";

% General texts

BackgroundModelling = "Background modeling: " + ...
    newline + "The data set represents a cradle to gate inventory, including all relevant process steps / technologies over the supply chain. The data set is based on different types of data: Process data is obtained from detailed process simulations or simplified modeling. International trade volumes and regional production capacities are mainly based on primary data and complemented by secondary data where necessary." + ...
    newline + "Electricity is modeled according to the individual country-specific situations, including national electricity grid mixes and imported electricity." + ...
    newline + "Steam and thermal energy supplies take into account the country-specific situation, wherever possible. Otherwise, larger regional averages are used." + ...
    newline + "The production of crude oil, naphtha, and natural gas is represented by either fully country-specific data or by partly representative data for a fully overlapping but not identical region (e.g., European average for a specific European country)." ;

    
%%
for i = 1:AmountProcesses
    
    String1 = "";
    String2 = "";
    String3 = "";
    String4 = "";
    String5 = "";
    
    Process = Model.meta_data_processes(1,i+1);
    MainProduct = Model.meta_data_processes(4,i+1);
    
    InputA = (Model.matrices.A.mean_values(:,i)<0);
    OutputA = (Model.matrices.A.mean_values(:,i)>0);
    
    MainProductA = strcmp(Model.meta_data_flows(2:end,1),MainProduct);
    
    Utilities = cell2mat(Model.meta_data_flows(2:end,2)) == 2;
    RawMaterials = cell2mat(Model.meta_data_flows(2:end,2)) == 1;
    
    UtilitiesInput = string(Model.meta_data_flows(find(InputA&Utilities)+1,1));
    % get amount of UtilitiesInput
    n_UtilitiesInput = size(UtilitiesInput,1);
    
    if any(OutputA&Utilities)
        UtilitiesOutput = string(Model.meta_data_flows(find(OutputA&Utilities)+1,1));
        % get amount of UtilitiesOutput
        n_UtilitiesOutput = size(UtilitiesOutput,1);
    end
    
    RawMaterialsInputs = string(Model.meta_data_flows(find(InputA&RawMaterials)+1,1));
    % get amount of RawMaterialsInputs
    
    if any(OutputA&RawMaterials&~MainProductA)
        RawMaterialsByProducts = string(Model.meta_data_flows(find(OutputA&RawMaterials&~MainProductA)+1,1));
        % get amount of RawMaterialsByProducts
        n_RawMaterialsByProducts = size(RawMaterialsByProducts,1);
    end
    
    % first sentence about the main product
        
    String1 = append(TextProcess1,Process,TextProcess2,MainProduct,TextProcess3);
    
    % second sentence about the main raw materials
    
    if any(InputA&RawMaterials)
        
        [EnumeratedString] = generate_enumeration(RawMaterialsInputs);
        String2 = append(RawMaterials1,EnumeratedString,RawMaterials2);
        
    end
    
    % third sentence about the utility consumption
    
    if any(InputA&Utilities)

        % decide between singular and plural
        if n_UtilitiesInput == 1 % singular, Attention! 1st flow letter is converted to caps!
        String3 = append(regexprep(UtilitiesInput,'(^[a-z])','${upper($1)}'),Utilities_singular);
        else % plural
        [EnumeratedString] = generate_enumeration(UtilitiesInput);
        String3 = append(Utilities1_plural,EnumeratedString,Utilities2_plural);
        end
        
    end
    
    % fourth sentence about the by-products
    
    if any(OutputA&RawMaterials&~MainProductA)
        
        % decide between singular and plural
        if n_RawMaterialsByProducts == 1 % singular, Attention! 1st flow letter is converted to caps!
        String4 = append(regexprep(RawMaterialsByProducts,'(^[a-z])','${upper($1)}'),ByProducts_singular);
        else % plural
        [EnumeratedString] = generate_enumeration(RawMaterialsByProducts);
        String4 = append(ByProducts1_plural,EnumeratedString,ByProducts2_plural);
        end
        
    end
    
    % fourth sentence about the utility
    
    if any(OutputA&Utilities)
        
        [EnumeratedString] = generate_enumeration(UtilitiesOutput);
        
        % decide between singular and plural
        if n_UtilitiesOutput == 1 % singular
        String5 = append(EnergyProducts1,EnumeratedString,EnergyProducts2_singular,EnergyProducts3_singular);
        else % plural
        String5 = append(EnergyProducts1,EnumeratedString,EnergyProducts2_plural,EnergyProducts3_plural);    
        end
        
    end
    
    ProcessDescription = append(String1,String2,String3,String4,String5,GendorfInformation,newline,BackgroundModelling);
    
    Model.meta_data_processes{3,i+1} = ProcessDescription;
    
end

%%
end