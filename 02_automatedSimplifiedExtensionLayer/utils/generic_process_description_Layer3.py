def generic_process_description_Layer3(Model):
    # Define generic process description
    AmountProcesses = len(Model['meta_data_processes'][0]) - 1

    TextProcess1 = "The "
    TextProcess2 = " produces the main product "
    TextProcess3 = ". "

    RawMaterials1 = "For this purpose, the process consumes "
    RawMaterials2 = ". "

    Utilities1_plural = "Utilities consumed comprise "
    Utilities2_plural = ". "
    Utilities_singular = " is consumed as utility. "

    ByProducts1_plural = "By-products comprise "
    ByProducts2_plural = ". "
    ByProducts_singular = " is produced as by-product. "

    EnergyProducts1 = "Additionally, the process produces "
    EnergyProducts2_plural = " as utility by-products."
    EnergyProducts2_singular = " as a utility by-product."
    EnergyProducts3_plural = " The multifunctionality problem for utility by-products is solved by system expansion via the avoided burden approach."
    EnergyProducts3_singular = " The multifunctionality problem for the utility by-product is solved by system expansion via the avoided burden approach."

    GendorfInformation = ("Utility demands in form of thermal energy and electricity are estimated according to "
                          "the chemical park in Gendorf. For each kg of chemical product, 1.2 MJ electricity and "
                          "2 MJ thermal energy are required. A conversion rate of 95% is assumed. Process wastes are "
                          "treated in municipal waste incinerators.")

    BackgroundModelling = ("Background modeling: \nThe data set represents a cradle to gate inventory, including "
                           "all relevant process steps / technologies over the supply chain. The data set is based on "
                           "different types of data: Process data is obtained from detailed process simulations or "
                           "simplified modeling. International trade volumes and regional production capacities are "
                           "mainly based on primary data and complemented by secondary data where necessary.\n"
                           "Electricity is modeled according to the individual country-specific situations, including "
                           "national electricity grid mixes and imported electricity.\nSteam and thermal energy supplies "
                           "take into account the country-specific situation, wherever possible. Otherwise, larger regional "
                           "averages are used.\nThe production of crude oil, naphtha, and natural gas is represented by "
                           "either fully country-specific data or by partly representative data for a fully overlapping but "
                           "not identical region (e.g., European average for a specific European country).")

    for i in range(AmountProcesses):
        String1 = ""
        String2 = ""
        String3 = ""
        String4 = ""
        String5 = ""

        Process = Model['meta_data_processes'][0][i + 1]
        MainProduct = Model['meta_data_processes'][3][i + 1]

        InputA = Model['matrices']['A']['mean_values'][:, i] < 0
        OutputA = Model['matrices']['A']['mean_values'][:, i] > 0

        MainProductA = [flow[0] == MainProduct for flow in Model['meta_data_flows'][1:]]

        Utilities = [int(flow[1]) == 2 for flow in Model['meta_data_flows'][1:]]
        RawMaterials = [int(flow[1]) == 1 for flow in Model['meta_data_flows'][1:]]

        UtilitiesInput = [Model['meta_data_flows'][idx + 1][0] for idx, val in enumerate(InputA & Utilities) if val]
        n_UtilitiesInput = len(UtilitiesInput)

        if any(OutputA & Utilities):
            UtilitiesOutput = [Model['meta_data_flows'][idx + 1][0] for idx, val in enumerate(OutputA & Utilities) if val]
            n_UtilitiesOutput = len(UtilitiesOutput)

        RawMaterialsInputs = [Model['meta_data_flows'][idx + 1][0] for idx, val in enumerate(InputA & RawMaterials) if val]

        if any(OutputA & RawMaterials & ~MainProductA):
            RawMaterialsByProducts = [Model['meta_data_flows'][idx + 1][0] for idx, val in enumerate(OutputA & RawMaterials & ~MainProductA) if val]
            n_RawMaterialsByProducts = len(RawMaterialsByProducts)

        # first sentence about the main product
        String1 = f"{TextProcess1}{Process}{TextProcess2}{MainProduct}{TextProcess3}"

        # second sentence about the main raw materials
        if any(InputA & RawMaterials):
            EnumeratedString = generate_enumeration(RawMaterialsInputs)
            String2 = f"{RawMaterials1}{EnumeratedString}{RawMaterials2}"

        # third sentence about the utility consumption
        if any(InputA & Utilities):
            if n_UtilitiesInput == 1:  # singular
                String3 = UtilitiesInput[0].capitalize() + Utilities_singular
            else:  # plural
                EnumeratedString = generate_enumeration(UtilitiesInput)
                String3 = f"{Utilities1_plural}{EnumeratedString}{Utilities2_plural}"

        # fourth sentence about the by-products
        if any(OutputA & RawMaterials & ~MainProductA):
            if n_RawMaterialsByProducts == 1:  # singular
                String4 = RawMaterialsByProducts[0].capitalize() + ByProducts_singular
            else:  # plural
                EnumeratedString = generate_enumeration(RawMaterialsByProducts)
                String4 = f"{ByProducts1_plural}{EnumeratedString}{ByProducts2_plural}"

        # fourth sentence about the utility outputs
        if any(OutputA & Utilities):
            EnumeratedString = generate_enumeration(UtilitiesOutput)
            if n_UtilitiesOutput == 1:  # singular
                String5 = f"{EnergyProducts1}{EnumeratedString}{EnergyProducts2_singular}{EnergyProducts3_singular}"
            else:  # plural
                String5 = f"{EnergyProducts1}{EnumeratedString}{EnergyProducts2_plural}{EnergyProducts3_plural}"

        ProcessDescription = f"{String1}{String2}{String3}{String4}{String5}{GendorfInformation}\n{BackgroundModelling}"

        Model['meta_data_processes'][2][i + 1] = ProcessDescription

    return Model

def generate_enumeration(items):
    if len(items) == 1:
        return items[0]
    return ', '.join(items[:-1]) + " and " + items[-1]
