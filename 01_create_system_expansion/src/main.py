import calc_automated_extension_layer as calc

def main(input_file_path, output_file_path):
    """Get System expansion and matrix table mass"""
    #1.) load all data from the reactions the excel sheet
    matrix_table = calc.load_prepare_data(input_file_path)
  
    #2) check if the carbon content is balanced
    calc.carbon_content_check(matrix_table)

    #3) the mass of all raw materials and products is calculted based
    # based on the stoichometric coefficents and normalized of 1kg product (mainflow)
    matrix_mass = calc.calculate_mass(matrix_table)

    #4) A yield of 0.95% is considered for the reactions.
    matrix_mass = calc.consider_yield_095(matrix_mass)

    #5) The energy demand is calculated according to Gendorf.
    matrix_demand = calc.calculate_energy_demand_Gendorf(matrix_mass, matrix_table)

    #6) the new mass based flows are saved in an excel file. 
    matrix_table_mass = calc.create_preprocessing_mass_table(matrix_mass, matrix_demand)

    #7) A matrix is created from the raw material and paroduct mass flows based on 
    # the matrix table design.
    matrixA = calc.matrix_creation(matrix_table_mass ,'process_id','raw_material_id','mass', 'raw materials', 'process')
    
    # The final systemexpansion table is created including process_meta_data, summary A, 
    # Summery B, and summary F
    calc.create_system_expansion(matrixA, output_file_path)
    print("Done!!!")


if __name__ == "__main__":
    main(input_file_path = 'reaction_extension_layer.xlsx',
         output_file_path = 'SYSTEMEXPANSION.xlsx' )
