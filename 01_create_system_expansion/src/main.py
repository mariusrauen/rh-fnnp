import calc_automated_extension_layer as calc
from pathlib import Path

def main(input_file_path: Path, output_file_path: Path):
    """Get System expansion and matrix table mass"""
    #1.) Load all data from the reactions the excel sheet.
    matrix_table = calc.load_prepare_data(input_file_path)
  
    #2) Check if the carbon content is balanced.
    calc.carbon_content_check(matrix_table)

    #3) The mass of all raw materials and products is calculated based
    #   on the stoichometric coefficents and normalized to 1kg product (mainflow).
    matrix_mass = calc.calculate_mass(matrix_table)

    #4) A yield of 0.95% is considered for the reactions. Therefore we increased
    #   the mass of the educts to compensate and still end up with 1kg product. 
    matrix_mass = calc.consider_yield_095(matrix_mass)

    #5) The energy demand is calculated according to Gendorf.
    matrix_demand = calc.calculate_energy_demand_Gendorf(matrix_mass, matrix_table)

    #6) The new mass based flows are saved in an excel file. 
    matrix_table_mass = calc.create_preprocessing_mass_table(matrix_mass, matrix_demand)

    #7) A matrix is created from the raw material and product mass flows based on 
    #   the matrix table design.
    matrixA = calc.matrix_creation(matrix_table_mass ,'process_id','raw_material_id','mass', 'raw materials', 'process')
    
    # The final systemexpansion table is created including process_meta_data, summary A, 
    # summary B, and summary F.
    calc.create_system_expansion(matrixA, output_file_path)
    print("Done!!!")


if __name__ == "__main__":
    INPUT_FILE_PATH = Path("../xlsx/reaction_extension_layer.xlsx")
    OUTPUT_FILE_PATH = Path("../xlsx/SYSTEMEXPANSION.xlsx") 
    main(input_file_path = INPUT_FILE_PATH,
         output_file_path = OUTPUT_FILE_PATH)
