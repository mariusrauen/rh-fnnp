# import regular exporession to search for patterns (both unicode- or 8bit strings)
import re

def clean_up_TM_symbols(Model):
    # corrupted (TM)-symbol
    codedstring_1 = '\u0099'
    invalid_character_1 = codedstring_1.encode('utf-8').decode('latin-1')

    # correct (TM)-symbol
    codedstring_2 = '\u2122'
    invalid_character_2 = codedstring_2.encode('utf-8').decode('utf-8')

    for i in range(1, len(Model['meta_data_processes'][0])):  # 1-indexed to 2 in MATLAB -> range starts at 1
        # Check for the corrupted (TM) symbol
        if invalid_character_1 in Model['meta_data_processes'][0][i]:
            Model['meta_data_processes'][0][i] = re.sub(invalid_character_1, '', Model['meta_data_processes'][0][i])
            Model['meta_data_processes'][1][i] = re.sub(invalid_character_1, '', Model['meta_data_processes'][1][i])
        
        # Check for the correct (TM) symbol
        if invalid_character_2 in Model['meta_data_processes'][0][i]:
            Model['meta_data_processes'][0][i] = re.sub(invalid_character_2, '', Model['meta_data_processes'][0][i])
            Model['meta_data_processes'][1][i] = re.sub(invalid_character_2, '', Model['meta_data_processes'][1][i])

    return Model
