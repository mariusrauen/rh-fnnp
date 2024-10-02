import numpy as np
import re
import pandas as pd

def get_waste_and_elementary_flows_direct_emissions(Model, elements, M_e, raw):
    # 1) Create new matrices for chemical formula - stored in waste_code dict
    RowsofUtilities = []

    for i in range(len(Model['meta_data_flows']) - 1):
        if Model['meta_data_flows'][i + 1][1] == 2:
            RowsofUtilities.append(i)
    
    # Delete all utility flows in meta_data for waste code
    meta_data_flows_without_utilities = Model['meta_data_flows'][:]
    for row in reversed(RowsofUtilities):
        del meta_data_flows_without_utilities[row + 1]
    
    # Delete all utility flows in A for waste code
    A_without_utilities = np.delete(Model['matrices']['A']['mean_values'], RowsofUtilities, axis=0)
    
    # Write inputs to waste_code dict
    waste_code = {
        'chemical_formula': {
            'flows': [row[0] for row in meta_data_flows_without_utilities[1:]],
            'chemical_formulas': [row[9] for row in meta_data_flows_without_utilities[1:]],
            'elements': elements,
            'values': np.zeros((len(meta_data_flows_without_utilities) - 1, len(elements))),
            'elements_missing': []
        },
        'M_e': M_e
    }

    # 2) Create matrix containing elements of each flow in A (without utilities)
    chemical_formulas_new = waste_code['chemical_formula']['chemical_formulas']

    for i, formula in enumerate(waste_code['chemical_formula']['chemical_formulas']):
        if isinstance(formula, str):
            # Modify chemical formula (e.g., CH3COOH --> C1H3C1O1O1H1)
            stringIn = formula
            stringOut = re.sub(r'([A-Z])', r' 1\1', stringIn)
            stringOut = re.sub(r'(\d) 1', r'\1', stringOut).replace(' ', '')
            chemical_formulas_new[i] = stringOut[1:]

            # Separate numbers and letters
            letters = re.findall(r'[A-Z][a-z]*', stringOut)
            numbers = [int(n) for n in re.findall(r'\d+', stringOut)]
            if len(numbers) < len(letters):
                numbers.append(1)

            # Fill matrix "waste_code['chemical_formula']['values']"
            for j, letter in enumerate(letters):
                for idx, element in enumerate(waste_code['chemical_formula']['elements']):
                    if letter == element:
                        waste_code['chemical_formula']['values'][i, idx] = numbers[j]

            # Find missing elements
            for letter in letters:
                if letter not in waste_code['chemical_formula']['elements'] and letter != '':
                    waste_code['chemical_formula']['elements_missing'].append(letter)

    if waste_code['chemical_formula']['elements_missing']:
        print('WARNING: See missing elements in "Model.waste_code.chemical_formula.elements_missing" and update in the excel file "elements_and_molar_mass"!')

    # 3) Create in-and output mass flows of each element in each process
    waste_code['createE'] = {
        'Z': waste_code['chemical_formula']['values'],
        's': waste_code['chemical_formula']['values'].shape[0],
        'e': waste_code['chemical_formula']['values'].shape[1],
        'M_e_matrix': np.tile(waste_code['M_e'] / 1000, (waste_code['chemical_formula']['values'].shape[0], 1)),
        'h': np.ones(waste_code['chemical_formula']['values'].shape[0]),
    }

    waste_code['createE']['M_s'] = waste_code['createE']['Z'] @ waste_code['M_e']
    waste_code['createE']['n_s'] = np.zeros(waste_code['createE']['s'])
    for i in range(waste_code['createE']['s']):
        if waste_code['createE']['M_s'][i] != 0:
            waste_code['createE']['n_s'][i] = waste_code['createE']['h'][i] / waste_code['createE']['M_s'][i] * 1000

    waste_code['createE']['N_s'] = np.tile(waste_code['createE']['n_s'], (waste_code['createE']['e'], 1)).T
    waste_code['createE']['E_n'] = waste_code['createE']['N_s'] * waste_code['createE']['Z']
    waste_code['createE']['E'] = (waste_code['createE']['E_n'] * waste_code['createE']['M_e_matrix']).T

    # 4) Complete Model
    waste_code['completeModel'] = {
        'M': {
            'mean_values': waste_code['createE']['E'] @ A_without_utilities
        }
    }

    waste_code['completeModel']['elementstoWaste'] = {
        'mean_values': np.clip(-waste_code['completeModel']['M']['mean_values'], a_min=0, a_max=None)
    }

    waste_code['completeModel']['tooManyOutputs'] = np.clip(waste_code['completeModel']['M']['mean_values'], a_min=0, a_max=None)

    zerorows = [i for i, row in enumerate(waste_code['completeModel']['tooManyOutputs']) if np.all(row == 0)]
    zerocols = [i for i in range(waste_code['completeModel']['tooManyOutputs'].shape[1]) if np.all(waste_code['completeModel']['tooManyOutputs'][:, i] == 0)]

    waste_code['completeModel']['tooManyOutputs'] = np.delete(np.delete(waste_code['completeModel']['tooManyOutputs'], zerorows, axis=0), zerocols, axis=1)

    waste_code['completeModel']['elements_tooManyOutputs'] = np.delete(waste_code['chemical_formula']['elements'], zerorows, axis=0)
    waste_code['completeModel']['processes_tooManyOutputs'] = np.delete(Model['meta_data_processes'][0][1:], zerocols)

    Matrix_tooManyOutputs = np.vstack((['elements/processes'] + list(waste_code['completeModel']['processes_tooManyOutputs']),
                                      np.hstack((np.array(waste_code['completeModel']['elements_tooManyOutputs']).reshape(-1, 1),
                                                waste_code['completeModel']['tooManyOutputs']))))

    # Overwrite raw
    raw_A = raw['rawA']
    raw_B = raw['rawB']

    delete_flows = [raw_A[i][0] for i in range(1, len(raw_A)) if raw_A[i][1] in ['unit', 'tkm']]

    raw_A = [row for row in raw_A if row[0] not in delete_flows]

    # Replace NaNs with zeros
    raw_A = [[0 if pd.isna(cell) else cell for cell in row] for row in raw_A]
    raw_B = [[0 if pd.isna(cell) else cell for cell in row] for row in raw_B]

    # Add elements (processes) if not existing in raw_A
    for element in waste_code['chemical_formula']['elements']:
        if element not in raw_A[0]:
            raw_A[0].append(element)
            for row in raw_A[1:]:
                row.append(0)

    # Add elements (processes) if not existing in raw_B
    for element in waste_code['chemical_formula']['elements']:
        if element not in raw_B[0]:
            raw_B[0].append(element)
            for row in raw_B[1:]:
                row.append(0)

    # Get direct emissions and utilities for waste treatment
    elements_to_waste = waste_code['completeModel']['elementstoWaste']['mean_values']
    elements_to_wastes_names = waste_code['chemical_formula']['elements']

    names_rawB = raw_B[0][4:]
    values_rawB = np.array([list(map(float, row[4:])) for row in raw_B[1:]])
    values_rawA = np.array([list(map(float, row[3:])) for row in raw_A[1:]])

    B_inc = np.zeros((values_rawB.shape[0], len(elements_to_wastes_names)))
    A_inc = np.zeros((values_rawA.shape[0], len(elements_to_wastes_names)))

    for i, element in enumerate(elements_to_wastes_names):
        for j, name in enumerate(names_rawB):
            if element == name:
                B_inc[:, i] = values_rawB[:, j]
                A_inc[:, i] = values_rawA[:, j]

    waste_code['completeModel']['EmissionsPerProcess'] = B_inc @ elements_to_waste
    waste_code['completeModel']['WasteUtilitiesPerProcess'] = A_inc @ elements_to_waste

    # Include flows in Model
    Model_waste = Model.copy()
    end_meta_data_flows = len(Model_waste['meta_data_flows'])

    delete = []
    for i in range(1, len(raw_A)):
        RowModel = [j for j, flow in enumerate(Model_waste['meta_data_flows'][1:]) if flow[0] == raw_A[i][0]]
        if RowModel:
            delete.append(i)
            Model_waste['matrices']['A']['mean_values'][RowModel[0], :] += waste_code['completeModel']['WasteUtilitiesPerProcess'][i - 1, :]

    raw_A = [raw_A[0]] + [row for i, row in enumerate(raw_A[1:], start=1) if i not in delete]
    waste_code['completeModel']['WasteUtilitiesPerProcess'] = np.delete(waste_code['completeModel']['WasteUtilitiesPerProcess'], delete, axis=0)

    Model_waste['meta_data_flows'].extend([[row[0], row[2], None, None, None, row[1]] for row in raw_A[1:]])

    Model_waste['matrices']['A']['mean_values'] = np.vstack((Model_waste['matrices']['A']['mean_values'], waste_code['completeModel']['WasteUtilitiesPerProcess']))

    # Handle B matrix and meta data elementary flow manipulation
    name_model = [flow[0] for flow in Model_waste['meta_data_elementary_flows'][1:]]
    medium_model = [flow[1] for flow in Model_waste['meta_data_elementary_flows'][1:]]
    location_model = [flow[2] for flow in Model_waste['meta_data_elementary_flows'][1:]]

    name_waste = [row[0] for row in raw_B[1:]]
    medium_waste = [row[1] for row in raw_B[1:]]
    location_waste = [row[2] for row in raw_B[1:]]

    for i, name_w in enumerate(name_waste):
        for j, name_m in enumerate(name_model):
            if name_w == name_m and medium_waste[i] == medium_model[j] and location_waste[i] == location_model[j]:
                Model_waste['matrices']['B']['mean_values'][j, :] += waste_code['completeModel']['EmissionsPerProcess'][i, :]
                break
        else:
            if not np.all(waste_code['completeModel']['EmissionsPerProcess'][i, :] == 0):
                raise ValueError(f"Elementary flow {name_w} from waste model not found in elementary flow list of defined ecoinvent version.")

    Model_waste['waste_code'] = waste_code
    Model_waste['waste_code']['Matrix_tooManyOutputs'] = Matrix_tooManyOutputs

    return Model_waste
