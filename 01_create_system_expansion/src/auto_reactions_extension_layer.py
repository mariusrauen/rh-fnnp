#%%
import pandas as pd
import numpy as np
import os
import xlsxwriter
import math 

#%%
# FUNCTIONS

def xlookup(lookup_value, lookup_array, return_array, if_not_found:str = ''):
    match_value = return_array.loc[lookup_array == lookup_value]
    if match_value.empty:
        return f'"{lookup_value}" not found!' if if_not_found == '' else if_not_found

    else:
        return match_value.tolist()[0]

#%%
# Excel from chemist and meta_data_flows in same subdir 

path = "reactions_extension_layer_raw.xlsx"
new_reactions = pd.read_excel(path, sheet_name=['raw_material_id','reference'])
meta_data = pd.read_excel('meta_data_flows.xlsx', sheet_name=['Tabelle1'])
md = meta_data['Tabelle1']

#%%
# REFERENCE 
# PROCESS_ID
# processes & stochiometric coefficients

procData = []
numReactions = (len(new_reactions['reference']))
materialLoc = [4,6,8,10,12,14,16,18,20,22]
index=0 

for i in range(numReactions):
    # column E  assumed (raw  material 1)
    new_reactions['reference'].iloc[i,1] = 'reaction of '+new_reactions['reference'].iloc[i,4]
    new_reactions['reference'].iloc[i,5] = '-'+str(new_reactions['reference'].iloc[i,5])
    
    # check entries column G (raw  material 2)
    if pd.isna(new_reactions['reference'].iloc[i,6]):
        pass
    else:
        new_reactions['reference'].iloc[i,1] = 'reaction of '+new_reactions['reference'].iloc[i,4] +' and '+ new_reactions['reference'].iloc[i,6]
        new_reactions['reference'].iloc[i,7] = '-'+str(new_reactions['reference'].iloc[i,7])
    
    # check entries column I (raw material 3)
    if pd.isna(new_reactions['reference'].iloc[i,8]):
        pass
    else:
        new_reactions['reference'].iloc[i,1] = 'reaction of '+str(new_reactions['reference'].iloc[i,4])+ \
            ', '+str(new_reactions['reference'].iloc[i,6])+' and '+str(new_reactions['reference'].iloc[i,8])
        new_reactions['reference'].iloc[i,9] = '-'+str(new_reactions['reference'].iloc[i,9])
    # check entries column K (raw material 4)
    if pd.isna(new_reactions['reference'].iloc[i,10]):
        pass
    else:
        new_reactions['reference'].iloc[i,1] = 'reaction of '+str(new_reactions['reference'].iloc[i,4])+\
            ', '+str(new_reactions['reference'].iloc[i,6])+', '+str(new_reactions['reference'].iloc[i,8])+\
                ' and '+str(new_reactions['reference'].iloc[i,10])
        new_reactions['reference'].iloc[i,11] = '-'+str(new_reactions['reference'].iloc[i,11])
    # check entries column  M (raw material 5)
    if pd.isna(new_reactions['reference'].iloc[i,12]):
        pass
    else:
        new_reactions['reference'].iloc[i,1] = 'reaction of '+str(new_reactions['reference'].iloc[i,4])+\
            ', '+str(new_reactions['reference'].iloc[i,6])+', '+str(new_reactions['reference'].iloc[i,8])+\
                ', '+str(new_reactions['reference'].iloc[i,10])+' and '+str(new_reactions['reference'].iloc[i,12])
        new_reactions['reference'].iloc[i,13] = '-'+str(new_reactions['reference'].iloc[i,13])
    
    currentReaction = new_reactions['reference'].iloc[i,1]
    currentMainFlow = new_reactions['reference'].iloc[i,2]
    procData.append([index, currentReaction, currentMainFlow])
    index +=1
    
proc = pd.DataFrame(procData, columns=['process_id','process','main flow']) 

# all species in columns B, C, D
exceptionList =['water', 'carbon monoxide']
for i in range(numReactions):
    for j in materialLoc:
        # all raw materials and co products
        if pd.isna(new_reactions['reference'].iloc[i,j]) or new_reactions['reference'].iloc[i,j] in exceptionList:
            pass
        else:
            new_reactions['reference'].loc[-1,'process'] = new_reactions['reference'].iloc[i,1]
            new_reactions['reference'].iloc[-1,2] = new_reactions['reference'].iloc[i,j]
            new_reactions['reference'].iloc[-1,3] = new_reactions['reference'].iloc[i,j+1]
            new_reactions['reference'] = new_reactions['reference'].reset_index(drop=True)
    ref = new_reactions['reference']
 
#%%
# RAW_MATERIAL_ID 

rm = new_reactions['raw_material_id'] 
uniqueMatList = []
c1=0
for column in rm['raw materials']:
    uniqueMatList.append(rm['raw materials'].values[c1])
    c1+=1
h=0
for column in ref['main flow']:
    
    if ref['main flow'].values[h] in uniqueMatList:
        h+=1
    else:
        uniqueMatList.append(ref['main flow'].values[h])
        currentSpecies = ref['main flow'].iloc[h]
        new_reactions['raw_material_id'].loc[-1,'raw materials'] = currentSpecies
        new_reactions['raw_material_id'].loc[-1,'raw_material_id'] =new_reactions['raw_material_id'].shape[0]-1 #...
        new_reactions['raw_material_id'].loc[-1,'CAS-Nr.'] = xlookup(currentSpecies, md['name'], md['CAS-Nr.'])
        new_reactions['raw_material_id'].loc[-1,'chemical formular'] = xlookup(currentSpecies, md['name'], md['chemical formular[C2C4 format]'])
        new_reactions['raw_material_id'].loc[-1,'molecular mass'] = xlookup(currentSpecies, md['name'], md['molecular mass'])
        new_reactions['raw_material_id'].loc[-1,'category'] = 1
        new_reactions['raw_material_id'].loc[-1,'[unit choice]'] = 'kg'
        new_reactions['raw_material_id'].loc[-1,'name'] = currentSpecies
        new_reactions['raw_material_id'].loc[-1,'SMILES'] = xlookup(currentSpecies, md['name'], md['SMILES CODE'])
        new_reactions['raw_material_id'].loc[-1,'comment'] ='AUTO GENERATED: CHECK'
        
        new_reactions['raw_material_id'] = new_reactions['raw_material_id'].reset_index(drop=True)
        h+=1
    rm = new_reactions['raw_material_id']
 
#%%
# MATRIX_TABLE

matData = []
matData.append([0,-1,-1.2])
matData.append([1,-1,-2])
mcount=0

for column in ref['main flow']:
    currentProcess = ref['process'].iloc[mcount]
    currentSpecies = ref['main flow'].iloc[mcount]
    rmid = xlookup(currentSpecies, rm['raw materials'], rm['raw_material_id'])
    pid = xlookup(currentProcess, proc['process'],proc['process_id'])
    coef = ref.iloc[mcount,3]
    matData.append([rmid, pid, coef]) 
    mcount+=1
    
mat = pd.DataFrame(matData, columns=['raw_material_id','process_id','coefficient'])  

#%%
# WRITING EXCEL FILE
dflist = [mat, rm, proc, ref]
sheet_names = ['matrix_table','raw_material_id','process_id','reference']
 
Excelwriter = pd.ExcelWriter('reactions_extension_layer_AUTO.xlsx',engine='xlsxwriter')

for i, df in enumerate (dflist):
    df.to_excel(Excelwriter, sheet_name=sheet_names[i],index=False)
Excelwriter.save()



 
  
