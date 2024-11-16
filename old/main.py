# IMPORT OFFICIAL LIBRARIES
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, List
import logging
import pandas as pd


# IMPORT LIBRARIES
from classLoader import ESOImport, SMARDImport, Tidy 
from classTransfomer import Transformer, timeformat

# Setup logger
logging.basicConfig(level=logging.DEBUG, 
                    filename='logdocumentation.log', 
                    filemode='w',
                    format='%(asctime)s - %(levelno)s - %(lineno)d - %(module)s - %(message)s',
                    style='%') #\nStack Info:\n%(stack_info)s

# INITIALIZE CLASSES
# initalize ESO and SMARD loader with the base directory
esoimport = ESOImport(base_dir=Path(__file__).parent.parent / 'data')
smardimport = SMARDImport(base_dir=Path(__file__).parent.parent / 'data')
# initialization for tidy up and transform the read in data
tidy = Tidy()
trans = Transformer()


# IMPORT DATA (from different sub-directories)
# from the national grid 'Electricity System Operator (ESO)' of the United Kingdom https://www.nationalgrideso.com/

df_bc = esoimport.combine_eso('eso_daily-balancing-services-use-of-system-cost-data', add_source_file=True)
df_bc = tidy.standardize_time(df_bc, 'SETT_DATE')
df_bc = trans.manipulate_esolog(df_bc, 'SETT_DATE', 'SETT_PERIOD')
df_bc = trans.set_time_span(df_bc)
tidy.df_info(df_bc)

df_bv = esoimport.combine_eso('eso_daily-balancing-services-use-of-system-volume-data', add_source_file=True)
df_bv = tidy.standardize_time(df_bv, 'SETT_DATE')
df_bv = trans.manipulate_esolog(df_bv, 'SETT_DATE', 'SETT_PERIOD')
df_bv = trans.set_time_span(df_bv)
tidy.df_info(df_bv) 

df_dd = esoimport.combine_eso('eso_historic-demand-data', add_source_file=True)
df_dd = tidy.standardize_time(df_dd, 'SETTLEMENT_DATE')
df_dd = trans.manipulate_esolog(df_dd, 'SETTLEMENT_DATE', 'SETTLEMENT_PERIOD')
df_dd = trans.set_time_span(df_dd)
tidy.df_info(df_dd) 

df_ci = esoimport.combine_eso('eso_historic-generation-mix-and-carbon-intensity', add_source_file=True)
df_ci = tidy.standardize_time(df_ci, 'DATETIME')
df_ci = trans.manipulate_esolog(df_ci, 'DATETIME')
df_ci = trans.set_time_span(df_ci)
tidy.df_info(df_ci)

df_si = esoimport.combine_eso('eso_system-inertia', add_source_file=True)
df_si = tidy.standardize_time(df_si, 'Settlement Date')
df_si = trans.manipulate_esolog(df_si, 'Settlement Date', 'Settlement Period')
df_si = trans.set_time_span(df_si)
tidy.df_info(df_si) 


# from the 'Bundesnetzagentur' of Germany https://www.smard.de/home

df_gcf = smardimport.combine_smard('smard_market-data_generation+consumption+physical-power-flow_GER', pos=1)
df_gcf = tidy.standardize_time(df_gcf, 'Datum von')
df_gcf = trans.manipulate_smardlog(df_gcf, 'Datum von') #(315744, 40) --> (157872, 40)
df_gcf = trans.set_time_span(df_gcf)
tidy.df_info(df_gcf) 

df_ss = smardimport.combine_smard('smard_market-data_system-stability_GER')
df_ss = tidy.standardize_time(df_ss, 'Datum von')
df_ss = trans.manipulate_smardlog(df_ss, 'Datum von') #(315744, 29) --> (157872, 29) 
df_ss = trans.set_time_span(df_ss)
tidy.df_info(df_ss) 


df_dict = {
    'df_bc': df_bc,
    'df_bv': df_bv,
    'df_dd': df_dd,
    'df_ci': df_ci,
    'df_si': df_si,
    'df_gcf': df_gcf,
    'df_ss': df_ss
}
trans.analyze_df_mismatches(df_dict)
