# **RENDER: ctrl+shift+V**

# **INTRODUCTION** 
Original Author: marius.rauen@rfh-campus.de

#### Render with Ctrl + Shift + V

This README contains information regarding the project structure and the processed data (Introduction), and the software exectuion (Instruction).
For this Data Science task, the Hadley-Wickham method is used to investigate the data provided by the 'Electricity System Operator' (ESO) and by 'Strom- und Gasmarktdaten' (SMARD) of the German 'Bundesnetzagentur. The method steps of Hadley-Wickham are shown in the image below.

#### AIM:
To be filled ... 


Sources of the raw data (2024/09/28):
| Name                                                      | Source                                                                                    | 
|-----------------------------------------------------------|-------------------------------------------------------------------------------------------|  
| **National Energy System Operator (ESO) (UK)**            | https://www.neso.energy/                                                                  |                                           
| Daily Balancing Services Use of System (BSUoS) Cost Data  | https://www.neso.energy/data-portal/daily-balancing-costs-balancing-services-use-system   |
| Daily Balancing Services Use of System (BSUoS) Volume Data| https://www.neso.energy/data-portal/daily-balancing-volume-balancing-services-use-system  |
| Historic Demand Data                                      | https://www.neso.energy/data-portal/historic-demand-data                                  |
| Historic generation mix and carbon intensity              | https://www.neso.energy/data-portal/historic-generation-mix                               |
| System Inertia                                            | https://www.neso.energy/data-portal/system-inertia                                        |
| **Strom- und Gasmarkdaten (SMARD) (GER)**                 | https://www.smard.de/home/downloadcenter/download-marktdaten/                             |
| Physikalischer Stromfluss                                 | Link -> Oberkategorie: Markt                                                              |
| Realisierte Erzeugung                                     | Link -> Oberkategorie: Stromerzeugung                                                     |
| Realisierter Stromverbrauch                               | Link -> Oberkategorie: Stromverbrauch                                                     |
| Ausgleichsenergie                                         | Link -> Oberkategorie: Systemstabilitaet                                                  |
| Exportierte Regelenergie                                  | Link -> Oberkategorie: Systemstabilitaet                                                  |
| Importierte Regelenergie                                  | Link -> Oberkategorie: Systemstabilitaet                                                  |
| Kosten                                                    | Link -> Oberkategorie: Systemstabilitaet                                                  |
| Minutenreserve                                            | Link -> Oberkategorie: Systemstabilitaet                                                  |
| Primaerregelreserve                                       | Link -> Oberkategorie: Systemstabilitaet                                                  |
| Sekundaerregelreserve                                     | Link -> Oberkategorie: Systemstabilitaet                                                 \ |


![Hadley-Wickham-method](./data/figures/Hadley-Wickham-method.png)


# **INSTRUCTION**
> **Get project from https://github.com/mariusrauen/rh-fnnp and connect Docker environment.** 

> **Run DataPrep.py to generate the data for the steps Visualise, Model and Communicate.**

> **After that, execute the jupyter notebooks for Visualise, Communicate and Model in the order as listed in this README.**

# classImporter.py
Contains the **Importer*** class to import data from national grid 'Electricity System Operator' (ESO) of Great Britain (GB) and from 'Bundesnetzagentur' in Germany.
#### Methods:
    - combine_eso
    - combine_smard



# classTidy.py
Contains the **Tidy** class for an initial data structuring by printing information of the raw data and standardizing the time stamps for further processing
#### Methods:
    - df_info
    - standardize_time



# classTransformer.py
Contains the **Transformer** class and its methods to transform the read in raw data into a format that is suitable for a regression task. 
#### Methods:
    - manipulate_esolog
    - manipulate_smardlog
    - set_time_span
    - analyze_df_mismatches
    - align_dataframes
    - prepare_for_regression
    - merge_dataframes

# DataPerp.py
Contains a **DataProcessor** class combining the code of Import, Tidy and Transformer to provide prepare data for the tasks in the subsequent steps of the Hadley-Wickham method.
#### Methods:
    - __post_init__
    - process_eso_data
    - process_smard_data
    - process_all_data
    - save_dataframes_to_csv
    - main



# **NOTE**
> **Up this point, the project steps Import, Tidy and Transform are processed with python scripts (.py). Further model steps Visualization, Modelling and Communication are handled in VisModCom.py. Information related to the used and processed datasets are saved to the directory that is named 'processed'. Information and results regarding the models are saved to the directory that is named 'models'.**
 


# VisModCom.py
## Visualization
In this project step, the prepared data are inspected.

## Modeler
Build a ML regression and a LSTM.

## Communicator

The Communicator is a framework to present insights in an effective and clear way. It reports and visualizes the output of a data science project.

# **TIME**
Target: 6CP = 6 x 25h = 150h

Lecture: CW37 (2024/09/09) until CW52 (2024/12/23) = 16 weeks with 3h/week = 48h - 6h (canceled lectures) = 42h

Project:

| Date       | CW        | Start      | End        | Break      | h [day]   | h [sum]   | Comment                            | 
|------------|-----------|------------|------------|------------|-----------|-----------|------------------------------------|                  
| 2024/09/28 | 39        | 08:00      | 14:00      | 01:00      | 05:00     | 005:00     | Init Project, seach data          |
| 2024/10/05 | 40        | 09:00      | 14:00      | 01:00      | 04:00     | 009:00     | Inform about Docker               |
| 2024/10/12 | 41        | 09:00      | 15:00      | 01:00      | 06:00     | 015:00     | Setup Project, implement Docker   |
| 2024/10/19 | 42        | 08:00      | 19:00      | 01:00      | 10:00     | 025:00     | First read in, inital strucutre   |
| 2024/10/31 | 44        | 14:00      | 17:00      | 00:00      | 03:00     | 028:00     | Setup git                         |
| 2024/11/01 | 44        | 09:00      | 19:00      | 04:00      | 06:00     | 034:00     | Work on Importer                  |  
| 2024/11/03 | 44        | 08:00      | 18:00      | 01:00      | 09:00     | 043:00     | Work on Tidy, Transformer         |
| 2024/11/09 | 45        | 07:00      | 15:00      | 01:00      | 07:00     | 050:00     | Work on Tidy, Transformer         |
| 2024/11/11 | 45        | 18:30      | 21:30      | 00:00      | 03:00     | 053:00     | Work on Tidy, Transformer         |
| 2024/11/15 | 46        | 08:00      | 14:00      | 02:00      | 04:00     | 057:00     | Restructure code                  |
| **End Data Preperation**                                                                                                   |
| 2024/11/16 | 46        | 09:00      | 18:00      | 02:00      | 07:00     | 064:00     | Setup notebooks, validate code    |
| 2024/11/19 | 47        | 19:00      | 21:00      | 00:00      | 02:00     | 066:00     | Visualize                         |
| 2024/11/20 | 47        | 09:00      | 14:00      | 00:00      | 05:00     | 071:00     | Normalize and heatmap             |
| 2024/11/21 | 47        | 15:30      | 18:00      | 00:00      | 02:30     | 073:30     | Start modeling                    |
| 2024/11/22 | 47        | 09:00      | 14:00      | 03:00      | 04:30     | 078:00     | LSTM initial model structure      |
| 2024/11/23 | 47        | 09:00      | 12:00      | 00:00      | 03:00     | 081:00     | LSTM model                        |
| 2024/11/24 | 47        | 13:00      | 16:00      | 00:00      | 03:00     | 084:00     | LSTM model                        |
| 2024/11/27 | 48        | 09:00      | 16:00      | 01:00      | 06:00     | 090:00     | LSTM model                        |
| 2024/11/28 | 48        | 08:00      | 21:00      | 02:00      | 11:00     | 101:00     | LSTM model                        |
| 2024/11/29 | 48        | 08:00      | 10:00      | 00:00      | 02:00     | 103:00     | LSTM model                        |
| 2024/11/30 | 48        | 09:00      | 19:30      | 01:30      | 09:00     | 112:00     | Toss .ipynb, troubleshooting      |
| 2024/12/01 | 48        | 10:00      | 00:00      | 00:00      | 00:00     | 112:00     | Evaluate, restructure             |
| 2024/12/02 | 49        | 00:00      | 00:00      | 00:00      | 00:00     | XXX:00     | 
| 2024/12/XX | XX        | 00:00      | 00:00      | 00:00      | 00:00     | XXX:00     | 
| 2024/12/XX | XX        | 00:00      | 00:00      | 00:00      | 00:00     | XXX:00     | 
| **End of Project**                                                                                                         |