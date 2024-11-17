# README INTRODUCTION
This README contains information regarding the features used in this data set.
For this Data Science task, the Hadley-Wickham method is used to investigate the data provided by the 'Electricity System Operator' (ESO) and by 'Strom- und Gasmarktdaten' (SMARD) of the German 'Bundesnetzagentur. The method steps of Hadley-Wickham are shown in the image below.

Sources of the raw data (2024/09/28):
https://www.neso.energy/, 
https://www.smard.de/home

![Dashboard Screenshot](./data/figures/Hadley-Wickham-method.png)



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



# DataPerparation.py
Contains a **DataProcessor** class combining the code of Import, Tidy and Transformer to provide prepare data for the tasks in the subsequent steps of the Hadley-Wickham method.
#### Methods:
    - __post_init__
    - process_eso_data
    - process_smard_data
    - process_all_data
    - save_dataframes_to_csv
    - main



# **NOTE**
> **Up this point, the project is processed with python scripts (.py). Further model tasks Visualization, Modelling, Communication will be handled with Jupyter Notebooks (.ipnyb). Information and results are saved to the 'processed' directory.**
 



# classVisualizer.ipnyb



# classModeler.ipnyb



# classCommunicator.ipnyb


# Timetrack
Target: 6CP = 6 x 25h = 150h

Lecture: CW37 until CW01 = 16 weeks with 3h/week = 48h

Project:

| Date       | Start      | End        | Break      | h [day]   | h [sum]   | Comment                           |
|------------|------------|------------|------------|-----------|-----------|-----------------------------------|
| 2024/09/28 | 08:00      | 14:00      | 01:00      | 05:00     | 05:00     | Init Project, seach data          |
| 2024/10/05 | 09:00      | 14:00      | 01:00      | 04:00     | 09:00     | Inform about Docker               |
| 2024/10/12 | 09:00      | 15:00      | 01:00      | 06:00     | 15:00     | Setup Project, implement Docker   |
| 2024/10/19 | 08:00      | 19:00      | 01:00      | 10:00     | 25:00     | First read in, inital strucutre   |
| 2024/10/31 | 14:00      | 17:00      | 00:00      | 03:00     | 28:00     | Setup git                         | 
| 2024/11/01 | 09:00      | 19:00      | 04:00      | 06:00     | 34:00     | Work on Importer                  |   
| 2024/11/03 | 08:00      | 18:00      | 01:00      | 09:00     | 43:00     | Work on Tidy, Transformer         | 
| 2024/11/09 | 07:00      | 15:00      | 01:00      | 07:00     | 50:00     | Work on Tidy, Transformer         |
| 2024/11/10 | 09:00      | 12:00      | 00:00      | 03:00     | 53:00     | Work on Tidy, Transformer         |
| 2024/11/15 | 08:00      | 14:00      | 02:00      | 04:00     | 57:00     | Restructe code                    |
| **End of Data Preperation**                                                                                   |
| 2024/11/16 | 09:00      | 18:00      | 02:00      | 07:00     | 64:00     | Setup notebooks, validate code    |
| 2024/11/21 | 00:00      | 00:00      | 00:00      | 00:00     | 64:00     |
| 2024/11/23 | 00:00      | 00:00      | 00:00      | 00:00     | 64:00     |
| 2024/11/30 | 00:00      | 00:00      | 00:00      | 00:00     | 64:00     |
| 2024/12/07 | 00:00      | 00:00      | 00:00      | 00:00     | 64:00     |
| 2024/12/14 | 00:00      | 00:00      | 00:00      | 00:00     | 64:00     |
| 2024/12/21 | 00:00      | 00:00      | 00:00      | 00:00     | 64:00     |
| 2024/12/28 | 00:00      | 00:00      | 00:00      | 00:00     | 64:00     |
| **End of Project**                                                        |