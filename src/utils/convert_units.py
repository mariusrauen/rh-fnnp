from utils.cm_classes import Stream
import pandas as pd
import logging

# This function is used to convert the units of the streams provided.
def convert_units(streams: list[Stream]) -> None:
    for stream in streams:
        # Convert the mass amount unit
        match stream.amount_unit:
            case 'TONNE':
                stream.amount *= 1E3
                stream.amount_unit = 'kg'
                stream.unit_type = 'Mass'
            case 'G':
                stream.amount /= 1E3
                stream.amount_unit = 'kg'
                stream.unit_type = 'Mass'
            case 'M3':
                stream.amount_unit = 'Nm3'
                stream.unit_type = 'Volume'
            case 'MNM3':
                stream.amount_unit = 'Nm3'
                stream.amount *= 1E3
                stream.unit_type = 'Volume'
            case 'NM3':
                stream.amount_unit = 'Nm3'
                stream.unit_type = 'Volume'
            case 'KWH':
                stream.amount *= 3.6  # Convert KWH to MJ
                stream.amount_unit = 'MJ'
                stream.unit_type = 'Energy'
            case 'EA':
                # EA (each) is not converted but categorized
                stream.amount = stream.amount
                stream.amount_unit = 'pcs'
                stream.unit_type = 'Pieces'
            case 'MMCAL':
                stream.amount *= 4.184  # Convert MMCAL to MJ
                stream.amount_unit = 'MJ'
                stream.unit_type = 'Energy'
            case _ if pd.isna(stream.amount_unit):
                pass  # Handle missing amount units
            case _:
                logging.info(f'Unknown amount unit: {stream.amount_unit}')
                stream.unit_type = 'Unknown'

        # Now convert the cost_unit of the Streams
        match stream.cost_unit:
            case '¢/KG':
                stream.cost_unit = '$/KG'
                stream.cost /= 100
            case '¢/G':
                stream.cost_unit = '$/KG'
                stream.cost = (stream.cost / 100) * 1000
            case '$/kg':
                stream.cost_unit = '$/KG'
            case '¢/EA':
                stream.cost_unit = '$/EA'
                stream.cost /= 100
            case '¢/TONNE':
                stream.cost_unit = '$/KG'
                stream.cost /= 1000 / 100
            case '¢/M3':
                stream.cost_unit = '$/NM3'
                stream.cost /= 100
            case '¢/NM3':
                stream.cost_unit = '$/NM3'
                stream.cost /= 100
            case '¢/KWH':
                stream.cost_unit = '$/MJ'
                stream.cost /= 3.6 / 100
            case '¢/MMCAL':
                stream.cost_unit = '$/MJ'
                stream.cost /= 4.184 / 100
            case _ if pd.isna(stream.cost_unit):
                pass  # Handle missing cost units
            case _:
                logging.info(f'Unknown cost unit: {stream.cost_unit}')

    # Normalize the main stream's amount to 1
    norm_factor = 1 / streams[0].amount
    for stream in streams:
        stream.amount *= norm_factor
