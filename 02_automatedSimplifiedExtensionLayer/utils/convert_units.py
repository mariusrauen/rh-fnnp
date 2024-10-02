


# This is used to convert the units of the streams given. 
def convert_units(streams: list[Stream]) -> None:
    for stream in streams:
        # First convert the mass amount unit
        match streams.amount_unit:
            case 'TONNE':
                stream.amount = stream.amount*1E3
                stream.amount_unit = 'kg'
                stream.unit_type = 'Mass'
            case 'G':
                stream.amount = stream.amount/1E3
                stream.amount_unit = 'kg'
                stream.unit_type = 'Mass'
            case 'M3':
                stream.amount_unit = 'Nm3'
                stream.unit_type = 'Volumen'
            case 'MNM3':
                streams.amount_unit = 'Nm3'
                streams.amount = stream.amount*1e3
                streams.unit_type = 'Volumen'
            case 'NM3':
                stream.amount_unit = 'Nm3'
                stream.unit_type = 'Volumen'
            case 'KWH':
                stream.amount = stream.amount*3.6 # convert KWH to MJ
                stream.amount_unit = 'MJ'
                stream.unit_type = 'Energy'
            case 'EA':
                stream.amount = stream.amount # Unit is unknown does not appear in IHS Documentation
                stream.amount_unit = 'pcs'
                stream.unit_type = 'Pieces'
            case 'MMCAL':
                stream.amount = stream.amount*4.187 # convert MMCAL to MJ
                stream.amount_unit = 'MJ'
                stream.unit_type = 'Energy'
            case pd.isnan(stream.cost_unit):
                    pass
            case _:
                logging.info(f'Amount Unit unknown: {stream.amount_unit}')
                stream.unit_type = 'Unknowen'
        # Now alter the cost_unit of the Streams
        match stream.cost_unit:
             case '¢/KG':
                stream.cost_unit = '$/KG'
                stream.cost = stream.cost/100
             case '¢/G':
                stream.cost_unit = '$/KG'
                stream.cost = stream.cost/100*1000
             case '¢/kg':
                stream.cost_unit = '$/KG'
             case '¢/EA':
                stream.cost_unit = '$/EA'
                stream.cost = stream.cost/100
             case '¢/TONNE':
                stream.cost_unit = '$/KG'
                stream.cost = stream.cost/1000/100
             case '¢/M3':
                stream.cost_unit = '$/NM3'
                stream.cost = stream.cost/100
             case '¢/NM3':
                stream.cost_unit = '$/NM3'
                stream.cost = stream.cost/100
             case '¢/KWH':
                stream.cost_unit = '$/MJ'
                stream.cost = stream.cost/3.6/100
             case '¢/MMCAL':
                stream.cost_unit = '$/MJ'
                stream.cost = stream.cost/4.187/100
             case pd.isnan(stream.cost_unit):
                pass
             case _:
                logging.info(f'Cost Unit unknown: {stream.cost_unit}')
    ## set the main stream to 1
    norm_factor = 1 / streams[0].amount
    for stream in streams:
        stream.amount = stream.amount * norm_factor
