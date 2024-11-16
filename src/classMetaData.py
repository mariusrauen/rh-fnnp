from dataclasses import dataclass
from typing import Optional, Dict
import logging

@dataclass
class DatasetConfig:
    """Configuration for a single dataset"""
    key: str
    filename: str
    date_column: str
    period_column: Optional[str] = None
    position: Optional[int] = None
    source: str = 'eso'  # 'eso' or 'smard'


class DatasetRegistry:
    """Registry of all dataset configurations"""
    
    @staticmethod
    def get_configs() -> Dict[str, DatasetConfig]:
        logging.info(f'Get meta data')
        return {
            # ESO datasets
            'df_bc': DatasetConfig(
                key='df_bc',
                filename='eso_daily-balancing-services-use-of-system-cost-data',
                date_column='SETT_DATE',
                period_column='SETT_PERIOD'
            ),
            'df_bv': DatasetConfig(
                key='df_bv',
                filename='eso_daily-balancing-services-use-of-system-volume-data',
                date_column='SETT_DATE',
                period_column='SETT_PERIOD'
            ),
            'df_dd': DatasetConfig(
                key='df_dd',
                filename='eso_historic-demand-data',
                date_column='SETTLEMENT_DATE',
                period_column='SETTLEMENT_PERIOD'
            ),
            'df_ci': DatasetConfig(
                key='df_ci',
                filename='eso_historic-generation-mix-and-carbon-intensity',
                date_column='DATETIME'
            ),
            'df_si': DatasetConfig(
                key='df_si',
                filename='eso_system-inertia',
                date_column='Settlement Date',
                period_column='Settlement Period'
            ),
            # SMARD datasets
            'df_gcf': DatasetConfig(
                key='df_gcf',
                filename='smard_market-data_generation+consumption+physical-power-flow_GER',
                date_column='Datum von',
                position=1,
                source='smard'
            ),
            'df_ss': DatasetConfig(
                key='df_ss',
                filename='smard_market-data_system-stability_GER',
                date_column='Datum von',
                source='smard'
            )
        }