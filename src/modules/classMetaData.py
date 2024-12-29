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
    
# ============================================================================================================================
## CREATE LOGGER1 

def setup_logger(name='unified_logger', model_dir=None, mode='w'):
    """
    Create logger instance with hybrid behavior
    
    Args:
        name (str): Name of the logger instance. Defaults to 'unified_logger'
        model_dir (Path, optional): Directory to save log file. Defaults to None
        mode (str, optional): File mode ('w' for write, 'a' for append). Defaults to 'w'
    
    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    
    if model_dir:
        # File-only logging for training
        file_handler = logging.FileHandler(model_dir / 'training.log', mode=mode)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        # Console-only logging for initial steps
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger