from dataclasses import dataclass
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
from datetime import datetime
from io import StringIO

# VISUALIZE COMPARABLE FEATURES OF THE WHOLE DATASET

# Define class to visualize
@dataclass
class Visualizer:
    df_eso: pd.DataFrame
    df_ger: pd.DataFrame

    def __post_init__(self):
        
        # Define the full path
        self.plot_dir = Path(__file__).resolve().parent.parent / '..' /'data' / 'processed' / 'plots'
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        
        #Carbon Intensity gCO2/kWh
        self.carbon_intensity = {
            'GAS': self.df_eso['GAS'],
            'COAL': self.df_eso['COAL'],
            'FOSSIL': self.df_eso['GAS'] + self.df_eso['COAL'],
            'NUCLEAR': self.df_eso['NUCLEAR'],
            'RENEW': self.df_eso['WIND'] + self.df_eso['HYDRO'] + self.df_eso['BIOMASS'] + self.df_eso['SOLAR'],
            'PUMP_STORAGE': self.df_eso['STORAGE']
        }

        self.eso_demand = {
            'OVERALL': self.df_eso['ENGLAND_WALES_DEMAND'],
            'PUMP_STORAGE': self.df_eso['PUMP_STORAGE_PUMPING']
        }

        self.eso_balancing = {
            'Energy Imbalance': self.df_eso['Energy Imbalance (MWh)'],
            'Frequency Control': self.df_eso['Frequency Control Offers (MWh)'] - self.df_eso['Frequency Control Bids (MWh)'],
            'Positive Reserve': self.df_eso['Positive Reserve (MWh)'],
            'Negative Reserve': self.df_eso['Negative Reserve (MWh)']
        }


        self.ger_power_gen = {
            'GAS': self.df_ger['Erzeugung_Erdgas [MWh]'],
            'SCOAL': self.df_ger['Erzeugung_Steinkohle [MWh]'],
            'BCOAL': self.df_ger['Erzeugung_Braunkohle [MWh]'],
            'FOSSIL': self.df_ger['Erzeugung_Erdgas [MWh]'] + self.df_ger['Erzeugung_Steinkohle [MWh]'] + self.df_ger['Erzeugung_Braunkohle [MWh]'],
            'NUCLEAR': self.df_ger['Erzeugung_Kernenergie [MWh]'],
            'RENEW': self.df_ger['Erzeugung_Wind Offshore [MWh]'] + self.df_ger['Erzeugung_Wind Onshore [MWh]'] +
                     self.df_ger['Erzeugung_Wasserkraft [MWh]'] + self.df_ger['Erzeugung_Biomasse [MWh]'] +
                     self.df_ger['Erzeugung_Photovoltaik [MWh]'],
            'PUMP_STORAGE': self.df_ger['Erzeugung_Pumpspeicher [MWh]']
        }

        self.ger_demand = {
            'OVERALL': self.df_ger['Stromverbrauch_Gesamt (Netzlast) [MWh]'],
            'PUMP_STORAGE': self.df_ger['Stromverbrauch_Pumpspeicher [MWh]']
        }

        self.ger_balancing = {
            'Energy Imbalance': self.df_ger['Ausgleichsenergie_Volumen (+) [MWh]'] - self.df_ger['Ausgleichsenergie_Volumen (-) [MWh]'],
            'Frequency Control': self.df_ger['Sekund_Abgerufene Menge (+) [MWh]'] - self.df_ger['Sekund_Abgerufene Menge (-) [MWh]'],
            'Positive Reserve': self.df_ger['Minutenreserve_Abgerufene Menge (+) [MWh]'],
            'Negative Reserve': self.df_ger['Minutenreserve_Abgerufene Menge (-) [MWh]']
        }

    def normalize(self, data):
        """Normalize data to the range [0, 1]."""
        return (data - data.min()) / (data.max() - data.min())

    def plot_data(self, n=100):
        """Generate the plots for power_gen, demand, and balancing."""
        time_axis_eso = self.df_eso['ID']
        time_axis_ger = self.df_ger['ID']

        fig, axes = plt.subplots(4, 2, figsize=(15, 20), sharex=False) # 5, 2
        fig.suptitle("Energy Data Visualization", fontsize=16)

        # Power generation
        """for i, category in enumerate(['FOSSIL', 'NUCLEAR']):
            ax = axes[0, i]
            ax.plot(time_axis_eso[::n], self.normalize(self.carbon_intensity[category])[::n], label=f"ESO {category}", color='blue')
            ax.plot(time_axis_ger[::n], self.normalize(self.ger_power_gen[category])[::n], label=f"GER {category}", color='orange')
            ax.set_title(f"Power Generation: {category}")
            ax.legend()
            ax.grid()"""

        for i, category in enumerate(['RENEW', 'PUMP_STORAGE']):
            ax = axes[0, i]
            ax.plot(time_axis_eso[::n], self.normalize(self.carbon_intensity[category])[::n], label=f"ESO {category}", color='blue')
            ax.plot(time_axis_ger[::n], self.normalize(self.ger_power_gen[category])[::n], label=f"GER {category}", color='orange')
            ax.set_title(f"Power Generation: {category}")
            ax.legend()
            ax.grid()

        # Demand
        for i, category in enumerate(self.eso_demand.keys()):
            ax = axes[1, i]
            ax.plot(time_axis_eso[::n], self.normalize(self.eso_demand[category])[::n], label=f"ESO {category}", color='blue')
            ax.plot(time_axis_ger[::n], self.normalize(self.ger_demand[category])[::n], label=f"GER {category}", color='orange')
            ax.set_title(f"Demand: {category}")
            ax.legend()
            ax.grid()

        # Balancing - Energy Imbalance and Frequency Control
        for i, category in enumerate(['Energy Imbalance', 'Frequency Control']):
            ax = axes[2, i]
            ax.plot(time_axis_eso[::n], self.normalize(self.eso_balancing[category])[::n], label=f"ESO {category}", color='blue')
            ax.plot(time_axis_ger[::n], self.normalize(self.ger_balancing[category])[::n], label=f"GER {category}", color='orange')
            ax.set_title(f"Balancing: {category}")
            ax.legend()
            ax.grid()

        # Balancing - Positive and Negative Reserve
        for i, category in enumerate(['Positive Reserve', 'Negative Reserve']):
            ax = axes[3, i]
            ax.plot(time_axis_eso[::n], self.normalize(self.eso_balancing[category])[::n], label=f"ESO {category}", color='blue')
            ax.plot(time_axis_ger[::n], self.normalize(self.ger_balancing[category])[::n], label=f"GER {category}", color='orange')
            ax.set_title(f"Balancing: {category}")
            ax.legend()
            ax.grid()

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save the plot
        plot_path = self.plot_dir / 'energy_data_visualization.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_correlation_heatmap(self, features_eso, features_ger, figsize=(12, 10)):
        """Plot and save correlation heatmaps."""
        # ESO correlation heatmap
        plt.figure(figsize=figsize)
        correlation_matrix = self.df_eso[features_eso].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.3f')
        plt.title('Feature Correlation Heatmap - ESO')
        plt.xticks(rotation=45, ha='right')
        plt.yticks()
        plt.tight_layout()
        
        # Save ESO heatmap
        eso_heatmap_path = self.plot_dir / 'eso_correlation_heatmap.png'
        plt.savefig(eso_heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()

        # GER correlation heatmap
        plt.figure(figsize=figsize)
        correlation_matrix = self.df_ger[features_ger].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('Feature Correlation Heatmap - GER')
        plt.xticks(rotation=45, ha='right')
        plt.yticks()
        plt.tight_layout()
        
        # Save GER heatmap
        ger_heatmap_path = self.plot_dir / 'ger_correlation_heatmap.png'
        plt.savefig(ger_heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()

# ============================================================================================================================
## INSPECT THE DATA READ IN AFTER DATA PREPERATION

# Define function for inspection
def inspect_data(df, output_path=None):

    # Set display options
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    
    # Prepare the information
    info_text = []
    info_text.append(f"Generated on: {datetime.now()}")
    info_text.append(f"Shape: {df.shape}")
    
    # Get info in string format
    buffer = StringIO()
    df.info(buf=buffer)
    info_text.append(f"\nDataFrame Info:")
    info_text.append(buffer.getvalue())

    # Add first 5 rows
    buffer_head = StringIO()
    df.head().to_string(buf=buffer_head)
    info_text.append(f"\nFirst 5 rows:")
    info_text.append(buffer_head.getvalue())

    # Add last 5 rows
    buffer_tail = StringIO()
    df.tail().to_string(buf=buffer_tail)
    info_text.append(f"\nLast 5 rows:")
    info_text.append(buffer_tail.getvalue())
    
    # Join all information
    full_report = '\n'.join(info_text)
    output_path = Path(output_path)
    output_path.write_text(full_report)
    
    return full_report

# ============================================================================================================================

def find_high_correlations(df, threshold, output_path=None):
    '''Finds and saves highly correlated feature pairs'''
    # Get correlation matrix without ID column
    corr_matrix = df.drop('ID', axis=1, errors='ignore').corr()
    
    # Store high correlations
    high_corr = {}
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr = corr_matrix.iloc[i, j]
            if threshold <= abs(corr) < 1.0:
                feat1, feat2 = corr_matrix.columns[i], corr_matrix.columns[j]
                if feat1.split('.')[0] != feat2.split('.')[0]:
                    high_corr[(feat1, feat2)] = round(corr, 3)
    
    # Sort and save results
    sorted_corr = dict(sorted(high_corr.items(), key=lambda x: abs(x[1]), reverse=True))
    if output_path:
        with open(output_path, 'w') as f:
            f.write(f"High Correlations (threshold >= {threshold}):\n")
            for (f1, f2), corr in sorted_corr.items():
                f.write(f"{f1} - {f2}: {corr}\n")
                
    return sorted_corr