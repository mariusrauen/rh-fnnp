# ============================================================================================================================
## HEADER
# Module: Fuzzy Systems and Neural Networks  
# Name: Raun, Marius
# Matricle number: 131242002
# Contact: marius.rauen@rfh-campus.de

# Strucutre information:
# ! Familiarize yourself with the README file
# ! New sections are introduced with two hashtags, space and in capitol letters, e.g. ## HEADER
# ! Comments are introduced with one hashtag and space, e.g. # Base directory path for all data operations
# ! Inactive code is marked with one hastag and no space, e.g. #print()
# ============================================================================================================================



# ============================================================================================================================
## IMPORT LIBRARIES

import pandas as pd
from sklearn. model_selection import train_test_split
from sklearn. preprocessing import MinMaxScaler
from pathlib import Path
from datetime import timedelta

from modules.classVisualizer import Visualizer
from modules.classVisualizer import find_high_correlations, inspect_data
from modules.classMetaData import setup_logger

logger = setup_logger()
# ============================================================================================================================



# ============================================================================================================================
## READ IN FROM DATA PREPERATION
logger.info("READ IN FROM DATA PREPERATION")

data_path = Path(__file__).resolve().parent.parent / 'data' / 'processed' / 'DataPrep' # Set the data path

# Read in data
df_eso = pd.read_csv(data_path / 'eso' / 'df_eso.csv', delimiter=',', parse_dates=['ID']) #, parse_dates=['ID']
df_ger = pd.read_csv(data_path / 'ger' / 'df_ger.csv', delimiter=',', parse_dates=['ID']) #, parse_dates=['ID']

# Define the inspection directory paths
eso_dir = data_path / 'eso'; eso_dir.mkdir(parents=True, exist_ok=True)
ger_dir = data_path / 'ger'; ger_dir.mkdir(parents=True, exist_ok=True)

# CALL FUNCTION FOR INSPECTION 
_ = inspect_data(df_eso, output_path=Path(f'{data_path}/eso/df_eso.txt'))
_ = inspect_data(df_ger, output_path=Path(f'{data_path}/ger/df_ger.txt'))
# ============================================================================================================================



# ============================================================================================================================
## VISUALIZE COMPARABLE FEATURES
logger.info("VISUALIZE COMPARABLE FEATURES")

# Visualize over time
visualizer = Visualizer(df_eso, df_ger) # Create a visualizer object
visualizer.plot_data(n=1000) 

# Visualize correlation heat map of directly comparable features form ESO and GER
features_eso = ['GAS', 'COAL', 'NUCLEAR', 'WIND', 'HYDRO', 'BIOMASS', 'SOLAR', 'STORAGE',
               'Energy Imbalance', 'Frequency Control', 'Positive Reserve', 'Negative Reserve']

features_ger = ['Erzeugung_Erdgas [MWh]', 'Erzeugung_Steinkohle [MWh]', 'Erzeugung_Braunkohle [MWh]',
               'Erzeugung_Kernenergie [MWh]', 'Erzeugung_Wind Offshore [MWh]', 'Erzeugung_Wind Onshore [MWh]',
               'Erzeugung_Wasserkraft [MWh]', 'Erzeugung_Biomasse [MWh]', 'Erzeugung_Photovoltaik [MWh]',
               'Erzeugung_Pumpspeicher [MWh]',
               'Stromverbrauch_Gesamt (Netzlast) [MWh]', 'Stromverbrauch_Pumpspeicher [MWh]',
               'Ausgleichsenergie_Volumen (+) [MWh]', 'Ausgleichsenergie_Volumen (-) [MWh]',
               'Sekund_Abgerufene Menge (+) [MWh]', 'Sekund_Abgerufene Menge (-) [MWh]', 'Minutenreserve_Abgerufene Menge (+) [MWh]', 'Minutenreserve_Abgerufene Menge (-) [MWh]']

visualizer = Visualizer(df_eso, df_ger)
visualizer.plot_correlation_heatmap(features_eso, features_ger)
# ============================================================================================================================



# ============================================================================================================================
# SPLITTING DATA
logger.info('SPLITTING DATA')

# SET TARGET FEATURE
target_name = 'TARGET' # OVERALL DEMAND

# ESO
eso_combine_cols = df_eso['ENGLAND_WALES_DEMAND'] # + df_eso['XXX'] + df_eso['XXX'] + ...
df_eso[target_name] = eso_combine_cols 
eso_target = df_eso[target_name]
X_eso_dd = df_eso.loc[:, ~df_eso.columns.isin(eso_target + eso_combine_cols)]
y_eso_dd = eso_target

# Drop the target column explicitly
X_eso_dd = df_eso.drop(columns=[target_name])
X_train_eso, X_eval_eso, y_train_eso, y_eval_eso = train_test_split(X_eso_dd, y_eso_dd, test_size=0.25, random_state=42)

y_train_eso = pd.DataFrame({
    'ID': X_train_eso['ID'],
    target_name: y_train_eso.values
}).set_index(X_train_eso.index)

y_eval_eso = pd.DataFrame({
    'ID': X_eval_eso['ID'],
    target_name: y_eval_eso.values
}).set_index(X_eval_eso.index)

# ----------------------------------------------------------------------------------------------------------------------------
# GER

ger_combine_cols = df_ger['Stromverbrauch_Gesamt (Netzlast) [MWh]'] # + df_ger['XXX'] + df_ger['XXX'] + ...
df_ger[target_name] = ger_combine_cols # allocate the features to the new created column named target
ger_target = df_ger[target_name]
X_ger_dd = df_ger.loc[:, ~df_ger.columns.isin(ger_target + ger_combine_cols)]
y_ger_dd = ger_target

# Drop the target column explicitly
X_ger_dd = df_ger.drop(columns=[target_name])
X_train_ger, X_eval_ger, y_train_ger, y_eval_ger = train_test_split(X_ger_dd, y_ger_dd, test_size=0.25, random_state=42)

y_train_ger = pd.DataFrame({
    'ID': X_train_ger['ID'],
    target_name: y_train_ger.values
}).set_index(X_train_ger.index)

y_eval_ger = pd.DataFrame({
    'ID': X_eval_ger['ID'],
    target_name: y_eval_ger.values
}).set_index(X_eval_ger.index)

# ----------------------------------------------------------------------------------------------------------------------------

'''
pprint(X_train_eso.head().to_dict())
pprint(X_eval_eso.head().to_dict())
pprint(y_train_eso.head().to_dict())
pprint(y_eval_eso.head().to_dict())
'''
# ============================================================================================================================



# ============================================================================================================================
# SELECT A TARGET VALUE FOR SUBSEQUENT INDEX EVALUATION 
logger.info("SELECT A Target VALUE FOR SUBSEQUENT INDEX EVALUATION")
#target_timestamp = pd.to_datetime('2017-04-12 03:30:00')
target_timestamp = y_train_eso['ID'].iloc[-1]
logger.info(f"Using latest available timestamp: {target_timestamp}")
eso_row = y_train_eso[y_train_eso['ID'] == target_timestamp].iloc[0]
ger_row = y_train_ger[y_train_ger['ID'] == target_timestamp].iloc[0]
logger.info(f"ESO {target_name} AT INDEX {eso_row.name} ({target_timestamp}): {eso_row[target_name]}")
logger.info(f"GER {target_name} AT INDEX {ger_row.name} ({target_timestamp}): {ger_row[target_name]}")
# ============================================================================================================================



# ============================================================================================================================
## NORMALIZING DATA
logger.info('NORMALIZING DATA')

def normalize(X_train, X_eval, y_train, y_eval):
    # Sort both datasets by ID
    X_train = X_train.sort_values('ID')
    X_eval = X_eval.sort_values('ID')
    y_train = y_train.sort_values('ID')
    y_eval = y_eval.sort_values('ID')

    # Initialize scaler
    X_scaler = MinMaxScaler(clip=True)
    y_scaler = MinMaxScaler(clip=True)
    
    # Create copies
    X_train_scaled = X_train.copy()
    X_eval_scaled = X_eval.copy()
    y_train_scaled = y_train.copy()
    y_eval_scaled = y_eval.copy()
    
    # Scale X
    for column in X_train.columns:
        if column != 'ID':
            X_train_scaled[column] = X_scaler.fit_transform(X_train[[column]])
            X_eval_scaled[column] = X_scaler.transform(X_eval[[column]])
    
    # Scale y
    for column in y_train.columns:
        if column != 'ID':
            y_train_scaled[column] = y_scaler.fit_transform(y_train[[column]])
            y_eval_scaled[column] = y_scaler.transform(y_eval[[column]])

    return X_train_scaled, X_eval_scaled, y_train_scaled, y_eval_scaled, y_scaler

# Use data normlaization function
X_train_eso, X_eval_eso, y_train_eso, y_eval_eso, yeso_scaler = normalize(X_train_eso, X_eval_eso, y_train_eso, y_eval_eso)
X_train_ger, X_eval_ger, y_train_ger, y_eval_ger, yger_scaler = normalize(X_train_ger, X_eval_ger, y_train_ger, y_eval_ger)

# Check scaler
eso_row = y_train_eso[y_train_eso['ID'] == target_timestamp].iloc[0]
ger_row = y_train_ger[y_train_ger['ID'] == target_timestamp].iloc[0]
normalized_esovalue = eso_row[target_name]
normalized_gervalue = ger_row[target_name]
original_esovalue = yeso_scaler.inverse_transform([[normalized_esovalue]])[0][0]
original_gervalue = yger_scaler.inverse_transform([[normalized_gervalue]])[0][0]
logger.info(f"ESO TARGET AT INDEX {eso_row.name} ({target_timestamp}): {normalized_esovalue}, NORMALIZE ESO BACK: {original_esovalue}")
logger.info(f"GER TARGEt AT INDEX {ger_row.name} ({target_timestamp}): {normalized_gervalue}, NORMALIZE GER BACK: {original_gervalue}")
# ============================================================================================================================



# ============================================================================================================================
## INSPECT NORMALIZED DATA AND CORRELATIONS
logger.info('INSPECT NORMALIZED DATA AND CORRELATIONS')

threshold = 0.7
path_eso_norm = Path(__file__).resolve().parent.parent / 'data' / 'processed' / 'normalized' / 'eso'; path_eso_norm.mkdir(parents=True, exist_ok=True)
path_ger_norm = Path(__file__).resolve().parent.parent / 'data' / 'processed' / 'normalized' / 'ger'; path_ger_norm.mkdir(parents=True, exist_ok=True)

# ESO
# fit_transformed
_ = inspect_data(X_train_eso, output_path=path_eso_norm / f'X_train_eso.txt')
_ = inspect_data(y_train_eso, output_path=path_eso_norm / f'y_train_eso.txt')
# transformed
_ = inspect_data(X_eval_eso, output_path=path_eso_norm / f'X_eval_eso.txt')
_ = inspect_data(y_eval_eso, output_path=path_eso_norm / f'y_eval_eso.txt')
# GER
# fit_transformed
_ = inspect_data(X_train_ger, output_path=path_ger_norm / f'X_train_ger.txt')
_ = inspect_data(y_train_eso, output_path=path_ger_norm / f'y_train_eso.txt')
# transformed
_ = inspect_data(X_eval_ger, output_path=path_ger_norm / f'X_eval_ger.txt')
_ = inspect_data(y_eval_ger, output_path=path_ger_norm / f'y_eval_ger.txt')

# Correlation
eso_corr = find_high_correlations(X_train_eso, threshold, output_path=path_eso_norm / f'eso_corr_tresh{threshold}.txt')
ger_corr = find_high_correlations(X_train_ger, threshold, output_path=path_ger_norm / f'ger_corr_tresh{threshold}.txt')
# ============================================================================================================================



# ============================================================================================================================
## IMPORT FOR MODELING
logger.info('IMPORT FOR MODELING')

import numpy as np
import matplotlib.pyplot as plt
import torch
import datetime
import pytz

from enum import Enum
from pathlib import Path
from enum import Enum
from datetime import datetime
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

# -------------------------------------------------------------------------------------------------------------------------------- 

# Make small dataset for testing
size = 400
X_train_eso = X_train_eso.iloc[:size]; X_train_ger = X_train_ger.iloc[:size]
y_train_eso = y_train_eso.iloc[:size]; y_train_ger = y_train_ger.iloc[:size]
X_eval_eso = X_eval_eso.iloc[:size]; X_eval_ger = X_eval_ger.iloc[:size]
y_eval_eso = y_eval_eso.iloc[:size]; y_eval_ger = y_eval_ger.iloc[:size]
# ============================================================================================================================



# ============================================================================================
class ModelConfig: # Configuration class for the model
    def __init__(self): 
        self.sequence_length = 48           # Number of datapoints for one prediction, used timesteps for one prediction, defines how many LSTM cells are processed in parallel (1 datapoints=1 LSTM cell)
        self.val_split = 0.25               # Split for modelling
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.hidden_size = 176              # Neurons in LSTM for each hidden layer
        self.num_layers = 3                 # Number of hidden layers
        self.batch_size = 56                # Number of sequences processed together in one iteration
                                            # Each sequence has length sequence_length (e.g. 48)
                                            # Total batches = dataset_size / batch_size
                                            # e.g. (datapoints=100000) / (batch_size=32) = 3125 batches
        self.learning_rate = 0.0012          # Learning rate for the optimizer
        self.epochs = 24                    # One complete pass of all data to train
        self.dropout = 0.12                  # TODO REMOVE DROPOUT FOR FINAL MODEL for regularization: randomly remove a fraction of the connections in your network for any givin training example to learn redundant information
        self.patience = 12                   # Number of epochs to wait for improvement before early stopping
        self.grad_clip = 0.8         
        self.scheduler_factor = 0.6         # Learning rate decay factor
        self.scheduler_patience = 8         # Learning rate decay patience
        self.output_size = 1
        self.num_threads = 6
                                            # residual connections?
                                            # batch normalization?
# ============================================================================================



# ============================================================================================
class TimeSeriesDataset(Dataset): 
    def __init__(self, X, y, config):
                
        self.X = torch.FloatTensor(X.select_dtypes(include=['float64']).values) # Create a tensor from X data, only float64 values
        self.y = torch.FloatTensor(y[target_name].values) # Create a tensor from y data, only target_name values 
        self.sequence_length = config.sequence_length # Set the sequence length
        
    def __len__(self): # Return the length of the dataset
        return len(self.X) - self.sequence_length # Return the length of the X data minus the sequence length
    
    def __getitem__(self, idx):
        return (self.X[idx:idx + self.sequence_length],
                self.y[idx + self.sequence_length])
# ============================================================================================



# ============================================================================================
class LSTMWithBN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMWithBN, self).__init__()
        self.cell = LSTMCellWithBN(input_size, hidden_size)
        self.hidden_size = hidden_size

    # ----------------------------------------------------------------------------------------

    def forward(self, x, hidden=None):
        batch_size, seq_len, _ = x.size()
        
        if hidden is None:
            hidden = (torch.zeros(batch_size, self.hidden_size).to(x.device),
                     torch.zeros(batch_size, self.hidden_size).to(x.device))
        
        outputs = []
        for t in range(seq_len):
            hidden = self.cell(x[:, t, :], hidden)
            outputs.append(hidden[0])
        
        return torch.stack(outputs, dim=1), hidden
# ============================================================================================



# ============================================================================================
class LSTMCellWithBN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCellWithBN, self).__init__()
        
        # Combined linear transformation for all gates
        self.gates = nn.Linear(input_size + hidden_size, 4 * hidden_size)
        self.hidden_size = hidden_size
        
        # Batch normalization layers - one for each gate type before activation
        self.bn_ingate = nn.BatchNorm1d(hidden_size)
        self.bn_forgetgate = nn.BatchNorm1d(hidden_size)
        self.bn_cellgate = nn.BatchNorm1d(hidden_size)
        self.bn_outgate = nn.BatchNorm1d(hidden_size)

    # ----------------------------------------------------------------------------------------

    def forward(self, x, hidden):
        hx, cx = hidden
        
        # Concatenate input and hidden state
        gates = self.gates(torch.cat((x, hx), dim=1))
        
        # Split gates,  each gate has hidden_size neurons and batch_size 
        chunks = gates.chunk(4, dim=1)
        
        # Apply batch norm before each activation function
        ingate = torch.sigmoid(self.bn_ingate(chunks[0]))
        forgetgate = torch.sigmoid(self.bn_forgetgate(chunks[1]))
        cellgate = torch.tanh(self.bn_cellgate(chunks[2]))
        outgate = torch.sigmoid(self.bn_outgate(chunks[3]))
        
        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)
        
        return hy, cy
# ============================================================================================



# ============================================================================================
class LSTMPredictor(nn.Module):
    def __init__(self, input_size, config):
        super(LSTMPredictor, self).__init__()
        
        # Create list of LSTM layers
        self.lstm_layers = nn.ModuleList()
        
        # First layer
        self.lstm_layers.append(LSTMWithBN(input_size, config.hidden_size))
        
        # Additional layers
        for _ in range(config.num_layers - 1):
            self.lstm_layers.append(LSTMWithBN(config.hidden_size, config.hidden_size))
        
        # Output layers
        self.dropout = nn.Dropout(config.dropout)       # TODO REMOVE DROPOUT FOR FINAL MODEL
        self.linear = nn.Linear(config.hidden_size, config.output_size)
    
    # ----------------------------------------------------------------------------------------

    def forward(self, x):
        hidden = None
        for lstm in self.lstm_layers:
            x, hidden = lstm(x, hidden)
            
        # Use only the last output
        x = x[:, -1, :]
        
        # Apply dropout and linear layer
        x = self.dropout(x)                             # TODO REMOVE DROPOUT FOR FINAL MODEL
        predictions = self.linear(x) 
        
        return predictions
# ============================================================================================



# ============================================================================================
## METRICS
def calculate_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # SMAPE calculation
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    smape = np.mean(numerator / denominator) * 100
    
    # MASE calculation
    mae = np.mean(np.abs(y_true - y_pred))
    seasonal_period = 48
    naive_errors = np.abs(y_true[seasonal_period:] - y_true[:-seasonal_period])
    mase = mae / np.mean(naive_errors)
    
    # Directional Accuracy
    direction_true = np.diff(y_true) > 0
    direction_pred = np.diff(y_pred) > 0
    da = np.mean(direction_true == direction_pred) * 100
    
    # Standard metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'SMAPE': smape,
        'MASE': mase,
        'DA': da,
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }
# ============================================================================================



# ============================================================================================
## TRAINING
def train_model(X_train, y_train, config, logger, model_dir):  

    start_time = datetime.now()
    epoch_times = []
    logger.info(f"Training started at: {start_time}")

    # Ensure we have enough data for the sequence length
    min_required_length = config.sequence_length + 1
    if len(X_train) < min_required_length:
        raise ValueError(f"Not enough data points. Need at least {min_required_length} samples.")
    
    # Calculate split point ensuring enough samples for sequence
    valid_length = len(X_train) - config.sequence_length
    split_idx = int(valid_length * (1 - config.val_split))
    
    # Split data
    X_train_split = X_train.iloc[:split_idx + config.sequence_length]
    X_val_split = X_train.iloc[split_idx:]
    y_train_split = y_train.iloc[:split_idx + config.sequence_length]
    y_val_split = y_train.iloc[split_idx:]
    
    logger.info(f"X_train split size 75%: {len(X_train_split)}")
    logger.info(f"y_train split size 75%: {len(y_train_split)}")
    logger.info(f"X_val split size 25%: {len(X_val_split)}")
    logger.info(f"y_val split size 25%: {len(y_val_split)}")
    
    # Create datasets
    train_dataset = TimeSeriesDataset(X_train_split, y_train_split, config)
    val_dataset = TimeSeriesDataset(X_val_split, y_val_split, config)
    
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError("Dataset length is 0 after splitting. Adjust sequence_length or val_split.")
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    # ---------------------------------------------------------------------------------------
    
    logger.info("Model Configuration:")
    for param, value in vars(config).items():
        logger.info(f"{param}: {value}")
    
    # Model setup
    input_size = X_train.select_dtypes(include=['float64']).shape[1]
    model = LSTMPredictor(input_size, config)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=config.scheduler_factor, patience=config.scheduler_patience, verbose=True)
    
    # Init lists for losses
    train_losses = []
    val_losses = []

    # For both saving and early stopping
    best_loss = float('inf')
    patience_counter = 0
    patience = config.patience
    
    train_metrics = None
    val_metrics = None

    for epoch in range(config.epochs):

        epoch_start = datetime.now()

        # Training phase
        model.train()
        train_loss = 0
        train_preds = []
        train_true = []
        
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.epochs} [Train]')
        for X_batch, y_batch in train_bar:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch.unsqueeze(1))
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip)
            
            optimizer.step()
            
            train_loss += loss.item()
            train_preds.extend(y_pred.detach().numpy().flatten())
            train_true.extend(y_batch.numpy().flatten())
            
            train_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_preds = []
        val_true = []
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{config.epochs} [Val]')
            for X_batch, y_batch in val_bar:
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch.unsqueeze(1))
                
                val_loss += loss.item()
                val_preds.extend(y_pred.numpy().flatten())
                val_true.extend(y_batch.numpy().flatten())
                
                val_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        if (epoch + 1) % 10 == 0:
            log_msg = (f"Epoch {epoch+1}:\n"
                        f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}\n"
                        f"Last Train - True: {train_true[-1]:.4f} | Pred: {train_preds[-1]:.4f}\n"
                        f"Last Val - True: {val_true[-1]:.4f} | Pred: {val_preds[-1]:.4f}")
            logger.info(log_msg)

        # Update learning rate
        scheduler.step(avg_val_loss)
        

        # Early stopping when notihing changes ------------------------------
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
            #torch.save(model.state_dict(), 'best_model.pth')
            logger.info(f"New best model saved (val_loss: {best_loss:.4f})")
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break
        # -------------------------------------------------------------------

        # Calculate metrics every 10 epochs
        if (epoch + 1) % 10 == 0:
            train_metrics = calculate_metrics(train_true, train_preds)
            val_metrics = calculate_metrics(val_true, val_preds)
            logger.info("\nTraining Metrics:")
            for metric, value in train_metrics.items():
                logger.info(f"{metric}: {value:.4f}")
            logger.info("\nValidation Metrics:")
            for metric, value in val_metrics.items():
                logger.info(f"{metric}: {value:.4f}")

        epoch_end = datetime.now()
        epoch_duration = epoch_end - epoch_start
        epoch_times.append(epoch_duration)
        
        logger.info(f'Epoch {epoch + 1} duration: {epoch_duration}, average batch time: {epoch_duration / len(train_loader)}')

    # Log final timing statistics
    logger.info(f"""
    Training Summary:
    Total duration: {datetime.now() - start_time}
    Average epoch time: {sum(epoch_times, timedelta()) / len(epoch_times)}
    Fastest epoch: {min(epoch_times)}
    Slowest epoch: {max(epoch_times)}
    """)
    
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save
    plt.tight_layout()
    plot_path = model_dir / 'training_loss.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return model, {'train': train_metrics, 'val': val_metrics}
# ============================================================================================



# ============================================================================================
## PREDICTION
def predict_and_compare(model, X_train, y_train, y_scaler, config, logger, model_dir,target_timestamp=None):
    # Verify timestamps
    y_train = y_train.sort_values('ID')
    X_train = X_train.sort_values('ID')
    
    logger.info("\nTimestamp range in training data:")
    logger.info(f"Start: {y_train['ID'].iloc[0]}")
    logger.info(f"End: {y_train['ID'].iloc[-1]}")
    logger.info(f"Total length X_train: {len(X_train)}")
    logger.info(f"Total length y_train: {len(y_train)}")
    
    # Set target index
    if target_timestamp is None:
        target_idx = len(X_train) - 1
    else:
        mask = X_train['ID'] == pd.to_datetime(target_timestamp)
        if not mask.any():
            raise ValueError(f"Target timestamp {target_timestamp} not found in data")
        target_idx = mask.idxmax()
    
    logger.info(f"Dataset internal target index: {target_idx}")
    
    # Get sequence indices
    sequence_start = target_idx - config.sequence_length
    if sequence_start < 0:
        raise ValueError(f"Not enough historical data for prediction. Need at least {config.sequence_length} points before target.")
    
    # Prepare sequence data for final prediction
    sequence_data = X_train.iloc[sequence_start:target_idx].select_dtypes(include=['float64']).values
    
    # Validate sequence data
    if len(sequence_data) != config.sequence_length:
        raise ValueError(f"Invalid sequence length: {len(sequence_data)}, expected: {config.sequence_length}")
    
    logger.info(f"Sequence shape before processing: {sequence_data.shape}")
    
    logger.info(f"\nX_train head: {X_train['ID'].head()}")
    logger.info(f"\ny_train tail: {y_train['ID'].tail()}")

    # Make predictions
    model.eval()
    with torch.no_grad():
        # Get historical values for visualization
        history_start = max(0, target_idx - 10)
        true_values_normalized = y_train[target_name].iloc[history_start:target_idx + 1].values
        timestamps = X_train['ID'].iloc[history_start:target_idx + 1]
        
        # Calculate predictions for each point including the target
        predicted_values_normalized = []
        valid_timestamps = []
        
        for i in range(history_start, target_idx + 1):
            seq_start = i - config.sequence_length
            if seq_start >= 0:  # Only predict if we have enough historical data
                seq_data = X_train.iloc[seq_start:i].select_dtypes(include=['float64']).values
                if len(seq_data) == config.sequence_length:
                    seq = torch.FloatTensor(seq_data).unsqueeze(0)
                    pred = model(seq)
                    predicted_values_normalized.append(pred.item())
                    valid_timestamps.append(X_train['ID'].iloc[i])


        
        # Inverse transform all values
        true_values_original = y_scaler.inverse_transform(true_values_normalized.reshape(-1, 1)).flatten()
        predicted_values_original = y_scaler.inverse_transform(np.array(predicted_values_normalized).reshape(-1, 1)).flatten()
        
        # Print comparison for the points we have predictions for
        logger.info("\nValues Comparison:")
        for ts, actual, pred in zip(valid_timestamps, true_values_original[-len(predicted_values_original):], predicted_values_original):
            logger.info(f"Timestamp: {ts}")
            logger.info(f"Actual: {actual:.2f}")
            logger.info(f"Predicted: {pred:.2f}")
            logger.info(f"Difference: {abs(pred - actual):.2f}\n")
        
        # Calculate final metrics
        final_true = true_values_original[-1]
        final_pred = predicted_values_original[-1]
        abs_error = abs(final_pred - final_true)
        rel_error = (abs_error / final_true) * 100
        
        logger.info("\nFinal Prediction Metrics:")
        logger.info(f"Absolute error: {abs_error:.4f}")
        logger.info(f"Relative error: {rel_error:.2f}%")
        
        if len(true_values_original) > 1:
            prev_true = true_values_original[-2]
            direction_match = ((final_pred > final_true) == (final_true > prev_true))
            logger.info(f"Direction Match: {'✓' if direction_match else '✗'}")
        

        # -------------------------------------------------------------------
        # First plot (detailed view)
        target_time = pd.to_datetime(timestamps.iloc[-1]).strftime('%Y%m%d_%H%M')
    
        plt.figure(figsize=(12, 8))
        # Plot actual values in green (continuous line)
        plt.plot(timestamps, true_values_original, 'g-o', 
                label='Actual Values', linewidth=2)
        # Plot predicted values in red (continuous line)
        plt.plot(valid_timestamps, predicted_values_original, 'r-o',
                label='Predicted Values', linewidth=2)
        
        plt.title('Model: Actual vs Predicted Values (Detail Last Points)')
        plt.xlabel('Time')
        plt.ylabel('TARGET Values')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
                      
        # Save first plot
        plot_path = model_dir / f'prediction_detail{target_time}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        # -------------------------------------------------------------------
        
        # Second plot (all datapoints)
        plt.figure(figsize=(15, 8))
        
        # Get all actual values
        all_true_values = y_train[target_name].values
        all_true_original = y_scaler.inverse_transform(all_true_values.reshape(-1, 1)).flatten()
        
        # Calculate predictions for all possible points
        all_predictions = []
        valid_timestamps = []
        
        for i in range(config.sequence_length, len(X_train)):
            seq_data = X_train.iloc[i-config.sequence_length:i].select_dtypes(include=['float64']).values
            seq = torch.FloatTensor(seq_data).unsqueeze(0)
            pred = model(seq)
            all_predictions.append(pred.item())
            valid_timestamps.append(X_train['ID'].iloc[i])
            
        all_predictions_original = y_scaler.inverse_transform(np.array(all_predictions).reshape(-1, 1)).flatten()
        
        # Plot all datac
        step = 200
        plt.plot(X_train['ID'][config.sequence_length::step], 
            all_true_original[config.sequence_length::step], 
            'g-', label='Actual Values (sampled)', linewidth=0.6)

        plt.plot(valid_timestamps[::step], 
            all_predictions_original[::step], 
            'r-', label='Predicted Values (sampled)', linewidth=0.6)
        
        plt.title('Model: Actual vs Predicted Values')
        plt.xlabel('Time')
        plt.ylabel('TARGET Values')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # Save second plot
        plot_path_all = model_dir / f'prediction_all_data_{target_time}.png'
        plt.savefig(plot_path_all, dpi=300, bbox_inches='tight')
        plt.close()
        
        return predicted_values_original[-1], true_values_original[-1]
# ============================================================================================



# ============================================================================================
class DatasetType(Enum):
    ESO = 0
    GER = 1
# ============================================================================================



# ============================================================================================
def train(dataset_type, config):
    # -----------------------------------------------------------------
    torch.set_num_threads(config.num_threads)
    timestamp = datetime.now(pytz.timezone('Europe/Berlin')).strftime('%Y%m%d_%H%M')
    base_dir = Path(__file__).resolve().parent.parent / 'data' / 'models' / dataset_type.name.lower()
    model_dir = base_dir / f'model_{timestamp}'
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logger with model directory path
    logger = setup_logger(model_dir=model_dir)
    logger.info(f"Starting training for {dataset_type.name} dataset")
     # -----------------------------------------------------------------

    try:
        if dataset_type == DatasetType.ESO:
            X_train, y_train = X_train_eso, y_train_eso
            y_scaler = yeso_scaler
        elif dataset_type == DatasetType.GER:
            X_train, y_train = X_train_ger, y_train_ger
            y_scaler = yger_scaler
        else:
            raise ValueError(f"Invalid dataset type: {dataset_type}")

        model, metrics = train_model(X_train, y_train, config, logger, model_dir)
        
        predicted_value, true_value = predict_and_compare(
            model, 
            X_train, 
            y_train, 
            y_scaler,  
            config, 
            logger,
            model_dir,
            target_timestamp=None
        )

        # Save model and config
        torch.save(model.state_dict(), model_dir / 'model.pth')
        logger.info(f"Model saved in: {model_dir}")
        
        return model, metrics, predicted_value, true_value  
    
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

config = ModelConfig()
logger.info('EVALUATE SAVED MODEL WITH NEW DATA')
# ============================================================================================



# ============================================================================================
def evaluate_model(dataset_type: DatasetType, timestamp: str):
    model_dir = Path(__file__).resolve().parent.parent / 'data' / 'models' / dataset_type.name.lower() / f'model_{timestamp}'
    logger = setup_logger(model_dir=model_dir, mode='a')
    logger.info('\n' + '='*80)
    logger.info(f"Starting new evaluation for {dataset_type.name} model from {timestamp}")

    try:
        # Select appropriate dataset and scaler
        if dataset_type == DatasetType.ESO:
            X_eval, y_eval = X_eval_eso, y_eval_eso
            y_scaler = yeso_scaler
        elif dataset_type == DatasetType.GER:
            X_eval, y_eval = X_eval_ger, y_eval_ger
            y_scaler = yger_scaler
        else:
            raise ValueError(f"Invalid dataset type: {dataset_type}")

        # Sort the evaluation data by timestamp
        X_eval = X_eval.sort_values('ID').copy()
        y_eval = y_eval.loc[X_eval.index].copy()

        logger.info(f"X_eval head: {X_eval['ID'].head()}")
        logger.info(f"X_eval tail: {X_eval['ID'].tail()}")
        logger.info(f"y_eval head: {y_eval['ID'].head()}")
        logger.info(f"y_eval tail: {y_eval['ID'].tail()}")

        config = ModelConfig()
        input_size = X_eval.select_dtypes(include=['float64']).shape[1]
        model = LSTMPredictor(input_size, config)
        model.load_state_dict(torch.load(model_dir / 'model.pth', weights_only=True))
        model.eval()

        # Create dataset and dataloader with sorted data
        eval_dataset = TimeSeriesDataset(X_eval, y_eval, config)
        eval_loader = DataLoader(eval_dataset, batch_size=config.batch_size, shuffle=False)

        # Initialize lists for predictions and true values
        all_predictions = []
        all_true_values = []
        timestamps = []
        eval_loss = 0
        criterion = nn.MSELoss()

        with torch.no_grad():
            for batch_idx, (X_batch, y_batch) in enumerate(tqdm(eval_loader, desc='Evaluating batches (total_samples/batch_size)')):
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch.unsqueeze(1))
                eval_loss += loss.item()
                
                # Get corresponding timestamps for this batch
                start_idx = batch_idx * config.batch_size + config.sequence_length
                end_idx = start_idx + len(y_batch)
                batch_timestamps = X_eval['ID'].iloc[start_idx:end_idx]
                
                all_predictions.extend(y_pred.numpy().flatten())
                all_true_values.extend(y_batch.numpy().flatten())
                timestamps.extend(batch_timestamps)

        avg_eval_loss = eval_loss / len(eval_loader)

        # Convert predictions back to original scale
        predictions_original = y_scaler.inverse_transform(
            np.array(all_predictions).reshape(-1, 1)
        ).flatten()
        
        true_values_original = y_scaler.inverse_transform(
            np.array(all_true_values).reshape(-1, 1)
        ).flatten()

        # Create DataFrame with results
        results_df = pd.DataFrame({
            'timestamp': timestamps,
            'true_values': true_values_original,
            'predictions': predictions_original
        }).sort_values('timestamp')

        # Calculate metrics
        eval_metrics = calculate_metrics(results_df['true_values'], results_df['predictions'])
        

        logger.info(f"\nEvaluation performed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Average evaluation loss: {avg_eval_loss:.4f}")
        logger.info("\nEvaluation Metrics:")
        for metric, value in eval_metrics.items():
            logger.info(f"{metric}: {value:.4f}")

        logger.info("\nSample Values Comparison:")
        last_points = results_df.tail(6)
        for _, row in last_points.iterrows():
            logger.info(f"Timestamp: {row['timestamp']}")
            logger.info(f"Actual: {row['true_values']:.2f}")
            logger.info(f"Predicted: {row['predictions']:.2f}")
            logger.info(f"Difference: {abs(row['predictions'] - row['true_values']):.2f}\n")

        # Calculate and log final prediction metrics
        last_row = results_df.iloc[-1]
        abs_error = abs(last_row['predictions'] - last_row['true_values'])
        rel_error = (abs_error / last_row['true_values']) * 100
        
        logger.info("\nFinal Prediction Metrics:")
        logger.info(f"Absolute error: {abs_error:.4f}")
        logger.info(f"Relative error: {rel_error:.2f}%")
        
        if len(results_df) > 1:
            second_last_true = results_df.iloc[-2]['true_values']
            last_true = last_row['true_values']
            last_pred = last_row['predictions']
            direction_match = ((last_pred > last_true) == (last_true > second_last_true))
            logger.info(f"Direction Match: {'✓' if direction_match else '✗'}")


        # Create visualizations
        # Detailed view (last 6 points)
        plt.figure(figsize=(12, 8))
        detail_df = results_df.tail(6)
        formatted_timestamps = [ts.strftime('%d %H:%M') for ts in detail_df['timestamp']]
        
        plt.plot(range(len(detail_df)), detail_df['true_values'], 'b-o', 
                label='Actual Values', linewidth=2)
        plt.plot(range(len(detail_df)), detail_df['predictions'], 'r-o',
                label='Predicted Values', linewidth=2)
        
        plt.title('Evaluation: Actual vs Predicted Values (Detail Last Points)')
        plt.xlabel('Time')
        plt.ylabel('TARGET Values')
        plt.xticks(range(len(detail_df)), formatted_timestamps, rotation=45)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        detail_plot_path = model_dir / f'evaluation_results_detail_{datetime.now().strftime("%Y%m%d_%H%M")}.png'
        plt.savefig(detail_plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        # Full dataset view
        plt.figure(figsize=(15, 8))
        step = max(len(results_df) // 100, 1)  # Sample points for clearer visualization
        
        plt.plot(results_df['timestamp'][::step], results_df['true_values'][::step], 'b-', 
                label='Actual Values', alpha=0.6, linewidth=0.6)
        plt.plot(results_df['timestamp'][::step], results_df['predictions'][::step], 'r-', 
                label='Predicted Values', alpha=0.4, linewidth=0.6)
        
        plt.title('Evaluation: Actual vs Predicted Values')
        plt.xlabel('Time')
        plt.ylabel('Target Values')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        full_plot_path = model_dir / f'evaluation_results_{datetime.now().strftime("%Y%m%d_%H%M")}.png'
        plt.savefig(full_plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        # Save detailed results
        results_path = model_dir / f'evaluation_results_{datetime.now().strftime("%Y%m%d_%H%M")}.csv'
        results_df.to_csv(results_path, index=False)
        
        logger.info(f"Evaluation results saved to: {model_dir}")
        logger.info('='*80 + '\n')
        
        return {
            'metrics': eval_metrics,
            'predictions': predictions_original,
            'true_values': true_values_original,
            'loss': avg_eval_loss
        }
    
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        logger.info('='*80 + '\n')
        raise
# ============================================================================================



# ============================================================================================
## USER INTERACTION
logger.info('START USER INTERACTION')

def get_user_choice():
    '''Choose action'''
    while True:
        print("\nWHAT WOULD YOU LIKE TO DO?")
        print("1: Train a new model")
        print("2: Evaluate an existing model")
        print("3: Exit")
        choice = input("Enter your choice (1-3): ")
        
        if choice in ['1', '2', '3']:
            return int(choice)
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

# --------------------------------------------------------------------------------------------------------------------------------

def get_dataset_choice():
    '''Choose dataset'''
    while True:
        print("\nSELECT DATASET TYPE:")
        print("1: ESO")
        print("2: GER")
        choice = input("Enter your choice (1-2): ")
        
        if choice == '1':
            return DatasetType.ESO
        elif choice == '2':
            return DatasetType.GER
        else:
            print("Invalid choice. Please enter 1 or 2.")

# --------------------------------------------------------------------------------------------------------------------------------

def get_timestamp():
    '''Choose timestamp'''
    while True:
        timestamp = input("\nENTER MODEL TIMESTAMP (FORMAT: YYYYMMDD_HHMM): ")
        if len(timestamp) == 13 and timestamp[8] == '_':
            return timestamp
        else:
            print("Invalid timestamp format. Please use YYYYMMDD_HHMM (e.g., 20241201_1430)")

# --------------------------------------------------------------------------------------------------------------------------------

# Main interaction loop
while True:
    choice = get_user_choice()
    
    if choice == 3:
        logger.info('USER CHOOSE TO EXIT')
        break
        
    dataset_type = get_dataset_choice()
    
    if choice == 1:
        # Train new model
        logger.info(f'Training new model for {dataset_type.name} dataset')
        print(f"\nTraining new model for {dataset_type.name} dataset...")
        config = ModelConfig()
        model, metrics, predicted_value, last_true_value = train(dataset_type, config)
        print("Training completed!")
        
    elif choice == 2:
        # Evaluate existing model
        timestamp = get_timestamp()
        logger.info(f'Evaluating model for {dataset_type.name} dataset from {timestamp}')
        print(f"\nEvaluating model for {dataset_type.name} dataset from {timestamp}...")
        results = evaluate_model(dataset_type, timestamp)
        print("Evaluation completed!")
        print("\nEvaluation Metrics:")
        for metric, value in results['metrics'].items():
            print(f"{metric}: {value:.4f}")

# --------------------------------------------------------------------------------------------------------------------------------

# END PROJECT
logger.info('END PROJECT')
# ============================================================================================================================