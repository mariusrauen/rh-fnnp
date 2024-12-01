# ============================================================================================================================
## IMPORT LIBRARIES

import pandas as pd
from sklearn. model_selection import train_test_split
from sklearn. preprocessing import MinMaxScaler
from pathlib import Path
from pprint import pprint

from modules.classVisualizer import Visualizer
from modules.classVisualizer import find_high_correlations, inspect_data
from modules.classMetaData import setup_logger

logger = setup_logger()

# ============================================================================================================================
# READ IN FROM DATA PREPERATION
logger.info("READ IN FROM DATA PREPERATION")

data_path = Path(__file__).resolve().parent.parent / 'data' / 'processed' / 'DataPrep'

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
## VISUALIZE COMPARABLE FEATURES
logger.info("VISUALIZE COMPARABLE FEATURES")

# Visualize over time
visualizer = Visualizer(df_eso, df_ger)
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
# SPLITTING DATA
logger.info('SPLITTING DATA')

# ESO
target_name = 'FOSSIL'
eso_combine_cols = df_eso['GAS'] + df_eso['COAL']
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
ger_combine_cols = df_ger['Erzeugung_Erdgas [MWh]'] + df_ger['Erzeugung_Steinkohle [MWh]'] + df_ger['Erzeugung_Braunkohle [MWh]']
df_ger[target_name] = ger_combine_cols
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

'''pprint(X_train_eso.head().to_dict())
pprint(X_eval_eso.head().to_dict())
pprint(y_train_eso.head().to_dict())
pprint(y_eval_eso.head().to_dict())'''


# ============================================================================================================================
# SELECT AN FOSSIL VALUE FOR SUBSEQUENT INDEX EVALUATION 
logger.info("SELECT A FOSSIL VALUE FOR SUBSEQUENT INDEX EVALUATION")
target_timestamp = pd.to_datetime('2017-04-12 03:30:00')
eso_row = y_train_eso[y_train_eso['ID'] == target_timestamp].iloc[0]
ger_row = y_train_ger[y_train_ger['ID'] == target_timestamp].iloc[0]
logger.info(f"ESO FOSSIL AT INDEX {eso_row.name} ({target_timestamp}): {eso_row['FOSSIL']}")
logger.info(f"GER FOSSIL AT INDEX {ger_row.name} ({target_timestamp}): {ger_row['FOSSIL']}")

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


row = y_train_eso[y_train_eso['ID'] == target_timestamp].iloc[0]
normalized_value = row['FOSSIL']
original_value = yeso_scaler.inverse_transform([[normalized_value]])[0][0]
logger.info(f"FOSSIL AT INDEX {row.name} ({target_timestamp}): {row['FOSSIL']}, NORMALIZE BACK: {original_value}")

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
## IMPORT FOR MODELING
logger.info('IMPORT FOR MODELING')
# https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
# https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html 
# TODO - Consider: batch size, epoche, learning rate, learning rate dk 
# TODO - Evaluation
# TODO - Commenting
# TODO - Docker

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

# ============================================================================================================================

X_train_eso = X_train_eso.iloc[:400]; X_train_ger = X_train_ger.iloc[:400]
y_train_eso = y_train_eso.iloc[:400]; y_train_ger = y_train_ger.iloc[:400]
X_eval_eso = X_eval_eso.iloc[:400]; X_eval_ger = X_eval_ger.iloc[:400]
y_eval_eso = y_eval_eso.iloc[:400]; y_eval_ger = y_eval_ger.iloc[:400]

class ModelConfig:
    def __init__(self):
        self.sequence_length = 24
        self.val_split = 0.25
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.hidden_size = 16
        self.num_layers = 2
        self.batch_size = 32
        self.learning_rate = 0.001
        self.epochs = 30
        self.dropout = 0.3  
        self.patience = 16
        self.grad_clip = 1.0         
        self.scheduler_factor = 0.6
        self.scheduler_patience = 10
        self.output_size = 1

# ============================================================================================

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, config):
        self.X = torch.FloatTensor(X.select_dtypes(include=['float64']).values)
        self.y = torch.FloatTensor(y['FOSSIL'].values)
        self.sequence_length = config.sequence_length
        
    def __len__(self):
        return len(self.X) - self.sequence_length
        
    def __getitem__(self, idx):
        return (self.X[idx:idx + self.sequence_length],
                self.y[idx + self.sequence_length])
    
# ============================================================================================

class LSTMPredictor(nn.Module):
    def __init__(self, input_size, config):
        super(LSTMPredictor, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout
        )
        #self.bn = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.linear = nn.Linear(config.hidden_size, config.output_size)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        #lstm_out = self.bn(lstm_out[:, -1, :])
        #lstm_out = self.dropout(lstm_out)
        lstm_out = self.dropout(lstm_out[:, -1, :])
        predictions = self.linear(lstm_out)
        return predictions

# ============================================================================================

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

def train_model(X_train, y_train, config, logger, model_dir):  
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
    
    logger.info(f"Training split size: {len(X_train_split)}")
    logger.info(f"Validation split size: {len(X_val_split)}")
    
    # Create datasets
    train_dataset = TimeSeriesDataset(X_train_split, y_train_split, config)
    val_dataset = TimeSeriesDataset(X_val_split, y_val_split, config)
    
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError("Dataset length is 0 after splitting. Adjust sequence_length or val_split.")
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    # ---------------------------------------------------------------------------------------
    # Logging info
    model_architecture = f"""Model Architecture:
        LSTMPredictor(
            (lstm): LSTM(84, 16, num_layers=2, batch_first=True, dropout=0.3)
            (dropout): Dropout(p=0.3, inplace=False)
            (linear): Linear(in_features=16, out_features=1, bias=True)
        )"""
    logger.info(model_architecture)

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
    # --------------------------------------------------------------------------
    train_metrics = None
    val_metrics = None


    for epoch in range(config.epochs):
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
    
    logger.info(f"Target index: {target_idx}")
    
    # Get sequence indices
    sequence_start = target_idx - config.sequence_length
    if sequence_start < 0:
        raise ValueError(f"Not enough historical data for prediction. Need at least {config.sequence_length} points before target.")
    
    # Prepare sequence data
    sequence_data = X_train.iloc[sequence_start:target_idx].select_dtypes(include=['float64']).values
    
    # Validate sequence data
    if len(sequence_data) != config.sequence_length:
        raise ValueError(f"Invalid sequence length: {len(sequence_data)}, expected: {config.sequence_length}")
    
    logger.info(f"Sequence shape before processing: {sequence_data.shape}")
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        sequence = torch.FloatTensor(sequence_data).unsqueeze(0)  # Add batch dimension
        if sequence.size(1) == 0:
            raise ValueError("Empty sequence data")
            
        prediction = model(sequence)
        predicted_value_normalized = prediction.item()
        
        # Get historical values for visualization
        history_start = max(0, target_idx - 5)
        true_values_normalized = y_train['FOSSIL'].iloc[history_start:target_idx + 1].values
        timestamps = X_train['ID'].iloc[history_start:target_idx + 1]
        
        # Inverse transform values
        true_values_original = y_scaler.inverse_transform(true_values_normalized.reshape(-1, 1)).flatten()
        predicted_value_original = y_scaler.inverse_transform([[predicted_value_normalized]])[0][0]
        
        # Log historical values
        logger.info("\nHistorical values (Original Scale):")
        for idx, (ts, val) in enumerate(zip(timestamps[:-1], true_values_original[:-1])):
            logger.info(f"Index: {history_start + idx}, Timestamp: {ts}, Value: {val:.2f}")
        
        target_true_original = true_values_original[-1]
        
        # Log prediction results
        logger.info(f"\nTarget timestamp: {timestamps.iloc[-1]}")
        logger.info(f"Target index: {target_idx}")
        logger.info(f"Actual value (Original Scale): {target_true_original:.4f}")
        logger.info(f"Predicted value (Original Scale): {predicted_value_original:.4f}")
        
        # Calculate metrics
        abs_error = abs(predicted_value_original - target_true_original)
        rel_error = (abs_error / target_true_original) * 100
        
        logger.info("\nPrediction Metrics (Original Scale):")
        logger.info(f"Absolute Error: {abs_error:.4f}")
        logger.info(f"Relative Error: {rel_error:.2f}%")
        
        if len(true_values_original) > 1:
            prev_true = true_values_original[-2]
            direction_match = ((predicted_value_original > target_true_original) == 
                             (target_true_original > prev_true))
            logger.info(f"Direction Match: {'✓' if direction_match else '✗'}")
        
        # Create visualization
        target_time = pd.to_datetime(timestamps.iloc[-1]).strftime('%Y%m%d_%H%M')
        
        plt.figure(figsize=(10, 6))
        plt.plot(timestamps[:-1], true_values_original[:-1], 'b-o', 
                label='Historical Values')
        plt.plot(timestamps.iloc[-1], true_values_original[-1], 'go',
                label='Actual Value')
        plt.plot(timestamps.iloc[-1], predicted_value_original, 'ro',
                label='Predicted Value')
        plt.title('Historical Values and Prediction (Original Scale)')
        plt.xlabel('Time')
        plt.ylabel('FOSSIL Values')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True)
        
        # Save plot
        plt.tight_layout()
        plot_path = model_dir / f'prediction_{target_time}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return predicted_value_original, target_true_original

# ============================================================================================

class DatasetType(Enum):
    ESO = 0
    GER = 1

# ============================================================================================

def train(dataset_type, config):
    # -----------------------------------------------------------------
    timestamp = datetime.now(pytz.timezone('Europe/Berlin')).strftime('%Y%m%d_%H%M')
    #model_dir = Path(__file__).resolve().parent.parent / 'data' / 'models' / f'{dataset_type.name.lower()}_model_{timestamp}'
    base_dir = Path(__file__).resolve().parent.parent / 'data' / 'models' / dataset_type.name.lower()
    model_dir = base_dir / f'model_{timestamp}'
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logger with model directory path
    #logger = setup_logger2(model_dir)
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
        
        return model, metrics, predicted_value, true_value  # Changed last_true_value to true_value
    
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

config = ModelConfig()
dataset_type = DatasetType.GER
model, metrics, predicted_value, last_true_value = train(dataset_type, config)

# ============================================================================================


