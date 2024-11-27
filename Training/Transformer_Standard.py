import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm
# Define the function to load and preprocess the dataset
def load_waveform_dataset(file_paths):

    """
    Loads waveform data from Excel files, preprocesses it, and returns a TensorDataset.

    Args:
        file_paths (list): List of file paths to the Excel files.

    Returns:
        dataset (TensorDataset): A dataset containing the input and target tensors.
        normalization_params (dict): A dictionary with mean and std values for normalization.
    """
    all_src = []
    all_tgt = []
    all_target = []

    for file_path in file_paths:
        # Load data from Excel file
        df = pd.read_excel(file_path)

        # Number of waveform pairs in the file
        num_waveform_pairs = df.shape[1] // 2  # Each pair is V and I

        for pair_idx in range(num_waveform_pairs):
            # Extract voltage and current columns
            v_col = df.iloc[:, pair_idx * 2].values  # Voltage column
            i_col = df.iloc[:, pair_idx * 2 + 1].values  # Current column

            # Ensure the waveform length is 129
            assert len(v_col) == 1000, f"Waveform length is not 1000 in {file_path}"

            # Encoder input: current waveform (entire sequence)
            src_seq = i_col[:, np.newaxis]  # Shape: (1000, 1)

            # Decoder input: voltage waveform shifted by one time step
            tgt_seq = v_col[:-1, np.newaxis]  # Shape: (1000, 1)

            # Target output: voltage waveform (next time steps)
            target_seq = v_col[1:, np.newaxis]  # Shape: (1000, 1)

            # Append to the lists
            all_src.append(src_seq)
            all_tgt.append(tgt_seq)
            all_target.append(target_seq)

    all_src = np.array(all_src)       # Shape: (num_samples, 129, 1)
    all_tgt = np.array(all_tgt)       # Shape: (num_samples, 128, 1)
    all_target = np.array(all_target) # Shape: (num_samples, 128, 1)

    # Normalize the data
    src_mean = np.mean(all_src, axis=(0, 1))
    src_std = np.std(all_src, axis=(0, 1))
    tgt_mean = np.mean(all_tgt, axis=(0, 1))
    tgt_std = np.std(all_tgt, axis=(0, 1))

    # Avoid division by zero
    src_std[src_std == 0] = 1
    tgt_std[tgt_std == 0] = 1

    # Normalize src and tgt
    all_src = (all_src - src_mean) / src_std
    all_tgt = (all_tgt - tgt_mean) / tgt_std
    all_target = (all_target - tgt_mean) / tgt_std  # Use the same mean and std as tgt

    # Convert to tensors
    src_tensor = torch.FloatTensor(all_src)       # Shape: (num_samples, 129, 1)
    tgt_tensor = torch.FloatTensor(all_tgt)       # Shape: (num_samples, 128, 1)
    target_tensor = torch.FloatTensor(all_target) # Shape: (num_samples, 128, 1)

    # Create TensorDataset
    dataset = TensorDataset(src_tensor, tgt_tensor, target_tensor)

    # Return dataset and normalization parameters
    normalization_params = {
        'src_mean': src_mean,
        'src_std': src_std,
        'tgt_mean': tgt_mean,
        'tgt_std': tgt_std
    }

    return dataset, normalization_params

# List of file paths to your Excel files
file_paths = [
    'ITEM_1561_#1_waveform_data_downsampled.xlsx',
    'ITEM_1561_#2_waveform_data_downsampled.xlsx',
    'ITEM_1561_#3_waveform_data_downsampled.xlsx',
    'ITEM_1561_#4_waveform_data_downsampled.xlsx',
    'ITEM_1561_#5_waveform_data_downsampled.xlsx',
    'ITEM_1561_#6_waveform_data_downsampled.xlsx',
    'ITEM_1561_#7_waveform_data_downsampled.xlsx',
    'ITEM_1561_#8_waveform_data_downsampled.xlsx',
    'ITEM_1561_#9_waveform_data_downsampled.xlsx',
    'ITEM_1561_#10_waveform_data_downsampled.xlsx'
]

# Load the dataset
dataset, normalization_params = load_waveform_dataset(file_paths)

# Split the dataset into training and validation sets
total_size = len(dataset)
train_size = int(0.9 * total_size)  # 90% for training
val_size = total_size - train_size   # 10% for validation

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create DataLoaders
batch_size = 128  # Adjust based on your system's memory capacity
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Define the Transformer model
class Transformer(nn.Module):
    def __init__(self,
        input_size: int,
        dec_seq_len: int,
        max_seq_len: int,
        out_seq_len: int,
        dim_val: int,
        n_encoder_layers: int,
        n_decoder_layers: int,
        n_heads: int,
        dropout_encoder: float,
        dropout_decoder: float,
        dropout_pos_enc: float,
        dim_feedforward_encoder: int,
        dim_feedforward_decoder: int,
    ):
        super().__init__()

        self.dec_seq_len = dec_seq_len
        self.out_seq_len = out_seq_len
        self.dim_val = dim_val

        # Encoder input layer
        self.encoder_input_layer = nn.Sequential(
            nn.Linear(input_size, dim_val),
            nn.Tanh(),
            nn.Linear(dim_val, dim_val)
        )

        # Decoder input layer
        self.decoder_input_layer = nn.Sequential(
            nn.Linear(1, dim_val),  # Decoder input is voltage only
            nn.Tanh(),
            nn.Linear(dim_val, dim_val)
        )

        # Positional encoding
        self.positional_encoding_layer = PositionalEncoder(
            d_model=dim_val, dropout=dropout_pos_enc, max_len=max_seq_len
        )

        # Transformer encoder and decoder layers
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_val,
            nhead=n_heads,
            dim_feedforward=dim_feedforward_encoder,
            dropout=dropout_encoder,
            activation="relu",
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer, num_layers=n_encoder_layers
        )

        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim_val,
            nhead=n_heads,
            dim_feedforward=dim_feedforward_decoder,
            dropout=dropout_decoder,
            activation="relu",
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=self.decoder_layer, num_layers=n_decoder_layers
        )

        # Output layer
        self.linear_mapping = nn.Sequential(
            nn.Linear(dim_val, dim_val),
            nn.Tanh(),
            nn.Linear(dim_val, 1)  # Output is voltage value
        )

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, device) -> torch.Tensor:
        # Encoder
        src = self.encoder_input_layer(src)  # Shape: (batch_size, 129, dim_val)
        src = self.positional_encoding_layer(src)
        src = self.encoder(src)

        # Decoder
        tgt = self.decoder_input_layer(tgt)  # Shape: (batch_size, 128, dim_val)
        tgt = self.positional_encoding_layer(tgt)

        tgt_mask = generate_square_subsequent_mask(
            sz1=self.dec_seq_len, sz2=self.dec_seq_len
        ).to(device)

        output = self.decoder(
            tgt=tgt,
            memory=src,
            tgt_mask=tgt_mask
        )
        output = self.linear_mapping(output)
        return output  # Shape: (batch_size, dec_seq_len, 1)

# Positional Encoder
class PositionalEncoder(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)].to(x.device)
        return self.dropout(x)

def generate_square_subsequent_mask(sz1: int, sz2: int) -> torch.Tensor:
    return torch.triu(torch.ones(sz1, sz2) * float('-inf'), diagonal=1)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Initialize the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

model = Transformer(
    input_size=1,               # Current sequence has 1 feature
    dec_seq_len=999,            # Decoder sequence length
    max_seq_len=1999,            # enc_seq_len + dec_seq_len (129 + 128)
    out_seq_len=999,            # Output sequence length
    dim_val=32,                # Dimension of the value
    n_encoder_layers=1,
    n_decoder_layers=1,
    n_heads=4,
    dropout_encoder=0.1,
    dropout_decoder=0.1,
    dropout_pos_enc=0.1,
    dim_feedforward_encoder=64,
    dim_feedforward_decoder=64,
)
model.to(device)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

training_losses = []
validation_losses = []

num_epochs = 1000  # AROUND 500 EPOCHS IS GOOD

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}", unit='batch') as pbar:
        for src, tgt, target in train_loader:
            src = src.to(device)
            tgt = tgt.to(device)    
            target = target.to(device)

            # Forward pass
            output = model(src, tgt, device)
            loss = criterion(output, target)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * src.size(0)

            # Update the progress bar
            pbar.update(1)
            pbar.set_postfix({'Loss': f'{loss.item():.6f}'})

    # Calculate average training loss over the epoch
    epoch_loss = running_loss / len(train_loader.dataset)
    training_losses.append(epoch_loss)

    # Validation step
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        with tqdm(total=len(val_loader), desc=f"Validation {epoch+1}/{num_epochs}", unit='batch') as pbar_val:
            for src_val, tgt_val, target_val in val_loader:
                src_val = src_val.to(device)
                tgt_val = tgt_val.to(device)
                target_val = target_val.to(device)

                output_val = model(src_val, tgt_val, device)
                loss_val = criterion(output_val, target_val)
                val_loss += loss_val.item() * src_val.size(0)

                pbar_val.update(1)
                pbar_val.set_postfix({'Val Loss': f'{loss_val.item():.6f}'})

    # Calculate average validation loss over the epoch
    epoch_val_loss = val_loss / len(val_loader.dataset)
    validation_losses.append(epoch_val_loss)

    print(f"\nEpoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_loss:.6f}, Validation Loss: {epoch_val_loss:.6f}\n")

# Save the model
torch.save(model.state_dict(), 'transformer_model.pth')

print('Number of parameters:', count_parameters(model))
#print('Validation losses:' , validation_losses)
#print('Training losses:', training_losses)


#Inference
model.eval()

# Lists to store predicted and actual voltage sequences
predicted_voltages = []
actual_voltages = []

# Iterate over the validation dataset
for idx in range(len(val_dataset)):
    src_sample, tgt_sample, target_sample = val_dataset[idx]
    src_sample = src_sample.unsqueeze(0).to(device)        # Shape: (1, enc_seq_len, 1)
    tgt_sample = tgt_sample.unsqueeze(0).to(device)        # Shape: (1, dec_seq_len, 1)
    target_sample = target_sample.unsqueeze(0).to(device)  # Shape: (1, dec_seq_len, 1)

    with torch.no_grad():
        output_sample = model(src_sample, tgt_sample, device)  # Shape: (1, dec_seq_len, 1)

    # Denormalize
    tgt_mean = normalization_params['tgt_mean']
    tgt_std = normalization_params['tgt_std']

    output_sample_denorm = output_sample.cpu().numpy() * tgt_std + tgt_mean
    target_sample_denorm = target_sample.cpu().numpy() * tgt_std + tgt_mean

    predicted_voltage = output_sample_denorm[0, :, 0]  # Shape: (dec_seq_len,)
    actual_voltage = target_sample_denorm[0, :, 0]     # Shape: (dec_seq_len,)

    # Append sequences to the lists
    predicted_voltages.append(predicted_voltage)
    actual_voltages.append(actual_voltage)

# Convert lists to NumPy arrays
predicted_voltages = np.array(predicted_voltages)  # Shape: (num_sequences, dec_seq_len)
actual_voltages = np.array(actual_voltages)        # Shape: (num_sequences, dec_seq_len)

# Save the data into a NumPy .npz file
np.savez('validation_predictions.npz', predicted_voltages=predicted_voltages, actual_voltages=actual_voltages)

print("Saved predicted and actual voltages to 'validation_predictions.npz'")
