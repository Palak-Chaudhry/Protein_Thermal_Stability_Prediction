##--------------------------------------------
## Import Libraries
##--------------------------------------------
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader, TensorDataset
import torch.nn.functional as F
import numpy as np
import re
import pandas as pd
import sklearn.linear_model
from scipy import stats
import matplotlib.pyplot as plt

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

## Create a vocab of all amino acids present
vocab = "ARNDCQEGHILKMFPSTWYVXU"

##----------------------------------------------
## Data Pre-processing functions
##----------------------------------------------

def load_train_data(path, val_split=False):
    """
        def - Loads training data
        path: path of stored training data
        val_split: Boolean parameter to check whether validation split is required
        Example -
        train, val = load_train_data("train.csv", val_split=True)
    """
    df = pd.read_csv(path)
    df.sequence = df.sequence.apply(
        lambda s: re.sub(r"[^A-Z]", "", s.upper())
        )  # remove special characters

    if val_split:
            num_rows_to_select = int(0.2 * len(df))
            val = df.iloc[:num_rows_to_select]
            train = df.iloc[num_rows_to_select:]
            return train, val
    else:
            return df

def load_test_data(path):
    """
        def - loads test data
        path: path of stored test data
        Example -
        test = load_test_data("test.csv")
    """
    df = pd.read_csv(path)
    df.sequence = df.sequence.apply(
            lambda s: re.sub(r"[^A-Z]", "", s.upper())
        )  # remove special characters
    return df


def one_hot_pad_seqs(s, length, vocab=vocab):
    """
        def - performs one hot encoding of protein sequence using amino acid vocab
        s: list of sequences
        length: length of sequence
        Example -
        X = one_hot_pad_seqs(s, length)
        ## Final shape of X -> (length, len(vocab))
    """
    aa_dict = {k: v for v, k in enumerate(vocab)}
    embedded = np.zeros([length, len(vocab)])
    for i, l in enumerate(s):
        if i >= length:
            break
        idx = aa_dict[l]
        embedded[i, idx] = 1
    return embedded

def get_seq(df, length=1500):
    """
        def - returns list of one hot encoded sequences (with padding if required)
        df: dataframe containing sequences
        length: length of sequence
        Example -
        X = one_hot_pad_seqs(s, length)
    """
    seq = df.sequence.values.tolist()
    X = [one_hot_pad_seqs(s, length) for s in seq]
    return X

def flat(w, k, s):
    """
        def - function to return final output of conv layer
        w: height/width of layer
        k: kernel size
        s: stride
        Example - 
        h = flat(2500,8,1)
    """
    x = np.floor((w - k) / s) + 1
    y = np.floor((x - 2) / 2) + 1
    return int(y)

##----------------------------------------------
## Generate plots for evaluation
##----------------------------------------------

def plot_loss_vs_epochs(train_loss, val_loss):
        """
	Plots the loss vs epochs graph for three different loss values.

        Parameters:
        train_loss (list): A list of loss values for the first graph.
        loss_values2 (list): A list of loss values for the second graph.
        loss_values3 (list): A list of loss values for the third graph.
        """
	# Create the figure and axis
        fig, ax = plt.subplots(figsize=(12, 6))
        # Plot the three loss vs epochs graphs
        ax.plot(range(1, len(train_loss) + 1), train_loss, marker='o', label='Train Loss')
        ax.plot(range(1, len(val_loss) + 1), val_loss, marker='s', label='Val loss')
        # Set the labels and title
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Loss vs Epochs')
        ax.legend()
        # Show the plot
        plt.savefig("loss.png")


def plot_cor_vs_epochs(cor_values):
        """
	Plots the loss vs epochs graph given a list of loss values.

        Parameters:
        loss_values (list): A list of loss values, where each value corresponds to the loss for a single epoch.
        """
	# Create the figure and axis
        fig, ax = plt.subplots(figsize=(8, 6))
        # Plot the loss vs epochs graph
        ax.plot(range(1, len(cor_values) + 1), cor_values, marker='o', label = 'Correlation score')
        # Set the labels and title
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Correlation score')
        ax.set_title('Correlation score vs Epochs')
        # Show the plot
        plt.savefig("cor.png")

##----------------------------------------------
## Model Architecture
    # CNNModel(
    # (conv1): Conv2d(1, 32, kernel_size=(8, 4), stride=(1, 1))
    # (bn1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # (conv2): Conv2d(32, 64, kernel_size=(5, 2), stride=(1, 1))
    # (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # (conv3): Conv2d(64, 128, kernel_size=(3, 2), stride=(1, 1))
    # (bn3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    # (dropout): Dropout(p=0.25, inplace=False)
    # (flatten): Flatten(start_dim=1, end_dim=-1)
    # (fc1): Linear(in_features=23552, out_features=512, bias=True)
    # (fc3): Linear(in_features=512, out_features=1, bias=True)
    # (relu): ReLU()
    # )
##----------------------------------------------

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        # Define convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(8,4))
        self.bn1 = nn.BatchNorm2d(32)  # Batch normalization after conv1
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5,2))
        self.bn2 = nn.BatchNorm2d(64)  # Batch normalization after conv2
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,2))
        self.bn3 = nn.BatchNorm2d(128)  # Batch normalization after conv3
        self.pool = nn.MaxPool2d(kernel_size=2)  # Max pooling layer
        self.dropout = nn.Dropout(p=0.25)  # Dropout layer
        self.flatten = nn.Flatten()  # Flatten layer to transition to fully connected layers
        h = flat(flat(flat(1500,8,1),5,1),3,1)  # Calculate height after convolutions
        w = flat(flat(flat(22,4,1),2,1),2,1)  # Calculate width after convolutions
        self.fc1 = nn.Linear(h * w * 128, 512)  # Fully connected layer 1
        self.fc3 = nn.Linear(512, 1)  # Fully connected layer 3 (output layer)
        self.relu = nn.ReLU()  # ReLU activation function
    
    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.conv1(x)  # Convolutional layer 1
        x = self.bn1(x)  # Batch normalization
        x = self.relu(x)  # ReLU activation
        x = self.pool(x)  # Max pooling
        x = self.conv2(x)  # Convolutional layer 2
        x = self.bn2(x)  # Batch normalization
        x = self.relu(x)  # ReLU activation
        x = self.pool(x)  # Max pooling
        x = self.conv3(x)  # Convolutional layer 3
        x = self.bn3(x)  # Batch normalization
        x = self.relu(x)  # ReLU activation
        x = self.pool(x)  # Max pooling
        x = self.flatten(x)  # Flatten the output for fully connected layers
        x = self.relu(self.fc1(x))  # Fully connected layer 1 with ReLU activation
        x = self.dropout(x)  # Dropout layer
        x = self.fc3(x)  # Fully connected layer 3 (output layer)
        return x

##----------------------------------------------
## Main Function
##----------------------------------------------


if __name__ == "__main__":

    ## Load data
    train, val = load_train_data("train.csv", val_split=True)
    test = load_test_data("test.csv")

    ## Pre-process data into one hot encoding
    x_train, x_val, x_test = get_seq(train), get_seq(val), get_seq(test)

    # Create PyTorch datasets and dataloaders for training and validation

    train_target = torch.tensor(train['target'].values.astype(np.float32))
    train_tensor = TensorDataset(torch.tensor(x_train), train_target)
    train_loader = DataLoader(dataset = train_tensor, batch_size = 500, shuffle = True)

    val_target = torch.tensor(val['target'].values.astype(np.float32))
    val_tensor = TensorDataset(torch.tensor(x_val), val_target)
    val_loader = DataLoader(dataset = val_tensor, batch_size = 500, shuffle = True)

    
    # Define model, loss function, and optimizer
    model = CNNModel().to(device)  # Initialize CNN model
    print("Model defined")
    criterion = nn.MSELoss()  # Mean Squared Error loss
    optimizer = optim.Adam(model.parameters(), lr=1e-4)  # Adam optimizer

    #Initialise variables to store evaluation metrics
    train__loss = []
    test__loss = []
    correlation = []

    # Train the model
    num_epochs = 100
    for epoch in range(num_epochs):
        print("Epoch started")
        model.train()
        train_loss = 0.0
        for sequences, scores in train_loader:
            optimizer.zero_grad()
            sequences = sequences.float().to(device)
            scores = scores.float().to(device)
            outputs = model(sequences)
            loss = criterion(outputs.squeeze(), scores)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(sequences)

        train_loss /= len(x_train)

        # Validation
        model.eval()
        val_loss = 0.0
        n = 0
        cor = 0.0
        with torch.no_grad():
            for sequences, scores in val_loader:
                n += 1
                sequences = sequences.float().to(device)
                scores = scores.float().to(device)
                outputs = model(sequences)
                # Calculate Spearman correlation coefficient for validation set
                correlation_coefficient, p_value = stats.spearmanr(outputs.squeeze().detach().cpu().numpy(), scores.detach().cpu().numpy())
                print("Spearman correlation coefficient:", correlation_coefficient)
                print("P-value:", p_value)
                loss = criterion(outputs.squeeze(), scores)
                val_loss += loss.item() * len(sequences)
                cor += correlation_coefficient

        val_loss /= len(x_val)
        cor /= n

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Cor: {cor:.4f}')
        train__loss.append(train_loss)
        test__loss.append(val_loss)
        correlation.append(cor)
    # Save the trained model
    torch.save(model.state_dict(), "cnn2.pth")
    print("Saved PyTorch Model State to cnn2.pth")

    #Plot graphs
    plot_loss_vs_epochs(train__loss, test__loss)
    plot_cor_vs_epochs(correlation)

    # Generate predictions for the test data
    test_id = test.id.values.tolist()
    model.eval()
    predictions = []
    with torch.no_grad():
        for sequences in x_test:
            sequences = torch.tensor(sequences).float().unsqueeze(0).to(device)
            output = model(sequences)
            predictions.append(output.item())

    # Save predictions to a CSV file
    with open("prediction.csv", "w") as f:
        f.write("id,target\n")
        for id, y in zip(test_id, predictions):
            f.write(f"{id},{y}\n")


