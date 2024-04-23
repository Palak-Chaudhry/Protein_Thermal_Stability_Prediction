# Protein Sequence Prediction using CNN

This script trains a Convolutional Neural Network (CNN) model to predict target properties from protein sequences.

## Dependencies

- Python 3.7 or higher
- PyTorch (version 1.12.0 or higher)
- NumPy
- Pandas
- Scikit-learn
- SciPy
- Matplotlib

You can install the required packages using the following command:

```
pip install -r requirements.txt
```

## Data Preparation

1. Place the training data file `train.csv` and the test data file `test.csv` in the same directory as the script.
2. The script assumes that the data files have the following structure:
   - `train.csv`: Contains a column named `sequence` with the protein sequences and a column named `target` with the target values.
   - `test.csv`: Contains a column named `sequence` with the protein sequences and a column named `id` with the unique identifiers.

## Running the Script

1. Open a terminal or command prompt and navigate to the directory containing the script.
2. Run the script using the following command:

   ```
   python protein_prediction.py
   ```

   This will execute the entire pipeline, including:
   - Loading the training and test data
   - Preprocessing the data (one-hot encoding of sequences)
   - Defining and training the CNN model
   - Evaluating the model on the validation set
   - Generating predictions for the test data
   - Saving the trained model to a file (`cnn2.pth`)
   - Saving the predictions to a CSV file (`prediction.csv`)

3. The script will print the training and validation loss, as well as the Spearman correlation coefficient for each epoch.

## Plotting Results

The script includes two functions to generate plots:

1. `plot_loss_vs_epochs`: Plots the training and validation loss over the training epochs.
2. `plot_cor_vs_epochs`: Plots the Spearman correlation coefficient over the training epochs.

The plots will be saved as `loss.png` and `cor.png`, respectively, in the same directory as the script.

## Adjusting Hyperparameters

You can modify the following hyperparameters in the script:

- `num_epochs`: The number of training epochs.
- `batch_size`: The batch size for the training and validation dataloaders.
- `learning_rate`: The learning rate for the Adam optimizer.

Feel free to experiment with these hyperparameters to see how they affect the model's performance.

## Saving and Loading the Model

The trained model is saved to the file `cnn2.pth`. To load the saved model, you can use the following code:

```python
model = CNNModel().to(device)
model.load_state_dict(torch.load("cnn2.pth"))
```

Then, you can use the loaded model for inference or further fine-tuning.
