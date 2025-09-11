import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import joblib

###############################################################################################################
#######                   Define the PyTorch model with improved trained architecture                   #######
###############################################################################################################
class NeuralNet(nn.Module):
    def __init__(self, input_dim):  # Add input_dim parameter
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 174)  # Use the parameter instead of X_train.shape[1]
        self.fc2 = nn.Linear(174, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(256, 256)
        self.fc6 = nn.Linear(256, 256)
        self.fc7 = nn.Linear(256, 512)
        self.fc8 = nn.Linear(512, 512)
        self.fc9 = nn.Linear(512, 256)
        self.fc10 = nn.Linear(256, 256)
        self.fc11 = nn.Linear(256, 256)
        self.fc12 = nn.Linear(256, 128)
        self.fc13 = nn.Linear(128, 128)
        self.fc14 = nn.Linear(128, 128)
        self.fc15 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.004)

        # Batch Normalization layers
        self.bn1 = nn.BatchNorm1d(174, momentum=0.07, eps=0.00001)
        self.bn2 = nn.BatchNorm1d(256, momentum=0.07, eps=0.00001)
        self.bn3 = nn.BatchNorm1d(256, momentum=0.07, eps=0.00001)
        self.bn4 = nn.BatchNorm1d(256, momentum=0.07, eps=0.00001)
        self.bn5 = nn.BatchNorm1d(256, momentum=0.07, eps=0.00001)
        self.bn6 = nn.BatchNorm1d(256, momentum=0.07, eps=0.00001)
        self.bn7 = nn.BatchNorm1d(512, momentum=0.07, eps=0.00001)
        self.bn8 = nn.BatchNorm1d(512, momentum=0.07, eps=0.00001)
        self.bn9 = nn.BatchNorm1d(256, momentum=0.07, eps=0.00001)
        self.bn10 = nn.BatchNorm1d(256, momentum=0.07, eps=0.00001)
        self.bn11 = nn.BatchNorm1d(256, momentum=0.07, eps=0.00001)
        self.bn12 = nn.BatchNorm1d(128, momentum=0.07, eps=0.00001)
        self.bn13 = nn.BatchNorm1d(128, momentum=0.07, eps=0.00001)
        self.bn14 = nn.BatchNorm1d(128, momentum=0.07, eps=0.00001)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.bn1(x)  # Batch Normalization
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.bn2(x)
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.bn3(x)
        x = self.dropout(x)
        x = torch.relu(self.fc4(x))
        x = self.bn4(x)
        x = self.dropout(x)
        x = torch.relu(self.fc5(x))
        x = self.bn5(x)
        x = self.dropout(x)
        x = torch.relu(self.fc6(x))
        x = self.bn6(x)
        x = self.dropout(x)
        x = torch.relu(self.fc7(x))
        x = self.bn7(x)
        x = self.dropout(x)
        x = torch.relu(self.fc8(x))
        x = self.bn8(x)
        x = self.dropout(x)
        x = torch.relu(self.fc9(x))
        x = self.bn9(x)
        x = self.dropout(x)
        x = torch.relu(self.fc10(x))
        x = self.bn10(x)
        x = self.dropout(x)
        x = torch.relu(self.fc11(x))
        x = self.bn11(x)
        x = self.dropout(x)
        x = torch.relu(self.fc12(x))
        x = self.bn12(x)
        x = self.dropout(x)
        x = torch.relu(self.fc13(x))
        x = self.bn13(x)
        x = self.dropout(x)
        x = torch.relu(self.fc14(x))
        x = self.bn14(x)
        x = self.dropout(x)
        x = self.fc15(x)
        return x


###############################################################################################################
#######                               Loading K-means clustering averages                               #######
###############################################################################################################
X_observed5_avg_loaded = np.loadtxt('X_observed5_avg.txt')
(X_observed5_avg1, X_observed5_avg2, X_observed5_avg3,
 X_observed1_avg1, X_observed1_avg2, X_observed1_avg3,
 X_observed2_avg1, X_observed2_avg2, X_observed2_avg3,
 X_observed3_avg1, X_observed3_avg2, X_observed3_avg3,
 X_observed4_avg1, X_observed4_avg2, X_observed4_avg3) = X_observed5_avg_loaded[:15]

###############################################################################################################
#######                                Loading ANN-PyTorch, RF, GB models                               #######
###############################################################################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load('final_model-best.pth', weights_only=True)
loaded_model = NeuralNet(input_dim=checkpoint['input_dim']).to(device)
loaded_model.load_state_dict(checkpoint['model_state_dict'])
loaded_model.eval()
rf_model = joblib.load('random_forest_model-best.joblib')
scaler_X_combined = joblib.load('scaler_X_combined-best.joblib')
ml_model = joblib.load('gradient_boosting_model-best.joblib')
scaler_X = joblib.load('scaler_X-best.joblib')  # For input features
scaler_y = joblib.load('scaler_y-best.joblib')  # For output (inverse transform)

###############################################################################################################
#######                 Predicting HSAFs using MHVSRs of sites listed in the control file               #######
###############################################################################################################
# Read the list of site data files and their corresponding site numbers
file_list = []
number_list = []
with open('control.txt', 'r') as f:
    for line in f:
        parts = line.strip().split()  # Split each line by whitespace
        if len(parts) >= 2:  # Make sure there are at least 2 columns
            file_list.append(parts[0])  # First column is the site filename
            number_list.append(parts[1])  # Second column is the site number

# Initialize lists to store plot data for all files
plot_data_list = []
# Process all site files (loop over the entire control files list)
for i, data_file in enumerate(file_list):
    X_new_data = np.loadtxt(data_file, skiprows=0)
    filename_number = number_list[i]

    # Prepare features
    X_new_observed0 = X_new_data[:, 0].reshape(-1, 1)                      # Frequency point increment
    X_new_observed1 = X_new_observed0 / (X_new_data[:, 1].reshape(-1, 1))  # First fundamental peak normalized by frequency
    X_new_observed2 = X_new_observed0 / (X_new_data[:, 2].reshape(-1, 1))  # First fundamental trough normalized by frequency
    X_new_observed3 = X_new_observed0 / (X_new_data[:, 3].reshape(-1, 1))  # Second peak normalized by frequency
    X_new_observed4 = X_new_observed0 / (X_new_data[:, 4].reshape(-1, 1))  # Second trough normalized by frequency
    X_new_observed6 = X_new_data[:, 5].reshape(-1, 1)                      # Average MHVSR curve
    X_new_observed7 = X_new_data[:, 6].reshape(-1, 1)                      # Average MHVSR curve - one standard deviation
    X_new_observed8 = X_new_data[:, 7].reshape(-1, 1)                      # Average MHVSR curve + one standard deviation

    # Prepare the averages for concatenation (matching the number of samples)
    n_samples = X_new_observed6.shape[0]  # Number of samples in new data

    X_observed_avg1 = np.full((n_samples, 1), X_observed5_avg1)
    X_observed_avg2 = np.full((n_samples, 1), X_observed5_avg2)
    X_observed_avg3 = np.full((n_samples, 1), X_observed5_avg3)

    X_observed_frq1 = np.full((n_samples, 1), X_observed1_avg1)
    X_observed_frq2 = np.full((n_samples, 1), X_observed1_avg2)
    X_observed_frq3 = np.full((n_samples, 1), X_observed1_avg3)

    X_observed_tro1 = np.full((n_samples, 1), X_observed2_avg1)
    X_observed_tro2 = np.full((n_samples, 1), X_observed2_avg2)
    X_observed_tro3 = np.full((n_samples, 1), X_observed2_avg3)

    X_observed_sfrq1 = np.full((n_samples, 1), X_observed3_avg1)
    X_observed_sfrq2 = np.full((n_samples, 1), X_observed3_avg2)
    X_observed_sfrq3 = np.full((n_samples, 1), X_observed3_avg3)

    X_observed_stro1 = np.full((n_samples, 1), X_observed4_avg1)
    X_observed_stro2 = np.full((n_samples, 1), X_observed4_avg2)
    X_observed_stro3 = np.full((n_samples, 1), X_observed4_avg3)

    # Concatenate features
    X_new_observed_combined = np.concatenate((
        X_new_observed1,
        X_observed_avg1, X_observed_avg2, X_observed_avg3,
        X_observed_frq1, X_observed_frq2, X_observed_frq3,
        X_observed_tro1, X_observed_tro2, X_observed_tro3,
        X_observed_sfrq1, X_observed_sfrq2, X_observed_sfrq3,
        X_observed_stro1, X_observed_stro2, X_observed_stro3,
        X_new_observed6, X_new_observed7, X_new_observed8
    ), axis=1)

    # Standardize new observed data
    X_new_observed_combined = scaler_X.transform(X_new_observed_combined)

    # Load the saved RandomForestRegressor model
    rf_model = joblib.load('random_forest_model-best.joblib')

    # Get Random Forest predictions for the new data
    rf_predictions_new = rf_model.predict(X_new_observed_combined).reshape(-1, 1)

    # Concatenate the Random Forest predictions with the new observed data
    X_new_combined_with_rf = np.concatenate((X_new_observed_combined, rf_predictions_new), axis=1)

    # Get predictions from the Gradient Boosting model
    gb_predictions = ml_model.predict(X_new_combined_with_rf).reshape(-1, 1)

    # Concatenate the Gradient Boosting predictions with the original features
    X_new_combined_with_gb = np.concatenate((X_new_observed_combined, gb_predictions), axis=1)

    # Standardize and predict
    X_new_combined_with_gb = scaler_X_combined.transform(X_new_combined_with_gb)
    X_new_tensor = torch.tensor(X_new_combined_with_gb, dtype=torch.float32).to(device)

    with torch.no_grad():
        predicted = loaded_model(X_new_tensor).cpu().numpy()

    predicted = scaler_y.inverse_transform(predicted)
    predicted = np.mean(predicted, axis=1).reshape(-1, 1)

    predicted = np.minimum(predicted, 3.0 * X_new_observed6)  # Constrained condition

    plot_data = {
        "X_new_data": X_new_data,  # Frequency increment
        "predicted": predicted,  # Predicted HSAF
        "X_new_observed6": X_new_observed6,  # Observed MHVSR curve
        "filename_number": filename_number  # Identifier for the site
    }
    plot_data_list.append(plot_data)

###############################################################################################################
#######                                   Plotting the predicted HSAF                                   #######
###############################################################################################################
nrows_plot, ncols_plot = 2, 2
fig_plot, axes_plot = plt.subplots(nrows=nrows_plot, ncols=ncols_plot, figsize=(7, 5))
axes_plot = axes_plot.flatten()
for i, plot_data in enumerate(plot_data_list[:4]):
    ax = axes_plot[i]
    X_new_data = plot_data["X_new_data"]
    predicted = plot_data["predicted"]
    X_new_observed6 = plot_data["X_new_observed6"]
    filename_number = plot_data["filename_number"]
    ax.spines['top'].set_linewidth(0.99)
    ax.spines['right'].set_linewidth(0.99)
    ax.spines['bottom'].set_linewidth(0.99)
    ax.spines['left'].set_linewidth(0.99)
    ax.plot(X_new_data[:, 0], predicted, label="HSAF", color='red', linewidth=2, linestyle='-')
    ax.plot(X_new_data[:, 0], X_new_observed6, label="MHVSR", color='blue', linewidth=1.5, linestyle='-')
    ax.set_xlim(0.1, 10)
    ax.set_ylim(0.3, max(predicted.max(), X_new_observed6.max()))
    ax.tick_params(axis='both', width=0.7, labelsize=13)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.text(0.05, 0.05, filename_number, fontsize=17, fontname='Times New Roman', transform=ax.transAxes,
            fontweight='bold')
    ax.grid(True)
    ax.legend(loc='lower center', ncol=1, fontsize=12, frameon=False, fancybox=False, shadow=False)

    row = i // ncols_plot
    col = i % ncols_plot
    if col == 0:
        ax.set_ylabel('MHVSR and HSAF', fontsize=13, fontname='Times New Roman')
    if row == nrows_plot - 1:
        ax.set_xlabel('Frequency (Hz)', fontsize=13, fontname='Times New Roman')

plt.tight_layout()
plt.savefig('combined_plot.png', dpi=300, bbox_inches='tight')
plt.show()
##########################################################################################
##########################################################################################