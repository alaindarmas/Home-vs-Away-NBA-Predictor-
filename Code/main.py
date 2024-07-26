import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import os

# Define the base path (current script directory)
base_path = os.path.dirname(os.path.abspath(__file__))

# Define the relative paths to the configuration file
config_file_path = os.path.join(base_path, '..', 'Documentation', 'ConfigFile.xlsx')

# Load the configuration file
config = pd.read_excel(config_file_path)
dataset_path = config.loc[config['Key'] == 'dataset_path', 'Value'].values[0]
output_path = config.loc[config['Key'] == 'output_path', 'Value'].values[0]

# Convert relative paths to absolute paths
dataset_path = os.path.normpath(os.path.join(os.path.dirname(config_file_path), dataset_path))
output_path = os.path.normpath(os.path.join(os.path.dirname(config_file_path), output_path))

# Load the dataset
data = pd.read_csv(dataset_path)

# Remove rows with missing values
data_cleaned = data.dropna()

# Create a Home_Away column
data_cleaned.loc[:, 'Home_Away'] = (data_cleaned['PTS_home'] > 0).astype(int)  # 1 for home, 0 for away

# Select relevant features: FG3_PCT and Home/Away indicator
data_cleaned.loc[:, 'FG3_PCT'] = data_cleaned.apply(lambda row: row['FG3_PCT_home'] if row['Home_Away'] == 1 else row['FG3_PCT_away'], axis=1)
X = data_cleaned[['FG3_PCT', 'Home_Away']]
y = data_cleaned['HOME_TEAM_WINS']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the feature data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a logistic regression model
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Cross-validation
cv_scores = cross_val_score(model, X, y, cv=5)

# Feature importance (coefficients)
feature_importance = model.coef_[0]
feature_importance_df = pd.DataFrame({'Feature': ['FG3_PCT'], 'Importance': [feature_importance[0]]}).sort_values(by='Importance', ascending=False)

# Calculate the likelihood of a home team winning
home_team_win_likelihood = y.mean()

# Function to generate random predictions within realistic ranges
def generate_random_predictions(num_predictions=10):
    random_data = np.random.rand(num_predictions, 2)
    q1_fg3 = X['FG3_PCT'].quantile(0.25)
    q3_fg3 = X['FG3_PCT'].quantile(0.75)
    iqr_fg3 = q3_fg3 - q1_fg3
    min_fg3 = max(X['FG3_PCT'].min(), q1_fg3 - 1.5 * iqr_fg3)
    max_fg3 = min(X['FG3_PCT'].max(), q3_fg3 + 1.5 * iqr_fg3)
    random_data[:, 0] = random_data[:, 0] * (max_fg3 - min_fg3) + min_fg3
    random_data[:, 1] = (random_data[:, 1] > 0.5).astype(int)  # Randomly choose between 0 and 1 for Home

    random_data_scaled = scaler.transform(pd.DataFrame(random_data, columns=['FG3_PCT', 'Home_Away']))
    random_predictions = model.predict(random_data_scaled)
    random_results_df = pd.DataFrame(random_data, columns=['FG3_PCT', 'Home_Away'])
    random_results_df.rename(columns={'Home_Away': 'Home'}, inplace=True)
    random_results_df['Win'] = random_predictions
    return random_results_df

random_results_df = generate_random_predictions()

# Write results to Excel
with pd.ExcelWriter(output_path) as writer:
    feature_importance_df.to_excel(writer, sheet_name='Feature Importance', index=False)
    pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Home Team Win Likelihood'],
        'Value': [accuracy, precision, recall, f1, home_team_win_likelihood]
    }).to_excel(writer, sheet_name='Model Evaluation', index=False)
    random_results_df.to_excel(writer, sheet_name='Random Predictions', index=False)

# Function to ask for custom inputs with realistic ranges
def get_custom_input():
    custom_data = {}
    q1_fg3 = X['FG3_PCT'].quantile(0.25)
    q3_fg3 = X['FG3_PCT'].quantile(0.75)
    iqr_fg3 = q3_fg3 - q1_fg3
    min_fg3 = max(X['FG3_PCT'].min(), q1_fg3 - 1.5 * iqr_fg3)
    max_fg3 = min(X['FG3_PCT'].max(), q3_fg3 + 1.5 * iqr_fg3)
    
    home_away = input("Enter 1 for home or 0 for away: ").strip()
    custom_data['Home_Away'] = int(home_away)
    
    fg3_pct = float(input(f"Enter value for FG3_PCT (range: {min_fg3:.2f} - {max_fg3:.2f}): ").strip())
    custom_data['FG3_PCT'] = fg3_pct

    custom_df = pd.DataFrame([custom_data], columns=['FG3_PCT', 'Home_Away'])
    custom_df_scaled = scaler.transform(custom_df)
    custom_prediction = model.predict(custom_df_scaled)[0]

    custom_df['Win'] = custom_prediction
    custom_df.rename(columns={'Home_Away': 'Home'}, inplace=True)
    return custom_df

# Main loop
choice = input("Do you want to make a custom prediction? (y/n): ").strip().lower()
if choice == 'y':
    custom_df = get_custom_input()
    print(f"Custom input prediction: {'Win' if custom_df['Win'].iloc[0] == 1 else 'Loss'}")
    with pd.ExcelWriter(output_path, mode='a', if_sheet_exists='new') as writer:
        custom_df.to_excel(writer, sheet_name='Custom Prediction', index=False)

# Print results for confirmation
print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')
print(f'Home Team Win Likelihood: {home_team_win_likelihood:.2f}')
print('Cross-validation scores:', cv_scores)
print('Feature importance:')
print(feature_importance_df)
print('Random predictions:')
print(random_results_df)
