# Import necessary libraries
# This script needs these libraries to be installed:
#   numpy, sklearn
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import wandb
from wandb.sklearn import plot_feature_importances
from wandb.sklearn import plot_learning_curve

import pandas as pd  # For data handling
import pickle       # For saving the trained model
from sklearn.model_selection import train_test_split  # For splitting data
from sklearn.linear_model import LinearRegression  # For Fitting the model


# Load the dataset from a CSV file
df = pd.read_csv(
    '/workspaces/REST-API-4ML/app/data/raw/Ecommerce Customers.csv')

# Define the features (input) and label (output) columns
features = ['Avg. Session Length', 'Time on App', 'Time on Website',
            'Length of Membership']
label = "Yearly Amount Spent"

# Extract input features (X) and output labels (y)
X = df[features]
y = df[label]

test_size = 0.3
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                    random_state=42)


# Create a Linear Regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions using the trained model
y_pred = model.predict(X_test)

# Print the model's predictions
print(y_pred)

# Save the trained model to a file named "model.pkl"
pickle.dump(model, open("/workspaces/REST-API-4ML/app/models/model.pkl", "wb"))
# start a new wandb run and add your model hyperparameters
model_params = model.get_params()

# Assuming y_true and y_pred are your actual and predicted values
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Log metrics

# Init wand
wandb.init(project='ML-regression-model', config=model_params)
wandb.log({"MSE": mse, "R-squared": r2})
# Add additional configs to wandb
wandb.config.update({"test_size": test_size,
                    "train_len": len(X_train),
                     "test_len": len(X_test)})

# log additional visualisations to wandb
plot_learning_curve(model, X_train, y_train)
plot_feature_importances(model)


# Residuals Plot
residuals = y_test - y_pred
plt.figure(figsize=(10, 5))
plt.scatter(y_pred, residuals)
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.title("Residuals Plot")
wandb.log({"Residuals Plot": plt})

# Prediction vs Actual
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Prediction vs Actual")
wandb.log({"Prediction vs Actual": plt})

# Log model as an artifact
model_artifact = wandb.Artifact('linear_regression_model', type='model')
model_artifact.add_file('/workspaces/REST-API-4ML/app/models/model.pkl')
wandb.log_artifact(model_artifact)

# [optional] finish the wandb run, necessary in notebooks
# wandb.finish()
