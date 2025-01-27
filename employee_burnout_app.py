import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import streamlit as st

# Set the title of the app
st.title("Employee Burnout Prediction App")

# File upload widget
uploaded_file = st.file_uploader("Upload your Excel or CSV file", type=["xlsx", "csv"])

# Check if file is uploaded
if uploaded_file is not None:
    try:
        # Load the dataset based on file extension (Excel or CSV)
        if uploaded_file.name.endswith(".xlsx"):
            data = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith(".csv"):
            data = pd.read_csv(uploaded_file)

        # Display basic information about the dataset
        st.write("Dataset loaded successfully!")
        st.write("Dataset Overview:")
        st.write(data.head())
        st.write("Missing Values:")
        st.write(data.isnull().sum())

        # Step 3: Drop rows with missing values
        data = data.dropna()
        st.write(f"Shape after dropping missing values: {data.shape}")

        # Step 4: Drop unnecessary columns (if they exist)
        if 'Employee ID' in data.columns:
            data = data.drop('Employee ID', axis=1)
        if 'Date of Joining' in data.columns:
            data = data.drop('Date of Joining', axis=1)

        # Step 5: Correlation with target variable (Burn Rate)
        if 'Burn Rate' in data.columns:
            st.write("Correlation with Burn Rate:")
            st.write(data.corr(numeric_only=True)['Burn Rate'][:-1])  # Correlation with all features except 'Burn Rate'

        # Step 6: Encode categorical columns
        categorical_columns = ['Company Type', 'WFH Setup Available', 'Gender']
        data = pd.get_dummies(data, columns=[col for col in categorical_columns if col in data.columns], drop_first=True)

        # Step 7: Split the data into features (X) and target (y)
        if 'Burn Rate' in data.columns:
            y = data['Burn Rate']  # Target variable
            X = data.drop('Burn Rate', axis=1)  # Features
        else:
            st.error("Target variable 'Burn Rate' is missing.")
            st.stop()

        # Step 8: Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=1)
        st.write("Train and Test data split completed.")

        # Step 9: Scale the features (Standardization)
        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

        # Step 10: Train the SVM regression model
        regressor = SVR(kernel='rbf')
        regressor.fit(X_train, y_train)

        # Step 11: Predictions and performance evaluation
        y_pred = regressor.predict(X_test)

        # Step 12: Model performance metrics
        st.write("Model Performance Metrics:")
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.write(f"Mean Squared Error (MSE): {mse:.4f}")
        st.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        st.write(f"Mean Absolute Error (MAE): {mae:.4f}")
        st.write(f"R-Squared Score (R²): {r2:.4f}")

        # Step 13: Visualization of Actual vs Predicted Burn Rate
        st.write("Actual vs Predicted Burn Rates:")
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, alpha=0.7, color='blue')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
        plt.title("Actual vs Predicted Burn Rates")
        plt.xlabel("Actual Burn Rate")
        plt.ylabel("Predicted Burn Rate")
        st.pyplot()

    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Please upload an Excel or CSV file to get started.")
