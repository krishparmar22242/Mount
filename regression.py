import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def run_regression(is_multiple=False):
    regression_type = 'Multiple' if is_multiple else 'Simple'
    print("\n" + '=' * 60)
    print(f"--- Running {regression_type} House Price Prediction ---")
    print('=' * 60)

    # 1. Generate Synthetic Data
    n_features = 5 if is_multiple else 1
    X, y = make_regression(n_samples=500, n_features=n_features, noise=20.0, random_state=42)
    
    # 2. TRANSFORM TO MEANINGFUL DATA (House Prices)
    if not is_multiple:
        cols = ['Size_sqft']
        X = X * 500 + 2000
    else:
        cols = ['Size_sqft', 'Bedrooms', 'Age', 'Commute_Dist', 'Walk_Score']
        X[:, 0] = X[:, 0] * 500 + 2000
        X[:, 1] = np.abs(np.round(X[:, 1] * 1 + 3)) # 2-5 Bedrooms
        X[:, 2] = np.abs(X[:, 2] * 10 + 20)        # Age 10-30 years
        X[:, 3] = np.abs(X[:, 3] * 5 + 10)         # Commute
        X[:, 4] = np.abs(X[:, 4] * 10 + 50)        # Walk score
    
    y = np.abs(y * 2000 + 350000) # Mean Price ~$350,000
    
    # Data Description
    df = pd.DataFrame(X, columns=cols)
    df['Price_USD'] = y
    print("\nData Description (First 5 rows):")
    print(df.head().round(2))
    print("\nStatistical Summary:")
    print(df.describe().round(2))
    
    # 3. Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. DEFINE MODELS
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regression": RandomForestRegressor(n_estimators=100, random_state=42)
    }

    # 5. LOOP THROUGH MODELS
    for name, model in models.items():
        print(f"\n>>> Model: {name}")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        # Evaluation Metrics
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        print(f"RMSE: ${rmse:,.2f} | R2 Score: {r2:.4f}")
        
        # Actual vs Predicted Sample Table
        comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': preds})
        print(comparison_df.head(5).round(2))

        # 6. VISUALIZATION
        # ---------------------------------------------------------
        # PLOT A: Regression Fit
        plt.figure(figsize=(8, 5))
        
        if not is_multiple:
            # For 1D: Scatter plot + Prediction Line/Steps
            plt.scatter(X_test, y_test, color='gray', alpha=0.5, label='Actual Data')
            X_grid = np.linspace(X_test.min(), X_test.max(), 1000).reshape(-1, 1)
            y_grid = model.predict(X_grid)
            plt.plot(X_grid, y_grid, color='red' if 'Linear' in name else 'green', 
                     linewidth=3, label=f'{name} Fit')
            plt.xlabel("House Size (sqft)")
            plt.ylabel("Price (USD)")
        else:
            # For Multiple: Actual vs Predicted Scatter Plot
            plt.scatter(y_test, preds, color='blue', alpha=0.5)
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Perfect Fit')
            plt.xlabel("Actual Price")
            plt.ylabel("Predicted Price")

        plt.title(f"{name} Results\nRMSE: ${rmse:,.0f}")
        plt.legend()
        plt.ticklabel_format(style='plain', axis='both')
        plt.show()

# Run everything
run_regression(is_multiple=False) # Simple Models
run_regression(is_multiple=True)  # Multiple Models + Feature Importance