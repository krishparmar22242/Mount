import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def run_regression(is_multiple=False):
    regression_type = 'Multiple' if is_multiple else 'Simple'
    print("\n")
    print('=' * 50)

    print("--- Running {} Linear Regression ---".format(regression_type))
    print('=' * 50)

    
    # 1. Generate Synthetic Data with Noise
    # 1 feature for Simple, 5 features for Multiple
    n_features = 5 if is_multiple else 1
    X, y = make_regression(n_samples=500, n_features=n_features, noise=15.0, random_state=42)
    
    # 2. Data Describe (EDA)
    df = pd.DataFrame(X, columns=["Feature_{}".format(i+1) for i in range(n_features)])
    df['Target'] = y
    print("\nData Description (First 5 rows):")
    print(df.head().round(2))
    print("\nStatistical Summary:")
    print(df.describe().round(2))
    
    # 3. Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Initialize and Train
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # 5. Predict and Evaluate
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print("\n=> RESULT: {} Regression RMSE: {:.2f}".format(regression_type, rmse))
    
    # 5b. PRINT ACTUAL VS PREDICTED VALUES
    # We put them side-by-side in a DataFrame so it looks clean in the terminal
    comparison_df = pd.DataFrame({'Actual Target': y_test, 'Predicted Target': preds})
    print(comparison_df.head(10).round(2)) 
    # Note: If your teacher wants to see all 100 testing values, change .head(10) to just .to_string()
    
    # 6. Visualization
    plt.figure(figsize=(8, 5))
    
    if not is_multiple:
        # Plot for SIMPLE Regression (Feature vs Target)
        plt.scatter(X_test, y_test, color='blue', label='Actual Data', alpha=0.6)
        plt.plot(X_test, preds, color='red', linewidth=2, label='Regression Line')
        plt.xlabel("Feature_1")
        plt.ylabel("Target")
        plt.title("Simple Linear Regression (RMSE: {:.2f})".format(rmse))
    else:
        # Plot for MULTIPLE Regression (Actual vs Predicted)
        plt.scatter(y_test, preds, color='green', alpha=0.6)
        # Plotting a perfect diagonal line for reference
        min_val = min(y_test.min(), preds.min())
        max_val = max(y_test.max(), preds.max())
        plt.plot([min_val, max_val], [min_val, max_val], color='red', linewidth=2, linestyle='--', label='Perfect Prediction Line')
        
        plt.xlabel("Actual Target Values")
        plt.ylabel("Predicted Target Values")
        plt.title("Multiple Linear Regression: Actual vs Predicted (RMSE: {:.2f})".format(rmse))
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

# Run BOTH one after the other!
run_regression(is_multiple=False) # This runs Simple Regression
run_regression(is_multiple=True)  # This runs Multiple Regression


