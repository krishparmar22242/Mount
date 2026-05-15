import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

# Import Boosting Models
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier

# XGBoost requires separate installation. This try-except prevents the exam from crashing!
try:
    from xgboost import XGBClassifier
    has_xgb = True
except ImportError:
    has_xgb = False
    print("WARNING: xgboost not installed on this PC. Skipping XGBoost.")

def run_boosting_experiments():
    print("\n" + '='*60)
    print("--- RUNNING BOOSTING EXPERIMENTS (SUPERVISED) ---")
    print('='*60)
    
    # 1. Generate Synthetic Data with Noise (flip_y=0.1)
    X, y = make_classification(n_samples=800, n_features=5, n_informative=3, 
                               n_redundant=1, n_classes=2, flip_y=0.1, random_state=42)
    
    # 2. Data Describe (EDA)
    feature_names = ["Feature_{}".format(i+1) for i in range(5)]
    df = pd.DataFrame(X, columns=feature_names)
    df['Target_Class'] = y
    
    print("\nData Description (First 5 rows):")
    print(df.head().round(2))
    print("\nStatistical Summary:")
    print(df.describe().round(2))
    print("\nClass Distribution:\n", df['Target_Class'].value_counts())
    
    # 3. Scale the Data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 4. Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # 5. Dictionary of Boosting Models
    models = {
        "AdaBoost": AdaBoostClassifier(n_estimators=50, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=50, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }
    
    # 6. Loop through each model, Train, Evaluate, and Plot
    for name, model in models.items():
        print("\n" + '='*50)
        print("--- Boosting Algorithm: {} ---".format(name))
        print('='*50)
        
        # Train and Predict
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        # 1. Accuracy Metric
        acc = accuracy_score(y_test, preds)
        print("=> RESULT: Accuracy: {:.4f} ({:.2f}%)".format(acc, acc*100))
        
        # 2. Actual vs Predicted Table (First 10 values)
        comparison_df = pd.DataFrame({'Actual Class': y_test, 'Predicted Class': preds})
        comparison_df['Correct?'] = (comparison_df['Actual Class'] == comparison_df['Predicted Class'])
        
        print("\n--- {}: Actual vs Predicted Samples (First 10) ---".format(name))
        print(comparison_df.head(10))
        
        # 3. Full Classification Report (Precision, Recall, F1)
        print("\nFull Classification Report for {}:\n".format(name))
        print(classification_report(y_test, preds))
        
        # 4. Feature Importance Plot (Special for Boosting)
        # Boosting models allow us to see which features were most useful
        plt.figure(figsize=(8, 4))
        importances = model.feature_importances_
        plt.barh(feature_names, importances, color='skyblue', edgecolor='black')
        plt.title("Feature Importances: {}".format(name))
        plt.xlabel("Importance Score")
        plt.show()

        # 5. Confusion Matrix Visualization
        ConfusionMatrixDisplay.from_predictions(y_test, preds, cmap='Reds')
        plt.title("Confusion Matrix: {}".format(name))
        plt.show()

# Execute the boosting script
if __name__ == "__main__":
    run_boosting_experiments()