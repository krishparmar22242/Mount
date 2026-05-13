import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

# Import Models
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

# XGBoost requires separate installation. This try-except prevents the exam from crashing!
try:
    from xgboost import XGBClassifier
    has_xgb = True
except ImportError:
    has_xgb = False
    print("WARNING: xgboost not installed on this PC. Skipping XGBoost.")

def run_classification():
    print(f"\n{'='*50}")
    print(f"--- Generating Classification Dataset ---")
    print(f"{'='*50}")
    
    # 1. Generate Synthetic Data (flip_y=0.1 adds 10% noise to the labels)
    X, y = make_classification(n_samples=800, n_features=5, n_informative=3, 
                               n_redundant=1, n_classes=2, flip_y=0.1, random_state=42)
    
    # 2. Data Describe (EDA)
    df = pd.DataFrame(X, columns=[f"Feature_{i+1}" for i in range(5)])
    df['Target_Class'] = y
    print("\nData Description (First 5 rows):")
    print(df.head().round(2))
    print("\nStatistical Summary:")
    print(df.describe().round(2))
    print("\nClass Distribution:\n",df['Target_Class'].value_counts())
    
    # 3. Scale the Data (CRUCIAL for KNN and SVM)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 4. Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # 5. Dictionary of all required models
    models = {
        "Decision Tree (Gini)": DecisionTreeClassifier(criterion='gini', random_state=42),
        "Decision Tree (Entropy)": DecisionTreeClassifier(criterion='entropy', random_state=42),
        "KNN (K=5)": KNeighborsClassifier(n_neighbors=5),
        "SVM (Linear Kernel)": SVC(kernel='linear', random_state=42),
        "SVM (Poly Kernel)": SVC(kernel='poly', random_state=42),
        "SVM (RBF Kernel)": SVC(kernel='rbf', random_state=42),
        "SVM (Sigmoid Kernel)": SVC(kernel='sigmoid', random_state=42),
        "Bagging (Random Forest)": RandomForestClassifier(n_estimators=50, random_state=42),
        "Boosting (AdaBoost)": AdaBoostClassifier(n_estimators=50, random_state=42),
        "Boosting (Gradient)": GradientBoostingClassifier(n_estimators=50, random_state=42)
    }
    
    if has_xgb:
        models["Boosting (XGBoost)"] = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

    # 6. Loop through each model, Train, Evaluate, and Plot!
    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"--- Model: {name} ---")
        print(f"{'='*50}")
        
        # Train and Predict
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        # Metrics
        acc = accuracy_score(y_test, preds)
        print(f"=> RESULT: Accuracy: {acc:.4f}")
        
        # Actual vs Predicted Table (First 10 values)
        comparison_df = pd.DataFrame({'Actual Class': y_test, 'Predicted Class': preds})
        # Add a column that says True if they match, False if they don't
        comparison_df['Correct?'] = comparison_df['Actual Class'] == comparison_df['Predicted Class']
        
        print(f"\n--- {name}: Actual vs Predicted (First 10) ---")
        print(comparison_df.head(10))
        
        print(f"\nClassification Report:\n{classification_report(y_test, preds)}")
        
        # Confusion Matrix Visualization
        ConfusionMatrixDisplay.from_predictions(y_test, preds, cmap='Blues')
        plt.title(f"{name}\nAccuracy: {acc:.2f}")
        plt.show()  # Closes the loop. You must close the graph window in VS Code to see the next model!

# Run the master script
run_classification()