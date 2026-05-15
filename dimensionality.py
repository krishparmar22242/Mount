import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Import Dimensionality Reduction Models
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def run_dim_reduction():
    print("\n" + '='*55)
    print("--- Generating High-Dimensional Dataset (10 Features) ---")
    print('='*55)
    
    # 1. Generate 10-Dimensional Data 
    # TRICK: class_sep=2.5 pushes the clusters apart, resulting in much higher accuracy!
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=6, 
                               n_redundant=2, n_classes=3, n_clusters_per_class=1, 
                               class_sep=2.5, flip_y=0.02, random_state=42)
    
    # 2. Data Describe (EDA)
    df = pd.DataFrame(X, columns=["Feature_{}".format(i+1) for i in range(10)])
    df['Target_Class'] = y
    print("\nData Description (First 5 rows of 10-D data):")
    print(df.head().round(2))
    print("\nOriginal Dataset Shape: {}".format(X.shape))
    
    # 3. Scale the Data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 4. Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # 5. Dictionary of Dimensionality Reducers
    reducers = {
        "PCA (Principal Component Analysis)": PCA(n_components=2, random_state=42),
        "SVD (Truncated SVD)": TruncatedSVD(n_components=2, random_state=42),
        "LDA (Linear Discriminant Analysis)": LinearDiscriminantAnalysis(n_components=2)
    }

    # 6. Loop through Reducers, Transform, Evaluate, and Plot
    for name, reducer in reducers.items():
        print("\n" + '='*55)
        print("--- Model: {} ---".format(name))
        print('='*55)
        
        # Fit and Transform
        if "LDA" in name:
            reducer.fit(X_train, y_train) # LDA is Supervised
        else:
            reducer.fit(X_train)          # PCA/SVD are Unsupervised
            
        X_test_reduced = reducer.transform(X_test)
        print("Reduced Test Data Shape: {} (Compressed to 2D)".format(X_test_reduced.shape))
        
        # METRIC: Explained Variance
        variance = reducer.explained_variance_ratio_
        total_variance = np.sum(variance) * 100
        print("Explained Variance per component: [{:.4f}, {:.4f}]".format(variance[0], variance[1]))
        print("Total Information Retained: {:.2f}%".format(total_variance))
        
        # LDA Specific Classification Metrics
        if "LDA" in name:
            preds = reducer.predict(X_test)
            acc = accuracy_score(y_test, preds)
            print("\n=> LDA Classification Accuracy: {:.4f} ({:.2f}%)".format(acc, acc*100))
            print("LDA Classification Report:\n{}".format(classification_report(y_test, preds)))

        # ==========================================
        # VISUALIZATION (Single Scatter Plot Only)
        # ==========================================
        plt.figure(figsize=(8, 5))
        scatter = plt.scatter(X_test_reduced[:, 0], X_test_reduced[:, 1], c=y_test, 
                              cmap='Set1', edgecolor='k', alpha=0.7, s=60)
        
        plt.title("{}\n10 Features Compressed to 2 (Retained: {:.1f}%)".format(name, total_variance))
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        
        # Legend setup
        handles, _ = scatter.legend_elements()
        plt.legend(handles, ['Class 0', 'Class 1', 'Class 2'], title="True Classes")
        plt.grid(True, linestyle='--', alpha=0.5)
        
        plt.show() # Close window to proceed to next model

run_dim_reduction()