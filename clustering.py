import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Import Models
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture

# K-Medoids requires scikit-learn-extra. Try-except prevents crashing!
try:
    from sklearn_extra.cluster import KMedoids
    has_kmedoids = True
except ImportError:
    has_kmedoids = False
    print("WARNING: scikit-learn-extra not installed. Skipping K-Medoids.")

def run_clustering():
    print(f"\n{'='*50}")
    print(f"--- Generating Clustering Dataset ---")
    print(f"{'='*50}")
    
    # 1. Generate Synthetic Data (cluster_std=1.2 adds spread/noise to make it challenging)
    # Using 2 features makes it perfectly plottable on a 2D graph!
    X, true_labels = make_blobs(n_samples=600, centers=4, n_features=2, 
                                cluster_std=1.2, random_state=42)
    
    # 2. Data Describe (EDA)
    df = pd.DataFrame(X, columns=['Feature_1', 'Feature_2'])
    print("\nData Description (First 5 rows):")
    print(df.head().round(2))
    print("\nStatistical Summary:")
    print(df.describe().round(2))
    
    # 3. Scale the Data (ABSOLUTELY CRUCIAL for distance-based K-Means/DBSCAN)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 4. Dictionary of Unsupervised Models
    # Note: K-Means/Medoids/EM require us to guess 'n_clusters'. DBSCAN finds it automatically using 'eps'.
    models = {
        "K-Means": KMeans(n_clusters=4, random_state=42, n_init='auto'),
        "DBSCAN": DBSCAN(eps=0.35, min_samples=5),
        "EM (Gaussian Mixture)": GaussianMixture(n_components=4, random_state=42)
    }
    
    if has_kmedoids:
        models["K-Medoids"] = KMedoids(n_clusters=4, random_state=42)

    # 5. Loop through models, Train, Evaluate, and Plot
    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"--- Model: {name} ---")
        print(f"{'='*50}")
        
        # Fit and Predict (EM uses .predict, others use .fit_predict)
        if name == "EM (Gaussian Mixture)":
            model.fit(X_scaled)
            labels = model.predict(X_scaled)
        else:
            labels = model.fit_predict(X_scaled)
            
        # Count number of clusters found (Ignoring -1 which is noise in DBSCAN)
        n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_points = list(labels).count(-1)
        
        print(f"Clusters Found: {n_clusters_found}")
        if n_noise_points > 0:
            print(f"Noise Points Detected: {n_noise_points}")
        
        # Calculate Metrics (Fails if DBSCAN groups everything into 1 cluster or pure noise)
        try:
            sil_score = silhouette_score(X_scaled, labels)
            db_score = davies_bouldin_score(X_scaled, labels)
            print(f"=> Silhouette Score: {sil_score:.4f} (Closer to 1 is better)")
            print(f"=> Davies-Bouldin Index: {db_score:.4f} (Closer to 0 is better)")
        except ValueError:
            print("=> Metrics Error: Model found only 1 cluster or classified everything as noise.")

        # ==========================================
        # VISUALIZATION
        # ==========================================
        plt.figure(figsize=(8, 5))
        
        # Scatter plot colored by the predicted cluster labels
        # Using cmap='viridis'. If label is -1 (noise), it gets handled naturally or we can force it black.
        scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='plasma', 
                              edgecolor='k', alpha=0.7, s=50)
        
        # Plotting the Centroids (Centers) for specific algorithms
        if name == "K-Means" or name == "K-Medoids":
            centers = model.cluster_centers_
            plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='X', 
                        linewidths=3, label='Centroids/Medoids')
            plt.legend()
        elif name == "EM (Gaussian Mixture)":
            centers = model.means_
            plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='X', 
                        linewidths=3, label='Gaussian Means')
            plt.legend()
            
        # DBSCAN doesn't have centers, but it has noise
        if name == "DBSCAN" and n_noise_points > 0:
            plt.title(f"{name} Clustering (Black/Dark points are Noise)")
        else:
            plt.title(f"{name} Clustering")
            
        plt.xlabel("Feature 1 (Scaled)")
        plt.ylabel("Feature 2 (Scaled)")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()  # Close graph window to move to the next model!

# Run the master script
run_clustering()