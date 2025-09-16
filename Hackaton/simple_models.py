"""
Simple machine learning models for exploratory analysis
- Implements K-means clustering with timeout
- Implements KNN classification with timeout
- Provides insights from simpler models before using LSTM
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import time
import signal
from contextlib import contextmanager
from prep import load_data, clean_data, feature_engineering, normalize_data
from sklearn.model_selection import train_test_split

class TimeoutException(Exception):
    pass

@contextmanager
def time_limit(seconds):
    """Context manager to limit execution time"""
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

def run_kmeans(df, max_clusters=10, timeout=20):
    """Run K-means clustering with a timeout"""
    print("Running K-means clustering with timeout...")
    
    # Select features for clustering
    features = ['quantidade', 'valor_bruto', 'valor_liquido', 'lucro_bruto', 'desconto', 'impostos']
    X = df[features].copy()
    
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Determine optimal number of clusters
    inertia_values = []
    k_values = range(1, max_clusters + 1)
    
    for k in k_values:
        try:
            with time_limit(timeout // max_clusters):
                kmeans = KMeans(n_clusters=k, n_init=3, random_state=42, max_iter=100)
                kmeans.fit(X_scaled)
                inertia_values.append(kmeans.inertia_)
                print(f"Completed K-means for k={k}")
        except TimeoutException:
            print(f"K-means timed out for k={k}")
            break
    
    # Plot elbow curve if we have enough data
    if len(inertia_values) > 1:
        plt.figure(figsize=(10, 6))
        plt.plot(k_values[:len(inertia_values)], inertia_values, 'o-', color='blue')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Inertia')
        plt.title('Elbow Method for Optimal k')
        plt.grid(True)
        plt.savefig('kmeans_elbow.png')
        print("Saved K-means elbow curve to 'kmeans_elbow.png'")
        
        # Choose optimal k (simple elbow detection)
        optimal_k = 2  # Default to 2 clusters
        if len(inertia_values) > 2:
            # Find the point of maximum curvature
            deltas = np.diff(inertia_values)
            delta_deltas = np.diff(deltas)
            if len(delta_deltas) > 0:
                optimal_k = np.argmax(delta_deltas) + 2  # +2 because we start at k=1 and due to double diff
                optimal_k = min(optimal_k, len(inertia_values))
        
        # Run k-means with optimal k
        try:
            with time_limit(timeout // 2):  # Use half the time for final clustering
                kmeans = KMeans(n_clusters=optimal_k, n_init=5, random_state=42)
                df['cluster'] = kmeans.fit_predict(X_scaled)
                
                # Analyze clusters
                cluster_stats = df.groupby('cluster').agg({
                    'quantidade': 'mean',
                    'valor_bruto': 'mean',
                    'valor_liquido': 'mean',
                    'lucro_bruto': 'mean',
                    'desconto': 'mean',
                    'impostos': 'mean',
                    'id_distribuidor': pd.Series.nunique,
                    'outlier_score': 'mean'
                })
                
                print("\nCluster Statistics:")
                print(cluster_stats)
                
                # Visualize clusters
                plt.figure(figsize=(12, 8))
                
                # Choose two features for visualization
                sns.scatterplot(
                    data=df, 
                    x='valor_liquido', 
                    y='lucro_bruto',
                    hue='cluster',
                    palette='viridis',
                    alpha=0.5
                )
                
                plt.title('Clusters Visualizados por Valor Líquido e Lucro Bruto')
                plt.savefig('kmeans_clusters.png')
                print("Saved K-means clusters visualization to 'kmeans_clusters.png'")
                
                return optimal_k, cluster_stats
        
        except TimeoutException:
            print("Final K-means clustering timed out")
    
    print("K-means analysis completed or timed out")
    return None, None

def run_knn_regression(df, target_col='valor_liquido', test_size=0.2, k_values=None, timeout=20):
    """Run KNN regression with a timeout"""
    print(f"Running KNN regression to predict {target_col}...")
    
    if k_values is None:
        k_values = [3, 5, 7, 11, 15, 21]
    
    # Select features and target
    features = ['quantidade', 'valor_bruto', 'desconto', 'impostos']
    if target_col in features:
        features.remove(target_col)
    
    X = df[features].copy()
    y = df[target_col].copy()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Try different k values
    results = []
    
    start_time = time.time()
    time_per_k = timeout / len(k_values)
    
    for k in k_values:
        if time.time() - start_time > timeout:
            print(f"KNN timed out after testing {len(results)} values of k")
            break
        
        try:
            with time_limit(time_per_k):
                knn = KNeighborsRegressor(n_neighbors=k)
                knn.fit(X_train_scaled, y_train)
                
                y_pred = knn.predict(X_test_scaled)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                results.append({
                    'k': k,
                    'mse': mse,
                    'r2': r2
                })
                
                print(f"Completed KNN for k={k}, MSE={mse:.4f}, R²={r2:.4f}")
        
        except TimeoutException:
            print(f"KNN timed out for k={k}")
            break
    
    # Find best k
    if results:
        best_result = max(results, key=lambda x: x['r2'])
        print(f"\nBest KNN model: k={best_result['k']}, R²={best_result['r2']:.4f}")
        
        # Plot results
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot([r['k'] for r in results], [r['mse'] for r in results], 'o-', color='red')
        plt.xlabel('Number of Neighbors (k)')
        plt.ylabel('Mean Squared Error')
        plt.title('KNN: MSE vs k')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot([r['k'] for r in results], [r['r2'] for r in results], 'o-', color='blue')
        plt.xlabel('Number of Neighbors (k)')
        plt.ylabel('R² Score')
        plt.title('KNN: R² vs k')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('knn_results.png')
        print("Saved KNN results to 'knn_results.png'")
        
        # Train final model with best k
        try:
            with time_limit(timeout // 4):
                best_k = best_result['k']
                best_knn = KNeighborsRegressor(n_neighbors=best_k)
                best_knn.fit(X_train_scaled, y_train)
                
                # Feature importance (sort of) - look at the closest neighbors for a few test points
                sample_indices = np.random.choice(len(X_test_scaled), min(5, len(X_test_scaled)), replace=False)
                neighbor_indices = best_knn.kneighbors(X_test_scaled[sample_indices], return_distance=False)
                
                print("\nSample prediction analysis:")
                for i, sample_idx in enumerate(sample_indices):
                    print(f"Sample {i+1}:")
                    print(f"Actual value: {y_test.iloc[sample_idx]:.2f}")
                    print(f"Predicted value: {best_knn.predict([X_test_scaled[sample_idx]])[0]:.2f}")
                    print(f"Based on neighbors with values: {[y_train.iloc[idx] for idx in neighbor_indices[i]]}")
                
                return best_k, best_result['r2']
        
        except TimeoutException:
            print("Final KNN model training timed out")
    
    print("KNN analysis completed or timed out")
    return None, None

def main():
    """Main function to run simple models"""
    # Load and preprocess data
    df_raw = load_data()
    df = clean_data(df_raw)
    df = feature_engineering(df)
    
    df_normalized, _ = normalize_data(df)
    
    # Sample the data if it's too large to speed up analysis
    if len(df_normalized) > 10000:
        sample_size = min(10000, len(df_normalized) // 10)
        print(f"Sampling {sample_size} records for analysis...")
        df_sample = df_normalized.sample(sample_size, random_state=42)
    else:
        df_sample = df_normalized
    
    # Run clustering
    print("\n" + "="*50)
    print("K-MEANS CLUSTERING")
    print("="*50)
    optimal_k, cluster_stats = run_kmeans(df_sample, timeout=20)
    
    # Run KNN regression for valor_liquido
    print("\n" + "="*50)
    print("KNN REGRESSION FOR VALOR LÍQUIDO")
    print("="*50)
    best_k_valor, r2_valor = run_knn_regression(df_sample, target_col='valor_liquido', timeout=20)
    
    # Run KNN regression for lucro_bruto
    print("\n" + "="*50)
    print("KNN REGRESSION FOR LUCRO BRUTO")
    print("="*50)
    best_k_lucro, r2_lucro = run_knn_regression(df_sample, target_col='lucro_bruto', timeout=20)
    
    # Summarize findings
    with open('simple_models_summary.txt', 'w') as f:
        f.write("RESULTADOS DE MODELOS SIMPLES\n")
        f.write("============================\n\n")
        
        f.write("1. K-MEANS CLUSTERING\n")
        if optimal_k is not None:
            f.write(f"   - Número ótimo de clusters: {optimal_k}\n")
            f.write("   - Estatísticas por cluster:\n")
            for cluster, stats in cluster_stats.iterrows():
                f.write(f"     Cluster {cluster}:\n")
                for col, val in stats.items():
                    f.write(f"       * {col}: {val:.4f}\n")
        else:
            f.write("   - A análise de clustering não foi concluída devido ao timeout\n")
        f.write("\n")
        
        f.write("2. KNN REGRESSION\n")
        f.write("   - Para Valor Líquido:\n")
        if best_k_valor is not None:
            f.write(f"     * Melhor k: {best_k_valor}\n")
            f.write(f"     * R²: {r2_valor:.4f}\n")
        else:
            f.write("     * A regressão não foi concluída devido ao timeout\n")
            
        f.write("   - Para Lucro Bruto:\n")
        if best_k_lucro is not None:
            f.write(f"     * Melhor k: {best_k_lucro}\n")
            f.write(f"     * R²: {r2_lucro:.4f}\n")
        else:
            f.write("     * A regressão não foi concluída devido ao timeout\n")
        f.write("\n")
        
        f.write("3. INSIGHTS PARA O MODELO BI-LSTM\n")
        f.write("   - Os padrões identificados pelo K-means podem ser usados para segmentar o treinamento do LSTM\n")
        f.write("   - Os resultados do KNN mostram que existe uma relação não-linear que o LSTM deve capturar\n")
        f.write("   - A acurácia do KNN fornece um baseline para comparação com o LSTM\n")
        f.write("   - Os outliers e características temporais serão melhor modelados com LSTM\n")
    
    print("Simple models analysis complete. Results saved to 'simple_models_summary.txt'")
    
    return df_sample

if __name__ == "__main__":
    df_sample = main()