
from src.postcode_cluster import load_and_prepare_postcode_data

def run_cluster(output_path, input_path, num_clusters, nrs, algorithm='kmeans', affinity='rbf', 
                linkage='ward', metric='euclidean', norm=True, n_init=10):
    # Set random seed
    np.random.seed(nrs)

    # Create output path
    output_path = f"{output_path}/{algorithm}/{num_clusters}/{affinity if algorithm == 'spectral' else linkage}/{n_init}/random_seed_{nrs}"    
    os.makedirs(output_path, exist_ok=True) 

    # Get dataset name
    dataset_name = os.path.basename(input_path).split('_')[0]
    run_name = dataset_name
    
    try:
        # Load and prepare data
        X_train, data_cols = load_and_prepare_postcode_data(input_path)

        # Run clustering
        model, scaler, pca, labels, X_principal, silhouette_avg, davies_bouldin, calinski_harabasz = train_clustering_model(
            X_train, num_clusters, algorithm=algorithm, affinity=affinity, linkage=linkage, 
            metric=metric, norm=norm,   random_state=nrs, n_init=n_init
        )

        # Save results
        save_results_full(output_path, run_name, model, scaler, pca, labels, X_principal, 
                          silhouette_avg, davies_bouldin, calinski_harabasz)

        # Plot variable distributions
        plot_variable_distributions(X_train, labels, os.path.join(output_path, run_name), data_cols)

    except Exception as e:
        print(f"An error occurred: {str(e)}")