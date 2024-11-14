from src.spectral_fns import run_cluster
# from src.gmm import run_cluster_gmm
import pandas as pd 
import itertools
import os 

output_path = '/Users/gracecolverd/postcode_clustering/results' 
input_path = "/Users/gracecolverd/City_clustering/notebooks/clean_v1_round2_secondfilter.csv" 

# list_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
list_clusters = [12, 13, 14, 15, 16, 17, 18, 19, 20]
list_clusters = [7,8,9,10,11,12,13]

list_clusters = [17,18,19, 20,21,22,23,24,25,26]
nrs = 1
subset=None
# subset=100000
output_path = os.path.join(output_path, f'subset_{subset}')

run_gmm= True 
run_km = False 
run_hc = False 


# Parameters for different algorithms
kmeans_params = {'algorithm': 'kmeans', 'n_init': [100], 'norm': False, 'subset': subset}
spectral_params = {'algorithm': 'spectral', 'affinity': 'rbf', 'n_init': [100], 'norm': True, 'subset': subset}
hierarchical_params = {'algorithm': 'hierarchical', 'linkage': ['single'], 'norm': False, 'subset': subset}
gmm_params = {'algorithm': 'gmm', 'n_init': [100], 'norm': True, 'subset': subset}



# Function to run clustering for a specific algorithm
def run_clustering(algo_params, num_clusters):
    base_params = algo_params.copy()
    base_params['num_clusters'] = num_clusters
    base_params['nrs'] = nrs

    if algo_params['algorithm'] == 'hierarchical':
        for linkage in base_params['linkage']:
                params = base_params.copy()
                params['linkage'] = linkage
                run_cluster(output_path, input_path, **params)
    elif algo_params['algorithm'] == 'gmm':
        for n_init in base_params['n_init']:
            params = base_params.copy()
            params['n_init'] = n_init
            run_cluster(output_path, input_path, **params)
    else:
        for n_init in base_params['n_init']:
            params = base_params.copy()
            params['n_init'] = n_init
            run_cluster(output_path, input_path, **params)

# Run clustering for each algorithm and number of clusters
for num_clusters in list_clusters:
    print(f'starting clusters: {num_clusters}')
    if run_gmm:
        print('GMM')
        run_clustering(gmm_params, num_clusters)
    if run_hc:
        print('HC')
        run_clustering(hierarchical_params, num_clusters)
    if run_km:
        print('KM')
        run_clustering(kmeans_params, num_clusters)
    # run_clustering(spectral_params, num_clusters)
    

print("Clustering analysis complete.")

