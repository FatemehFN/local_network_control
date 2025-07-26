import pickle
import networkx as nx
import pystablemotifs as sm
from pyboolnet.trap_spaces import compute_trap_spaces
import classification_operations as CLO  # Assuming CLO is the cleaned-up code from the previous turn
import numpy as np
import os

# --- Parameters ---
N_list = [50, 100, 150]  # Number of nodes in the network
k_list = [2.4, 3.4]      # Average degree
t_list = [2, 3]          # Number of targets (t1 in the original code)
mu_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] # Parameter 'mu'
number_of_models = 20    # Number of Boolean models to process for each network
no_of_phenotypes = 2     # Number of phenotypes for hierarchical clustering


# --- Functions ---
def process_network(N, k, maxk, t1):
    """
    Processes a set of Boolean network models generated with specific parameters.
    For each network, it performs community detection on minimal trapping sets,
    identifies influential nodes using betweenness centrality, and calculates
    the percentage coverage of identified phenotypes by the logical domain of influence
    of these influential nodes.

    Args:
        N (int): Number of nodes in the network.
        k (float): Average degree of the network.
        maxk (int): Maximum degree allowed for nodes.
        t1 (int): Number of targets (t in the original problem context).
    """
    path_to_files = f'N {N} -k {k} -maxk {maxk} -t1 {t1}/'

    # Ensure the directory for storing results exists
    os.makedirs(path_to_files, exist_ok=True)

    output_filename = os.path.join(path_to_files, 'GBC_mu_percent_coverage.txt')

    with open(output_filename, 'w') as f_out:
        for mu in mu_list:
            print(f'Processing N={N}, k={k}, t1={t1}, mu={mu} with GBC control')

            network_path = os.path.join(path_to_files, f'mu {mu}', 'network.graphml')
            if not os.path.exists(network_path):
                print(f"Network file not found: {network_path}. Skipping.")
                continue

            # Load the network graph
            G = nx.read_graphml(network_path)

            # Calculate Betweenness Centrality for all nodes in the network
            dict_betweenness_centrality = nx.betweenness_centrality(G)

            # Select the top 10% nodes based on their betweenness centrality
            top_10_percent_count = max(1, int(len(G.nodes()) * 0.10))
            control_nodes = sorted(dict_betweenness_centrality,
                                   key=dict_betweenness_centrality.get,
                                   reverse=True)[:top_10_percent_count]

            print('Betweenness centrality values:', dict_betweenness_centrality)
            print('Control nodes (top 10% by betweenness):', control_nodes)

            all_percentages = [] # Stores percentage coverage for each model
            for i in range(number_of_models):
                print(f'Processing model {i} for mu={mu}')

                try:
                    # Load Boolean primes (rules) for the current model
                    primes_path = os.path.join(path_to_files, f'mu {mu}', 'Boolean models', f'{i}.booleannet')
                    if not os.path.exists(primes_path):
                        print(f"Boolean model file not found: {primes_path}. Skipping.")
                        continue

                    primes = sm.format.import_primes(primes_path)
                    
                    # Compute minimal trapping spaces (stable motifs)
                    min_trap_spaces = compute_trap_spaces(primes, 'min')

                    if not min_trap_spaces:
                        print(f"No minimal trapping spaces found for model {i}. Skipping.")
                        continue

                    # Hierarchically cluster the minimal trapping spaces into phenotypes
                    clustered_mints = CLO.hierarchial_cluster_mints(
                        mints=min_trap_spaces,
                        list_of_nodes=list(G.nodes()), # Pass as a list
                        no_of_clusters=no_of_phenotypes
                    )

                    for group_indices in clustered_mints:
                        # Determine the common part (fixed states) for the current cluster/phenotype
                        if len(group_indices) > 0: # Ensure group is not empty
                            if len(group_indices) == 1:
                                common_part = min_trap_spaces[int(group_indices[0])]
                            else:
                                common_part = CLO.common_part(min_trap_spaces, group_indices)
                        else:
                            common_part = {} # Empty group, empty common part

                        print('Common part (phenotype fixed states):', common_part)

                        # Identify control nodes that are part of the common_part
                        common_part_keys = common_part.keys()
                        mutual_with_control = CLO.mutual_items(list(common_part_keys), control_nodes)
                        
                        # Create a dictionary of the fixed states for the control nodes
                        dict_control = {c_n: common_part[c_n] for c_n in mutual_with_control}

                        if dict_control:
                            # Calculate the Logical Domain of Influence (LDOI) for the control nodes
                            # The LDOI represents nodes whose states are determined by the fixed control nodes
                            LDOI = sm.drivers.logical_domain_of_influence(dict_control, primes)
                            
                            # Combine LDOI fixed states with the control node fixed states
                            resultant_fixed_states = {**LDOI[0], **dict_control}
                            print('Constants (LDOI + control):', resultant_fixed_states)

                            # Calculate how many of the phenotype's fixed states are covered by the LDOI
                            number_coverage = CLO.number_covered_in_phenotype(
                                resultant_fixed_states,
                                common_part
                            )
                            
                            # Calculate percentage coverage
                            percentage = number_coverage / len(common_part) if common_part else 0
                            print('Percentage coverage:', percentage)
                            all_percentages.append(percentage)
                        else:
                            print("No control nodes intersect with common part. Skipping LDOI calculation.")
                            # Consider how to handle this case in results (e.g., append 0 or skip)
                            # For now, we don't append anything, which means it won't contribute to avg/std.

                except Exception as e:
                    print(f"Error processing model {i} for mu={mu}: {e}")
                    # Continue to the next model even if one fails

            # Calculate and write summary statistics (average, standard deviation, count)
            if all_percentages:
                avg_coverage = np.mean(all_percentages)
                std_coverage = np.std(all_percentages)
                successful_models_count = len(all_percentages)
            else:
                avg_coverage = 0.0
                std_coverage = 0.0
                successful_models_count = 0

            f_out.write(
                f"{mu} {avg_coverage:.4f} {std_coverage:.4f} {successful_models_count} {len(control_nodes)}\n"
            )


# --- Main Execution ---
if __name__ == "__main__":
    maxk = 6  # Fixed maximum degree as per original script

    for N in N_list:
        for k in k_list:
            for t1 in t_list:
                process_network(N, k, maxk, t1)

    print("All network processing complete.")