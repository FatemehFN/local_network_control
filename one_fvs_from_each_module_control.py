import pickle
import networkx as nx
import copy
import pystablemotifs as sm
from pyboolnet.trap_spaces import compute_trap_spaces
import BooleanDOI_processing as BDOIp
import operator
import classification_operations as CLO
import FVS
from iteration_utilities import deepflatten
import numpy as np
import os

# Parameters to iterate over
N_list = [50, 100, 150]
k_list = [2.4, 3.4]
t_list = [2, 3]
mu_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
number_of_models = 20
steps = 60
no_of_phenotypes = 2


def process_network(params, method='leiden'):
    N, k, maxk, t1 = params
    path_to_files = f'N {N} -k {k} -maxk {maxk} -t1 {t1}/'

    # Create output files
    if method == 'leiden':
        f_out = open(path_to_files + 'la_one_fvs_percent_coverage_control_size_1_w.txt', 'w')
    else:
        f_out = open(path_to_files + 'lp_one_fvs_percent_coverage_control_size_1_w.txt', 'w')

    for mu in mu_list:
        print(f'Processing N={N}, k={k}, t={t1}, mu={mu} with {method}')

        # Load network and get FVS
        G = nx.read_graphml(path_to_files + f'mu {mu}/network.graphml')
        FVS_nodes = FVS.FVS(G)
        FVS_size = len(FVS_nodes)

        # Apply community detection
        if method == 'leiden':
            G = CLO.leiden_partitions(G)
        else:
            G = CLO.label_propagation(G, steps=steps, write_graphml=False)

        # Get control nodes using one FVS member per community
        control_in_communities = CLO.influential_nodes_in_communities_one_fvs_member(G, FVS_nodes)
        control = list(control_in_communities.values())

        # print('structural control')
        # print(control)

        all_per = []
        for i in range(number_of_models):
            print(i)

            path_to_model = path_to_files + f'mu {mu}/Boolean models/{i}.booleannet'

            try:
                primes = sm.format.import_primes(path_to_model)
                min_trap_spaces = compute_trap_spaces(primes, 'min')

                clustered_mints = CLO.hierarchial_cluster_mints(
                    mints=min_trap_spaces,
                    list_of_nodes=G.nodes(),
                    no_of_clusters=no_of_phenotypes
                )

                for group in clustered_mints:
                    if len(group) != 1:
                        common_part = CLO.common_part(min_trap_spaces, group)

                    else:
                        common_part = min_trap_spaces[int(group[0])]


                    common_part_keys = common_part.keys()
                    mutual_with_control = CLO.mutual_items(common_part_keys, control)
                    dict_control = {}
                    for c_n in mutual_with_control:
                        dict_control.update({c_n: common_part[c_n]})

                    if dict_control != {}:
                        LDOI = sm.drivers.logical_domain_of_influence(dict_control, primes)
                        number_coverage = CLO.number_covered_in_phenotype(
                            {**LDOI[0], **dict_control},
                            common_part
                        )
                        percentage = number_coverage / len(common_part)
                        all_per.append(percentage)

            except Exception as e:
                print(f"Error processing model {i}: {str(e)}")
                continue

        # Calculate statistics
        if all_per:
            avg = np.mean(all_per)
            std = np.std(all_per)
            count = len(all_per)
        else:
            avg = std = 0
            count = 0

        # Write results
        f_out.write(
            f"{mu} {avg} {std} {count} {len(control)}\n"
        )

    f_out.close()


# Main execution
for N in N_list:
    maxk = 6  # Kept constant as in original
    for k in k_list:
        for t1 in t_list:
            params = (N, k, maxk, t1)

            # Create directory if it doesn't exist
            path_to_files = f'N {N} -k {k} -maxk {maxk} -t1 {t1}/'
            if not os.path.exists(path_to_files):
                os.makedirs(path_to_files)

            # Process with both methods
            process_network(params, method='leiden')
            process_network(params, method='label_propagation')