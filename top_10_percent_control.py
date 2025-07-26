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
import matplotlib.pyplot as plt
import seaborn as sns


# Parameters to iterate over
N_list = [50, 100, 150]
k_list = [2.4, 3.4]
t_list = [2, 3]
mu_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
number_of_models = 20
steps = 60
no_of_phenotypes = 2


def process_network(params, method='leiden'):
    import matplotlib.pyplot as plt
    import seaborn as sns

    N, k, maxk, t1 = params
    path_to_files = f'N {N} -k {k} -maxk {maxk} -t1 {t1}/'
    out_name = 'la_percent_coverage_control_size_1_w.txt' if method == 'leiden' else 'lp_percent_coverage_control_size_1_w.txt'
    f_out = open(path_to_files + out_name, 'w')

    all_betweenness = []
    all_cycle = []
    all_bc_ranks = []
    all_cycle_ranks = []

    for mu in mu_list:
        print(f'Processing N={N}, k={k}, t={t1}, mu={mu} with {method}')
        G = nx.read_graphml(path_to_files + f'mu {mu}/network.graphml')
        G = CLO.leiden_partitions(G) if method == 'leiden' else CLO.label_propagation(G, steps=steps, write_graphml=False)

        control_in_communities, bet_scores, cycle_scores, bc_ranks, cycle_ranks = CLO.influential_nodes_in_communities_rank_based(G)
        all_betweenness.extend(bet_scores)
        all_cycle.extend(cycle_scores)
        all_bc_ranks.extend(bc_ranks)
        all_cycle_ranks.extend(cycle_ranks)

        control = []
        for item_c_key in control_in_communities.keys():
            control += control_in_communities[item_c_key]

        all_per = []
        for i in range(number_of_models):
            path_to_model = path_to_files + f'mu {mu}/Boolean models/{i}.booleannet'
            try:
                primes = sm.format.import_primes(path_to_model)
                min_trap_spaces = compute_trap_spaces(primes, 'min')
                clustered_mints = CLO.hierarchial_cluster_mints(min_trap_spaces, list(G.nodes()), no_of_phenotypes)

                for group in clustered_mints:
                    common_part = (
                        CLO.common_part(min_trap_spaces, group)
                        if len(group) != 1 else min_trap_spaces[int(group[0])]
                    )
                    mutual_with_control = CLO.mutual_items(common_part.keys(), control)
                    dict_control = {c_n: common_part[c_n] for c_n in mutual_with_control}

                    if dict_control:
                        LDOI = sm.drivers.logical_domain_of_influence(dict_control, primes)
                        number_coverage = CLO.number_covered_in_phenotype(
                            {**LDOI[0], **dict_control}, common_part
                        )
                        percentage = number_coverage / len(common_part)
                        all_per.append(percentage)

            except Exception as e:
                print(f"Error processing model {i}: {str(e)}")
                continue

        avg = np.mean(all_per) if all_per else 0
        std = np.std(all_per) if all_per else 0
        count = len(all_per)
        f_out.write(f"{mu} {avg} {std} {count} {len(control)}\n")

    f_out.close()
    return all_betweenness, all_cycle, all_bc_ranks, all_cycle_ranks




all_betweenness_leiden = []
all_cycle_leiden = []
all_bc_rank_leiden = []
all_cycle_rank_leiden = []

all_betweenness_label = []
all_cycle_label = []
all_bc_rank_label = []
all_cycle_rank_label = []

for N in N_list:
    maxk = 6
    for k in k_list:
        for t1 in t_list:
            params = (N, k, maxk, t1)
            path_to_files = f'N {N} -k {k} -maxk {maxk} -t1 {t1}/'
            if not os.path.exists(path_to_files):
                os.makedirs(path_to_files)

            b_leiden, c_leiden, r_bc_leiden, r_cyc_leiden = process_network(params, method='leiden')
            b_label, c_label, r_bc_label, r_cyc_label = process_network(params, method='label_propagation')

            all_betweenness_leiden.extend(b_leiden)
            all_cycle_leiden.extend(c_leiden)
            all_bc_rank_leiden.extend(r_bc_leiden)
            all_cycle_rank_leiden.extend(r_cyc_leiden)

            all_betweenness_label.extend(b_label)
            all_cycle_label.extend(c_label)
            all_bc_rank_label.extend(r_bc_label)
            all_cycle_rank_label.extend(r_cyc_label)

def print_summary_stats(name, scores):
    avg = np.mean(scores) if scores else 0
    std = np.std(scores) if scores else 0
    print(f"{name} — Avg: {avg:.4f}, Std: {std:.4f}")

print("\n--- Summary Statistics ---")
print_summary_stats("Leiden Betweenness", all_betweenness_leiden)
print_summary_stats("Label Propagation Betweenness", all_betweenness_label)
print_summary_stats("Leiden Cycle Scores", all_cycle_leiden)
print_summary_stats("Label Propagation Cycle Scores", all_cycle_label)
print_summary_stats("Leiden Betweenness Ranks", all_bc_rank_leiden)
print_summary_stats("Label Propagation Betweenness Ranks", all_bc_rank_label)
print_summary_stats("Leiden Cycle Ranks", all_cycle_rank_leiden)
print_summary_stats("Label Propagation Cycle Ranks", all_cycle_rank_label)
