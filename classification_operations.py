import networkx as nx
from scipy.cluster import hierarchy
import numpy as np
from iteration_utilities import deepflatten
from matplotlib import pyplot as plt
from pyboolnet.file_exchange import bnet2primes
from pyboolnet.prime_implicants import percolate
import operator
from sklearn.cluster import AffinityPropagation
import copy
import itertools
import pystablemotifs as sm
import igraph as ig
import leidenalg
from statistics import mean
import re
from deepdiff import DeepDiff


def form_matrix_from_mints(mints, list_of_nodes):
    """
    Forms a binary matrix where rows represent nodes and columns represent "mints" (likely minimal trapping sets).
    A cell (i, j) is 1 if node i is present and active (value 1) in mint j, 0 if present and inactive (value 0),
    and 0.5 if the node is not present in the mint.

    Args:
        mints (list of dict): A list of dictionaries, where each dictionary represents a mint
                               and maps node names to their states (0 or 1).
        list_of_nodes (list): A list of all node names in the network.

    Returns:
        np.array: A transposed numpy array representing the formed matrix.
    """
    matrix = np.zeros((len(list_of_nodes), len(mints)))
    for i, node in enumerate(list_of_nodes):
        for j, mint in enumerate(mints):
            if node in mint:
                matrix[i][j] = 1 if mint[node] == 1 else 0
            else:
                matrix[i][j] = 0.5
    return matrix.transpose()


def in_list(cluster, clusters_to_be_returned):
    """
    Checks if a given cluster (or a subset of it) is already present in a list of clusters.

    Args:
        cluster (list): The cluster to check.
        clusters_to_be_returned (list of lists): A list of existing clusters.

    Returns:
        bool: True if the cluster (or a subset) is found, False otherwise.
    """
    for item in clusters_to_be_returned:
        if set(item).issubset(set(cluster)):
            return True
    return False


def in_list_reverse(list1, list_of_lists):
    """
    Checks if a given list (list1) is a subset of any list within a list of lists.

    Args:
        list1 (list): The list to check.
        list_of_lists (list of lists): The list of lists to search within.

    Returns:
        bool: True if list1 is a subset of any list in list_of_lists, False otherwise.
    """
    for item in list_of_lists:
        if set(list1).issubset(set(item)):
            return True
    return False


def hierarchial_cluster_mints(mints, list_of_nodes, no_of_clusters=2, rekas_proposal=True, folder_path='', name=''):
    """
    Performs hierarchical clustering on "mints" based on node presence and state.

    Args:
        mints (list of dict): A list of dictionaries, where each dictionary represents a mint
                               and maps node names to their states (0 or 1).
        list_of_nodes (list): A list of all node names in the network.
        no_of_clusters (int, optional): The desired number of clusters. Defaults to 2.
        rekas_proposal (bool, optional): If True, generates dendrogram and saves matrix (commented out).
                                         Defaults to True.
        folder_path (str, optional): Path to the folder for saving files (if rekas_proposal is True). Defaults to ''.
        name (str, optional): Name for saved files (if rekas_proposal is True). Defaults to ''.

    Returns:
        list of lists: A list of clusters, where each cluster is a list of mint indices.
    """
    matrix = form_matrix_from_mints(mints, list_of_nodes)

    if rekas_proposal:
        # fig, ax = plt.subplots(1, 1, figsize=(14, 10)) # Uncomment to show dendrogram
        Z = hierarchy.linkage(matrix, 'single', metric='euclidean')
        # hierarchy.dendrogram(Z, ax=ax, orientation='top') # Uncomment to show dendrogram
        # ax.tick_params(direction='in') # Uncomment to show dendrogram

        # text_path = folder_path + 'matrices/matrix' + name + '.txt'
        # np.savetxt(text_path, matrix, fmt="%d", delimiter="\t") # Uncomment to save matrix

        Z = hierarchy.linkage(matrix, 'single', metric='euclidean')
        # fig.savefig(folder_path + 'dendrograms/dendrogram' + name + '.pdf', bbox_inches='tight') # Uncomment to save dendrogram

    clusters = []
    n = len(Z) + 1
    len_Z_fix = len(Z) + 1  # This variable name seems slightly misleading, as it's the number of original items + 1
    for item in Z:
        # Determine the first entry of the new cluster
        if item[0] >= len_Z_fix:
            entry_1 = list(deepflatten([c[0] for c in clusters if c[2] == item[0]][0] + [c[1] for c in clusters if c[2] == item[0]][0]))
        else:
            entry_1 = item[0]

        # Determine the second entry of the new cluster
        if item[1] >= len_Z_fix:
            entry_2 = list(deepflatten([c1[0] for c1 in clusters if c1[2] == item[1]][0] + [c1[1] for c1 in clusters if c1[2] == item[1]][0]))
        else:
            entry_2 = item[1]

        # Handle cases where both entries are original items
        if item[0] < len_Z_fix and item[1] < len_Z_fix:
            entry_1 = item[0]
            entry_2 = item[1]

        clusters.append([entry_1, entry_2, n, item[2]])
        n += 1

    clusters_to_be_returned = []
    for k in range(no_of_clusters - 1, -1, -1):  # Iterate from largest clusters downwards
        # Ensure entries are lists for consistent processing
        current_cluster_elements = clusters[-k-1][0] if isinstance(clusters[-k-1][0], list) else [clusters[-k-1][0]]
        
        if not in_list(current_cluster_elements, clusters_to_be_returned):
            clusters_to_be_returned.append(current_cluster_elements)
        
        if len(clusters_to_be_returned) == no_of_clusters:
            break
        
        current_cluster_elements = clusters[-k-1][1] if isinstance(clusters[-k-1][1], list) else [clusters[-k-1][1]]
        if not in_list(current_cluster_elements, clusters_to_be_returned):
            clusters_to_be_returned.append(current_cluster_elements)
            
        if len(clusters_to_be_returned) == no_of_clusters:
            break

    return clusters_to_be_returned


def weighted_cycle_score(G):
    """
    Calculates a weighted cycle score for each node in a directed graph.
    The score is based on the number of sign-consistent cycles a node is part of,
    weighted inversely by the square of the cycle length.

    Args:
        G (nx.DiGraph): A NetworkX directed graph where edges have a 'negative' attribute (True/False).

    Returns:
        dict: A dictionary of normalized cycle scores for each node, sorted by score.
    """
    cycles = list(nx.simple_cycles(G))
    
    sign_consistent_cycles = []
    for c in cycles:
        count_neg = 0
        for i in range(len(c)):
            u, v = c[i], c[(i + 1) % len(c)]
            if G.has_edge(u, v) and G[u][v].get('negative', False):
                count_neg += 1
        if count_neg % 2 == 0:
            sign_consistent_cycles.append(c)

    total_number_of_cycles = len(sign_consistent_cycles)
    cycle_score = {}

    for node in G.nodes():
        weight = 0
        for c in sign_consistent_cycles:
            if node in c:
                weight += 1 / (pow(len(c), 2))  # Inverse squared weight function
        cycle_score[node] = weight

    max_score = max(cycle_score.values()) if cycle_score else 0
    if max_score != 0:
        normalized_scores = {k: v / max_score for k, v in cycle_score.items()}
        return dict(sorted(normalized_scores.items(), key=lambda item: item[1]))
    else:
        return dict(sorted(cycle_score.items(), key=lambda item: item[1]))


def CheiRank(G):
    """
    Calculates the CheiRank scores for nodes in a directed graph, which is
    equivalent to PageRank on the reversed graph.

    Args:
        G (nx.DiGraph): A NetworkX directed graph.

    Returns:
        dict: A dictionary of CheiRank scores for each node.
    """
    rev_net = nx.reverse(G)
    return nx.pagerank(rev_net)


def sm_score_of_communities(G, max_trap_spaces):
    """
    Calculates a score for each community based on the proportion of maximal trapping
    sets (max_trap_spaces) that are fully contained within that community.

    Args:
        G (nx.DiGraph): A NetworkX directed graph with nodes having a 'label' attribute.
        max_trap_spaces (list of dict): A list of dictionaries, where each dictionary
                                         represents a maximal trapping space.

    Returns:
        dict: A dictionary where keys are community labels and values are their SM scores.
    """
    labels = list(set([x[-1]['label'] for x in G.nodes(data=True)]))
    communities = {l: [x[0] for x in G.nodes(data=True) if x[-1]['label'] == l] for l in labels}

    total_number_maxts = len(max_trap_spaces)
    sm_score_of_communities = {}
    for com_key, nodes_in_community in communities.items():
        count = 0
        set_nodes_in_this_community = set(nodes_in_community)
        for maxts in max_trap_spaces:
            if set(maxts.keys()).issubset(set_nodes_in_this_community):
                count += 1
        sm_score_of_communities[com_key] = count / total_number_maxts if total_number_maxts > 0 else 0
    return sm_score_of_communities


def average_structure_score_in_communities(G):
    """
    Calculates the average structural score for each community in the graph.
    The structural score for a node is the sum of its betweenness centrality
    and its weighted cycle score within its community.

    Args:
        G (nx.DiGraph): A NetworkX directed graph with nodes having a 'label' attribute.

    Returns:
        dict: A dictionary where keys are community labels and values are their
              average structural scores.
    """
    labels = list(set([x[-1]['label'] for x in G.nodes(data=True)]))
    communities = {l: [x[0] for x in G.nodes(data=True) if x[-1]['label'] == l] for l in labels}

    communities_with_average_structure_scores = {}
    for key, nodes_in_community in communities.items():
        community_subgraph = G.subgraph(nodes_in_community)
        dict_betweenness_centrality_this_community = nx.betweenness_centrality(community_subgraph)
        cycle_score = weighted_cycle_score(community_subgraph)

        total_score_within_this_community = {
            node: dict_betweenness_centrality_this_community.get(node, 0) + cycle_score.get(node, 0)
            for node in nodes_in_community
        }
        communities_with_average_structure_scores[key] = mean(total_score_within_this_community.values()) if total_score_within_this_community else 0

    return communities_with_average_structure_scores


def influential_nodes_in_communities(G):
    """
    Identifies the top 10% most influential nodes within each community based on
    a combined score of betweenness centrality and weighted cycle score.

    Args:
        G (nx.DiGraph): A NetworkX directed graph with nodes having a 'label' attribute.

    Returns:
        tuple: A tuple containing:
               - dict: A dictionary where keys are community labels and values are
                       lists of influential node names within that community.
               - list: A list of all betweenness centrality scores calculated across all nodes.
               - list: A list of all weighted cycle scores calculated across all nodes.
    """
    labels = list(set([x[-1]['label'] for x in G.nodes(data=True)]))
    communities = {l: [x[0] for x in G.nodes(data=True) if x[-1]['label'] == l] for l in labels}

    control_in_communities = {}
    all_betweenness_scores = []
    all_cycle_scores = []

    for key, community_nodes in communities.items():
        total_score_within_this_community = {}
        community_subgraph = G.subgraph(community_nodes)
        dict_betweenness = nx.betweenness_centrality(community_subgraph)
        dict_cycle = weighted_cycle_score(community_subgraph)

        for node in community_nodes:
            b_score = dict_betweenness.get(node, 0)
            c_score = dict_cycle.get(node, 0)
            total_score = b_score + c_score
            total_score_within_this_community[node] = total_score

            all_betweenness_scores.append(b_score)
            all_cycle_scores.append(c_score)

        top_k = max(1, int(len(community_nodes) * 0.1))
        max_nodes = sorted(total_score_within_this_community, key=total_score_within_this_community.get, reverse=True)[:top_k]
        control_in_communities[key] = max_nodes

    return control_in_communities, all_betweenness_scores, all_cycle_scores


def influential_nodes_in_communities_rank_based(G):
    """
    Identifies influential nodes within each community based on a combined rank
    of betweenness centrality and weighted cycle score.

    Args:
        G (nx.DiGraph): A NetworkX directed graph with nodes having a 'label' attribute.

    Returns:
        tuple: A tuple containing:
               - dict: A dictionary where keys are community labels and values are
                       lists of influential node names within that community.
               - list: A list of all betweenness centrality scores.
               - list: A list of all weighted cycle scores.
               - list: A list of all betweenness centrality ranks.
               - list: A list of all weighted cycle ranks.
    """
    labels = list(set([x[-1]['label'] for x in G.nodes(data=True)]))
    communities = {l: [x[0] for x in G.nodes(data=True) if x[-1]['label'] == l] for l in labels}

    control_in_communities = {}
    all_betweenness_scores = []
    all_cycle_scores = []
    all_bc_ranks = []
    all_cycle_ranks = []

    for key, community_nodes in communities.items():
        community_subgraph = G.subgraph(community_nodes)

        dict_betweenness = nx.betweenness_centrality(community_subgraph)
        dict_cycle = weighted_cycle_score(community_subgraph)

        # Ranking
        bc_sorted = sorted(dict_betweenness.items(), key=lambda x: x[1], reverse=True)
        bc_ranks = {node: len(community_nodes) - i for i, (node, _) in enumerate(bc_sorted)}

        cycle_sorted = sorted(dict_cycle.items(), key=lambda x: x[1])
        cycle_ranks = {node: i + 1 for i, (node, _) in enumerate(cycle_sorted)}

        total_rank = {}
        for node in community_nodes:
            total = bc_ranks.get(node, 0) + cycle_ranks.get(node, 0)
            total_rank[node] = total
            all_betweenness_scores.append(dict_betweenness.get(node, 0))
            all_cycle_scores.append(dict_cycle.get(node, 0))
            all_bc_ranks.append(bc_ranks.get(node, 0))
            all_cycle_ranks.append(cycle_ranks.get(node, 0))

        top_k = max(1, int(len(community_nodes) * 0.1))
        top_nodes = sorted(total_rank, key=total_rank.get, reverse=True)[:top_k]
        control_in_communities[key] = top_nodes

    return control_in_communities, all_betweenness_scores, all_cycle_scores, all_bc_ranks, all_cycle_ranks


def common_part(min_trap_spaces, group):
    """
    Identifies the common fixed states (keys and values) across a group of minimal trapping sets.

    Args:
        min_trap_spaces (list of dict): A list of dictionaries, where each dictionary
                                         represents a minimal trapping set.
        group (list): A list of indices indicating which minimal trapping sets from
                      min_trap_spaces belong to the current cluster.

    Returns:
        dict: A dictionary representing the common fixed states among the specified mints.
    """
    mints_in_the_cluster = [min_trap_spaces[int(index)] for index in group]
    nodes_in_each_mint = [set(mint.keys()) for mint in mints_in_the_cluster]

    if not nodes_in_each_mint:
        return {}

    mutual_keys = set.intersection(*nodes_in_each_mint)

    common_part_dict = {}
    for key in mutual_keys:
        reference_value = mints_in_the_cluster[0][key]
        if all(mint[key] == reference_value for mint in mints_in_the_cluster[1:]):
            common_part_dict[key] = reference_value
    return common_part_dict


def addNegativeEdges(G, pNeg=0.25, seed=0):
    """
    Adds a 'negative' attribute (True/False) to each edge in the graph randomly.

    Args:
        G (nx.DiGraph): The NetworkX directed graph.
        pNeg (float, optional): The probability of an edge being labeled as negative. Defaults to 0.25.
        seed (int, optional): Seed for the random number generator for reproducibility. Defaults to 0.
    """
    rng = np.random.default_rng(seed)
    for u, v, data in G.edges(data=True):
        isNeg = rng.random()
        data['negative'] = (isNeg < pNeg)


def Boolean_control_size(fixed_variables, primes):
    """
    Calculates the average size of minimal drivers needed to control the Boolean network
    to reach a state consistent with the given `fixed_variables`.

    Args:
        fixed_variables (dict): A dictionary representing the desired fixed states of some nodes.
        primes (dict): The Boolean rules of the network in pyboolnet prime implicant format.

    Returns:
        float: The average size of minimal drivers. Returns 0 if no drivers are found.
    """
    drivers = sm.drivers.minimal_drivers(fixed_variables, primes)
    
    if not drivers:
        return 0
    
    len_drivers = [len(d) for d in drivers]
    return sum(len_drivers) / len(len_drivers)


def mutual_items(list_1, list_2):
    """
    Finds common items between two lists.

    Args:
        list_1 (list): The first list.
        list_2 (list): The second list.

    Returns:
        list: A list of mutual items, without duplicates.
    """
    return list(set(list_1) & set(list_2))


def choose_label(dict_sum_labels, G, current_label):
    """
    Chooses the new label for a node during label propagation.
    The choice is based on the label with the highest weighted sum from neighbors,
    breaking ties by preferring the label that is most frequent in the overall network,
    and then if still tied, the current label.

    Args:
        dict_sum_labels (dict): A dictionary where keys are neighbor labels and values
                                are their summed weights.
        G (nx.DiGraph): The NetworkX directed graph.
        current_label (hashable): The current label of the node.

    Returns:
        hashable: The chosen new label for the node.
    """
    if not dict_sum_labels:
        return current_label  # Self-loop or isolated node with no other edges affecting its label

    max_weight = max(dict_sum_labels.values())
    max_labels = [label for label, weight in dict_sum_labels.items() if weight == max_weight]

    if len(max_labels) == 1:
        return max_labels[0]

    # Tie-breaking: prefer the label that is most frequent in the overall network
    all_labels_in_the_network = [x[-1]['label'] for x in G.nodes(data=True)]
    label_counts = {label: all_labels_in_the_network.count(label) for label in max_labels}

    # If the current node's label is among max_labels, decrement its count for tie-breaking
    # This prevents the current node from biasing its own label's global count unfairly
    if current_label in label_counts:
        label_counts[current_label] -= 1
        
    # Find labels with the maximum count
    if not label_counts: # All labels had their count decremented to 0
        return max_labels[0] # Pick any, say the first one

    max_count_value = max(label_counts.values())
    labels_with_max_count = [label for label, count in label_counts.items() if count == max_count_value]

    # Further tie-breaking: prefer the current label if it's among the maximum count labels
    if current_label in labels_with_max_count:
        return current_label
    
    # If still tied, prefer the alphabetically first label
    return sorted(labels_with_max_count)[0]


def steady(G_array):
    """
    Checks if the label propagation process has reached a steady state by comparing
    the node labels in the last two graphs in G_array.

    Args:
        G_array (list of nx.DiGraph): A list of graphs representing the network
                                      at different steps of label propagation.

    Returns:
        bool: True if the labels of all nodes are the same in the last two graphs, False otherwise.
    """
    if len(G_array) < 2:
        return False

    last_G = G_array[-1]
    second_to_last_G = G_array[-2]

    labels_last = {node[0]: node[-1]['label'] for node in last_G.nodes(data=True)}
    labels_second_to_last = {node[0]: node[-1]['label'] for node in second_to_last_G.nodes(data=True)}

    return labels_last == labels_second_to_last


def positive_edge_connect(G):
    """
    Merges communities based on a majority of positive edges connecting them.
    If the number of positive edges between two communities is significantly
    higher (at least 1 more) than negative edges, those communities are merged.

    Args:
        G (nx.DiGraph): A NetworkX directed graph with nodes having a 'label' attribute
                        and edges having a 'negative' and 'direction weight' attribute.

    Returns:
        nx.DiGraph: The graph with communities potentially merged.
    """
    labels = list(set([x[-1]['label'] for x in G.nodes(data=True)]))
    communities = {l: [x[0] for x in G.nodes(data=True) if x[-1]['label'] == l] for l in labels}

    communities_graph = nx.Graph()
    communities_graph.add_nodes_from(labels)

    positive_flow_merge_labels = []
    community_keys = list(communities.keys())

    for i, key1 in enumerate(community_keys):
        for key2 in community_keys[i + 1:]:  # Avoid self-comparison and duplicate pairs
            nodes1 = communities[key1]
            nodes2 = communities[key2]

            w_positive = 0
            w_negative = 0

            # Iterate over all edges in G to find inter-community edges
            for u, v, data in G.edges(data=True):
                is_inter_community = (u in nodes1 and v in nodes2) or \
                                     (u in nodes2 and v in nodes1)
                
                if is_inter_community:
                    if data.get('negative', False):
                        w_negative += data.get('direction weight', 0)
                    else:
                        w_positive += data.get('direction weight', 0)

            if w_positive - w_negative >= 1:
                positive_flow_merge_labels.append(tuple(sorted((key1, key2)))) # Store as sorted tuple to avoid duplicates

    for l1, l2 in positive_flow_merge_labels:
        if not communities_graph.has_edge(l1, l2):
            communities_graph.add_edge(l1, l2)

    # Merge communities based on connected components in the communities_graph
    cc_communities_graph = list(nx.connected_components(communities_graph))

    for hub_list in cc_communities_graph:
        representative_label = list(hub_list)[0]  # Choose one label as the representative for the merged community
        for node_data in G.nodes(data=True):
            if node_data[-1]['label'] in hub_list:
                node_data[-1]['label'] = representative_label
    return G


def label_propagation(G, steps=30, write_graphml=False, name_of_file='G_LPed.graphml'):
    """
    Performs label propagation on the graph to identify communities,
    considering both positive and negative edge weights.

    Args:
        G (nx.DiGraph): The NetworkX directed graph. Edges should have 'negative' attribute.
                        Nodes are initialized with an implicit label (their own name).
        steps (int, optional): The maximum number of label propagation iterations. Defaults to 30.
        write_graphml (bool, optional): If True, writes the final graph with labels to a GraphML file. Defaults to False.
        name_of_file (str, optional): The name of the GraphML file if write_graphml is True. Defaults to 'G_LPed.graphml'.

    Returns:
        nx.DiGraph: The graph with updated node labels representing communities.
    """
    U = nx.DiGraph(G) # Create a mutable copy

    # Initialize node labels with their own names
    for node_id in U.nodes():
        U.nodes[node_id]['label'] = node_id

    # Calculate edge weights
    for u, v, data in U.edges(data=True):
        if u == v: # Skip self-loops
            data['direction weight'] = 0 
            continue

        if data.get('negative', False) is False:
            all_edges_plus_u = [e for e in U.edges(u, data=True) if e[2].get('negative', False) is False] + \
                               [e for e in U.in_edges(u, data=True) if e[2].get('negative', False) is False]
            all_edges_plus_v = [e for e in U.edges(v, data=True) if e[2].get('negative', False) is False] + \
                               [e for e in U.in_edges(v, data=True) if e[2].get('negative', False) is False]
            k_plus_out_u = len([e for e in U.out_edges(u, data=True) if e[2].get('negative', False) is False])
            k_plus_in_v = len([e for e in U.in_edges(v, data=True) if e[2].get('negative', False) is False])
            
            denominator = (len(all_edges_plus_u) * len(all_edges_plus_v))
            data['direction weight'] = 1 - ((k_plus_out_u * k_plus_in_v) / denominator) if denominator else 0
        else:
            all_edges_neg_u = [e for e in U.edges(u, data=True) if e[2].get('negative', False) is True] + \
                              [e for e in U.in_edges(u, data=True) if e[2].get('negative', False) is True]
            all_edges_neg_v = [e for e in U.edges(v, data=True) if e[2].get('negative', False) is True] + \
                              [e for e in U.in_edges(v, data=True) if e[2].get('negative', False) is True]
            k_neg_out_u = len([e for e in U.out_edges(u, data=True) if e[2].get('negative', False) is True])
            k_neg_in_v = len([e for e in U.in_edges(v, data=True) if e[2].get('negative', False) is True])

            denominator = (len(all_edges_neg_u) * len(all_edges_neg_v))
            data['direction weight'] = 1 - ((k_neg_out_u * k_neg_in_v) / denominator) if denominator else 0


    # Label Propagation
    G_array = []
    for _ in range(steps):
        current_node_labels = {node_id: U.nodes[node_id]['label'] for node_id in U.nodes()}
        
        for node_id in sorted(U.nodes()): # Sort nodes for deterministic updates
            
            incoming_neighbours = [source for source, target in U.in_edges(node_id) if source != node_id]
            outgoing_neighbours = [target for source, target in U.out_edges(node_id) if target != node_id]
            
            all_neighbor_labels = [current_node_labels[n] for n in incoming_neighbours + outgoing_neighbours if n in current_node_labels]
            unique_neighbor_labels = list(set(all_neighbor_labels))

            dict_sum_labels = {}
            for label in unique_neighbor_labels:
                positive_w = 0
                negative_w = 0
                for neighbor_id in incoming_neighbours + outgoing_neighbours:
                    if current_node_labels.get(neighbor_id) == label:
                        if U.has_edge(neighbor_id, node_id):  # Incoming edge
                            edge_data = U[neighbor_id][node_id]
                            if not edge_data.get('negative', False):
                                positive_w += edge_data.get('direction weight', 0)
                            else:
                                negative_w += edge_data.get('direction weight', 0)
                        if U.has_edge(node_id, neighbor_id):  # Outgoing edge
                            edge_data = U[node_id][neighbor_id]
                            if not edge_data.get('negative', False):
                                positive_w += edge_data.get('direction weight', 0)
                            else:
                                negative_w += edge_data.get('direction weight', 0)
                dict_sum_labels[label] = positive_w - negative_w
            
            # Sort for deterministic tie-breaking in choose_label
            sorted_dict_sum_labels = dict(sorted(dict_sum_labels.items()))
            
            chosen_label = choose_label(sorted_dict_sum_labels, U, U.nodes[node_id]['label'])
            U.nodes[node_id]['label'] = chosen_label
        
        # After all nodes are updated, record the state for stability check
        G_array.append(copy.deepcopy(U))
        if len(G_array) >= 2 and steady(G_array):
            break

    # Final community merging based on positive edges
    # U = positive_edge_connect(U) # This can be applied if a further merge based on positive connections is desired after LP

    final_labels = list(set([data['label'] for _, data in U.nodes(data=True)]))
    print(f'Label propagation done with {len(final_labels)} communities')

    if write_graphml:
        H = copy.deepcopy(U)
        for node_h_id, node_h_data in H.nodes(data=True):
            node_h_data['label'] = f"{node_h_id}->{node_h_data['label']}"
        nx.write_graphml(H, name_of_file)

    return U


def number_covered_in_phenotype(resultant_constants, common_part):
    """
    Counts how many fixed states in `common_part` are consistent with `resultant_constants`.

    Args:
        resultant_constants (dict): A dictionary of fixed node states (e.g., a phenotype).
        common_part (dict): A dictionary of common fixed states within a cluster of mints.

    Returns:
        int: The number of consistent fixed states.
    """
    count = 0
    for key, value in common_part.items():
        if key in resultant_constants and resultant_constants[key] == value:
            count += 1
    return count


def number_deviated_from_phenotype(resultant_constants, common_part):
    """
    Counts how many fixed states in `common_part` deviate from `resultant_constants`.

    Args:
        resultant_constants (dict): A dictionary of fixed node states (e.g., a phenotype).
        common_part (dict): A dictionary of common fixed states within a cluster of mints.

    Returns:
        int: The number of deviating fixed states.
    """
    count = 0
    for key, value in common_part.items():
        if key in resultant_constants and resultant_constants[key] != value:
            count += 1
    return count


def label_propagation_unsigned(G, steps=30, write_graphml=False, name_of_file='G_LPed.graphml'):
    """
    Performs label propagation on an unsigned graph, considering only the presence of edges
    and not their sign (positive/negative).

    Args:
        G (nx.DiGraph): The NetworkX directed graph. Edges are not expected to have 'negative' attributes.
                        Nodes are initialized with an implicit label (their own name).
        steps (int, optional): The maximum number of label propagation iterations. Defaults to 30.
        write_graphml (bool, optional): If True, writes the final graph with labels to a GraphML file. Defaults to False.
        name_of_file (str, optional): The name of the GraphML file if write_graphml is True. Defaults to 'G_LPed.graphml'.

    Returns:
        nx.DiGraph: The graph with updated node labels representing communities.
    """
    U = nx.DiGraph(G) # Create a mutable copy

    # Initialize node labels with their own names
    for node_id in U.nodes():
        U.nodes[node_id]['label'] = node_id

    # Calculate edge weights (unsigned)
    for u, v, data in U.edges(data=True):
        if u == v: # Skip self-loops
            data['direction weight'] = 0
            continue

        all_edges_u = [e for e in U.edges(u)] + [e for e in U.in_edges(u)]
        all_edges_v = [e for e in U.edges(v)] + [e for e in U.in_edges(v)]
        k_out_u = U.out_degree(u)
        k_in_v = U.in_degree(v)

        denominator = (len(all_edges_u) * len(all_edges_v))
        data['direction weight'] = 1 - ((k_out_u * k_in_v) / denominator) if denominator else 0


    # Label Propagation
    G_array = []
    for _ in range(steps):
        current_node_labels = {node_id: U.nodes[node_id]['label'] for node_id in U.nodes()}

        for node_id in sorted(U.nodes()): # Sort nodes for deterministic updates
            incoming_neighbours = [source for source, target in U.in_edges(node_id) if source != node_id]
            outgoing_neighbours = [target for source, target in U.out_edges(node_id) if target != node_id]
            
            all_neighbor_labels = [current_node_labels[n] for n in incoming_neighbours + outgoing_neighbours if n in current_node_labels]
            unique_neighbor_labels = list(set(all_neighbor_labels))

            dict_sum_labels = {}
            for label in unique_neighbor_labels:
                positive_w = 0
                for neighbor_id in incoming_neighbours + outgoing_neighbours:
                    if current_node_labels.get(neighbor_id) == label:
                        if U.has_edge(neighbor_id, node_id):  # Incoming edge
                            positive_w += U[neighbor_id][node_id].get('direction weight', 0)
                        if U.has_edge(node_id, neighbor_id):  # Outgoing edge
                            positive_w += U[node_id][neighbor_id].get('direction weight', 0)
                dict_sum_labels[label] = positive_w
            
            # Sort for deterministic tie-breaking in choose_label
            sorted_dict_sum_labels = dict(sorted(dict_sum_labels.items()))

            chosen_label = choose_label(sorted_dict_sum_labels, U, U.nodes[node_id]['label'])
            U.nodes[node_id]['label'] = chosen_label
        
        # After all nodes are updated, record the state for stability check
        G_array.append(copy.deepcopy(U))
        if len(G_array) >= 2 and steady(G_array):
            break

    final_labels = list(set([data['label'] for _, data in U.nodes(data=True)]))
    print(f'Label propagation done with {len(final_labels)} communities')

    if write_graphml:
        H = copy.deepcopy(U)
        for node_h_id, node_h_data in H.nodes(data=True):
            node_h_data['label'] = f"{node_h_id}->{node_h_data['label']}"
        nx.write_graphml(H, name_of_file)

    return U


def affinity_propagation(G):
    """
    Applies Affinity Propagation clustering to the graph based on a similarity matrix
    derived from shared incoming and outgoing connections.

    Args:
        G (nx.DiGraph): The NetworkX directed graph.

    Returns:
        int: The estimated number of clusters found by Affinity Propagation.
    """
    node_list = list(G.nodes())
    len_nodes = len(node_list)

    # Initialize adjacency matrices for positive and negative edges
    A_minus = np.zeros((len_nodes, len_nodes))
    A_plus = np.zeros((len_nodes, len_nodes))

    node_to_idx = {node: i for i, node in enumerate(node_list)}

    for u, v, data in G.edges(data=True):
        u_idx = node_to_idx[u]
        v_idx = node_to_idx[v]
        if data.get('negative', False):
            A_minus[u_idx][v_idx] = 1
        else:
            A_plus[u_idx][v_idx] = 1

    A_plus_t = A_plus.transpose()
    A_minus_t = A_minus.transpose()

    B_plus = np.dot(A_plus, A_plus_t)
    C_plus = np.dot(A_plus_t, A_plus)

    B_minus = np.dot(A_minus, A_minus_t)
    C_minus = np.dot(A_minus_t, A_minus)

    S = np.zeros((len_nodes, len_nodes))
    for i in range(len_nodes):
        for j in range(len_nodes):
            sum_A_minus_i = np.sum(A_minus[i, :]) + np.sum(A_minus[:, i])
            sum_A_plus_i = np.sum(A_plus[i, :]) + np.sum(A_plus[:, i])

            sum_A_minus_j = np.sum(A_minus[j, :]) + np.sum(A_minus[:, j])
            sum_A_plus_j = np.sum(A_plus[j, :]) + np.sum(A_plus[:, j])

            denominator = max(sum_A_minus_i + sum_A_plus_i, sum_A_minus_j + sum_A_plus_j)
            
            # Avoid division by zero
            if denominator == 0:
                S[i][j] = 0
            else:
                S[i][j] = (B_plus[i][j] + B_minus[i][j] + C_plus[i][j] + C_minus[i][j]) / denominator
    
    # Affinity Propagation algorithm
    af = AffinityPropagation(random_state=10, damping=0.9, max_iter=1000).fit(S)
    cluster_centers_indices = af.cluster_centers_indices_
    labels = af.labels_

    n_clusters = len(cluster_centers_indices)
    print(f"Estimated number of clusters: {n_clusters}")
    
    # Assign labels back to graph nodes if needed, though the prompt only asks for number of clusters
    # for i, node_id in enumerate(node_list):
    #     G.nodes[node_id]['label'] = labels[i]

    return n_clusters


def label_propagation_incoming_edges(G, steps=30, write_graphml=False, name_of_file='G_LPed.graphml'):
    """
    Performs label propagation where nodes primarily adopt labels based on their incoming neighbors,
    considering both positive and negative edge weights.

    Args:
        G (nx.DiGraph): The NetworkX directed graph. Edges should have 'negative' attribute.
                        Nodes are initialized with an implicit label (their own name).
        steps (int, optional): The maximum number of label propagation iterations. Defaults to 30.
        write_graphml (bool, optional): If True, writes the final graph with labels to a GraphML file. Defaults to False.
        name_of_file (str, optional): The name of the GraphML file if write_graphml is True. Defaults to 'G_LPed.graphml'.

    Returns:
        nx.DiGraph: The graph with updated node labels representing communities.
    """
    U = nx.DiGraph(G) # Create a mutable copy

    # Initialize node labels with their own names
    for node_id in U.nodes():
        U.nodes[node_id]['label'] = node_id

    # Calculate edge weights
    for u, v, data in U.edges(data=True):
        if u == v: # Skip self-loops
            data['direction weight'] = 0
            continue

        if data.get('negative', False) is False:
            all_edges_plus_u = [e for e in U.edges(u, data=True) if e[2].get('negative', False) is False] + \
                               [e for e in U.in_edges(u, data=True) if e[2].get('negative', False) is False]
            all_edges_plus_v = [e for e in U.edges(v, data=True) if e[2].get('negative', False) is False] + \
                               [e for e in U.in_edges(v, data=True) if e[2].get('negative', False) is False]
            k_plus_out_u = len([e for e in U.out_edges(u, data=True) if e[2].get('negative', False) is False])
            k_plus_in_v = len([e for e in U.in_edges(v, data=True) if e[2].get('negative', False) is False])
            
            denominator = (len(all_edges_plus_u) * len(all_edges_plus_v))
            data['direction weight'] = 1 - ((k_plus_out_u * k_plus_in_v) / denominator) if denominator else 0
        else:
            all_edges_neg_u = [e for e in U.edges(u, data=True) if e[2].get('negative', False) is True] + \
                              [e for e in U.in_edges(u, data=True) if e[2].get('negative', False) is True]
            all_edges_neg_v = [e for e in U.edges(v, data=True) if e[2].get('negative', False) is True] + \
                              [e for e in U.in_edges(v, data=True) if e[2].get('negative', False) is True]
            k_neg_out_u = len([e for e in U.out_edges(u, data=True) if e[2].get('negative', False) is True])
            k_neg_in_v = len([e for e in U.in_edges(v, data=True) if e[2].get('negative', False) is True])

            denominator = (len(all_edges_neg_u) * len(all_edges_neg_v))
            data['direction weight'] = 1 - ((k_neg_out_u * k_neg_in_v) / denominator) if denominator else 0

    # Label Propagation
    G_array = []
    for _ in range(steps):
        current_node_labels = {node_id: U.nodes[node_id]['label'] for node_id in U.nodes()}

        for node_id in sorted(U.nodes()): # Sort nodes for deterministic updates
            incoming_neighbours = [source for source, target in U.in_edges(node_id)]
            
            all_neighbor_labels = [current_node_labels[n] for n in incoming_neighbours if n in current_node_labels]
            unique_neighbor_labels = list(set(all_neighbor_labels))

            dict_sum_labels = {}
            for label in unique_neighbor_labels:
                positive_w = 0
                negative_w = 0
                for neighbor_id in incoming_neighbours:
                    if current_node_labels.get(neighbor_id) == label:
                        if U.has_edge(neighbor_id, node_id):  # Incoming edge
                            edge_data = U[neighbor_id][node_id]
                            if not edge_data.get('negative', False):
                                positive_w += edge_data.get('direction weight', 0)
                            else:
                                negative_w += edge_data.get('direction weight', 0)
                dict_sum_labels[label] = positive_w - negative_w
            
            # Sort for deterministic tie-breaking in choose_label
            sorted_dict_sum_labels = dict(sorted(dict_sum_labels.items()))

            chosen_label = choose_label(sorted_dict_sum_labels, U, U.nodes[node_id]['label'])
            U.nodes[node_id]['label'] = chosen_label

        G_array.append(copy.deepcopy(U))
        if len(G_array) >= 2 and steady(G_array):
            break

    final_labels = list(set([data['label'] for _, data in U.nodes(data=True)]))
    print(f'Label propagation done with {len(final_labels)} communities')

    if write_graphml:
        H = copy.deepcopy(U)
        for node_h_id, node_h_data in H.nodes(data=True):
            node_h_data['label'] = f"{node_h_id}->{node_h_data['label']}"
        nx.write_graphml(H, name_of_file)

    return U


def label_propagation_incoming_edges_unweighted(G, steps=30, write_graphml=False, name_of_file='G_LPed.graphml'):
    """
    Performs unweighted label propagation where nodes primarily adopt labels
    based on the most frequent label among their incoming neighbors.

    Args:
        G (nx.DiGraph): The NetworkX directed graph. Nodes are initialized with an implicit label (their own name).
        steps (int, optional): The maximum number of label propagation iterations. Defaults to 30.
        write_graphml (bool, optional): If True, writes the final graph with labels to a GraphML file. Defaults to False.
        name_of_file (str, optional): The name of the GraphML file if write_graphml is True. Defaults to 'G_LPed.graphml'.

    Returns:
        nx.DiGraph: The graph with updated node labels representing communities.
    """
    U = nx.DiGraph(G) # Create a mutable copy

    # Initialize node labels with their own names
    for node_id in U.nodes():
        U.nodes[node_id]['label'] = node_id

    for _ in range(steps):
        # Create a copy of labels to ensure asynchronous updates (all nodes use labels from previous step)
        current_node_labels = {node_id: U.nodes[node_id]['label'] for node_id in U.nodes()}

        for node_id in sorted(U.nodes()):  # Sort nodes for deterministic updates
            incoming_neighbors = [source for source, target in U.in_edges(node_id)]
            
            if not incoming_neighbors:
                continue # No incoming neighbors, label remains unchanged

            neighbor_labels = [current_node_labels[n] for n in incoming_neighbors if n in current_node_labels]
            
            if not neighbor_labels:
                continue # No labeled incoming neighbors

            # Find the most frequent label
            from collections import Counter
            label_counts = Counter(neighbor_labels)
            
            max_count = max(label_counts.values())
            most_frequent_labels = [label for label, count in label_counts.items() if count == max_count]

            # Tie-breaking: if multiple labels have the same highest frequency, choose the one with the lexicographically smallest value.
            chosen_label = sorted(most_frequent_labels)[0]
            U.nodes[node_id]['label'] = chosen_label

    final_labels = list(set([data['label'] for _, data in U.nodes(data=True)]))
    print(f'Label propagation done with {len(final_labels)} communities')

    if write_graphml:
        H = copy.deepcopy(U)
        for node_h_id, node_h_data in H.nodes(data=True):
            node_h_data['label'] = f"{node_h_id}->{node_h_data['label']}"
        nx.write_graphml(H, name_of_file)

    return U


def leiden_partitions(G):
    """
    Applies the Leiden algorithm for community detection to the graph.
    The 'negative' edge attribute is preserved during conversion to igraph.

    Args:
        G (nx.DiGraph): The NetworkX directed graph. Edges are expected to have a 'negative' attribute.

    Returns:
        nx.DiGraph: The original graph with updated node labels corresponding to Leiden partitions.
    """
    U = nx.DiGraph(G) # Create a mutable copy to work with labels

    node_list = list(U.nodes())
    mapping = {node: i for i, node in enumerate(node_list)}
    rev_mapping = {i: node for i, node in enumerate(node_list)}

    g = ig.Graph(directed=True)
    g.add_vertices(len(node_list))

    negatives = []
    for u, v, data in U.edges(data=True):
        g.add_edge(mapping[u], mapping[v])
        negatives.append(data.get('negative', False))

    g.es['negative'] = negatives

    # Apply Leiden algorithm
    part = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition)

    # Update node labels in the NetworkX graph U
    for cluster in part:
        # Choose the label of the first node in the cluster (after reverse mapping) as the community label
        representative_label = rev_mapping[cluster[0]]
        for index in cluster:
            node_id = rev_mapping[index]
            U.nodes[node_id]['label'] = representative_label

    print(f'Leiden algorithm done with {len(part)} clusters')
    return U


def influential_nodes_in_communities_one_fvs_member(G, FVS_list):
    """
    Identifies one influential node within each community that is also part of the
    Feedback Vertex Set (FVS_list), using a rank-based system for influence.
    The influence is based on a combined rank of betweenness centrality and weighted cycle score.

    Args:
        G (nx.DiGraph): A NetworkX directed graph with nodes having a 'label' attribute.
        FVS_list (list): A list of node names belonging to the Feedback Vertex Set.

    Returns:
        dict: A dictionary where keys are community labels and values are the
              selected influential FVS member from that community.
    """
    labels = list(set([x[-1]['label'] for x in G.nodes(data=True)]))
    communities = {l: [x[0] for x in G.nodes(data=True) if x[-1]['label'] == l] for l in labels}

    control_in_communities = {}

    for key, community_nodes in communities.items():
        if not community_nodes:
            continue

        community_subgraph = G.subgraph(community_nodes)
        
        # Calculate centrality and cycle scores for nodes within this community
        dict_betweenness = nx.betweenness_centrality(community_subgraph)
        cycle_score = weighted_cycle_score(community_subgraph) # Assuming this function is available

        # --- Ranking System ---
        # Rank nodes based on betweenness centrality (higher BC = lower rank number)
        # Use node count for ranking to make it consistent with the original `influential_nodes_in_communities_rank_based`
        bc_sorted = sorted(dict_betweenness.items(), key=lambda x: x[1], reverse=True)
        bc_ranks = {node: len(community_nodes) - i for i, (node, _) in enumerate(bc_sorted)}

        # Rank nodes based on weighted cycle score (lower cycle score = lower rank number)
        # The original `weighted_cycle_score` gives higher values for 'more central' nodes
        # To make it rank-based consistently (lower rank number for better), we sort in ascending order
        # and then reverse the rank.
        cycle_sorted = sorted(cycle_score.items(), key=lambda x: x[1], reverse=True) # Sort descending for higher scores being better
        cycle_ranks = {node: len(community_nodes) - i for i, (node, _) in enumerate(cycle_sorted)}


        total_rank_score = {}
        for node in community_nodes:
            # Sum the ranks. Lower total rank score means more influential.
            # Using get with 0 default for nodes that might somehow miss a score
            total_rank_score[node] = bc_ranks.get(node, 0) + cycle_ranks.get(node, 0)

        # Sort nodes by their total rank score in ascending order (lower rank score first)
        sorted_nodes_by_rank = sorted(total_rank_score, key=total_rank_score.get)
        
        # Find the first node in the sorted list that is also in the FVS_list
        for node in sorted_nodes_by_rank:
            if node in FVS_list:
                control_in_communities[key] = node
                break
    return control_in_communities


def influential_nodes_in_ranked_communities(G, FVS_list):
    """
    Identifies influential nodes in communities, prioritizing communities with
    a higher proportion of FVS members and a higher average structural score.
    It then selects the most influential FVS members from these top-ranked communities
    until a target number of FVS members (50% of total FVS) is reached.

    Args:
        G (nx.DiGraph): A NetworkX directed graph with nodes having a 'label' attribute.
        FVS_list (list): A list of node names belonging to the Feedback Vertex Set.

    Returns:
        dict: A dictionary where keys are community labels and values are lists of
              selected influential FVS members from those communities.
    """
    wanted = int(0.5 * len(FVS_list)) if FVS_list else 0

    labels = list(set([x[-1]['label'] for x in G.nodes(data=True)]))
    communities = {l: [x[0] for x in G.nodes(data=True) if x[-1]['label'] == l] for l in labels}

    communities_fvs_score = {}
    for com_key, nodes_in_community in communities.items():
        fvs_in_community = mutual_items(nodes_in_community, FVS_list)
        if FVS_list:
            communities_fvs_score[com_key] = len(fvs_in_community) / len(FVS_list)
        else:
            communities_fvs_score[com_key] = 0

    average_structure_score = average_structure_score_in_communities(G)

    total_community_score = {}
    for ky in communities_fvs_score.keys():
        total_community_score[ky] = average_structure_score.get(ky, 0) + communities_fvs_score.get(ky, 0)

    # Sort communities by their total score in descending order
    sorted_key_communities = sorted(total_community_score, key=total_community_score.get, reverse=True)

    control_in_communities = {}
    ranked_communities_with_sorted_nodes = {}
    counter = 0

    for key in sorted_key_communities:
        if counter >= wanted:
            break

        community_nodes = communities[key]
        community_subgraph = G.subgraph(community_nodes)
        dict_betweenness_centrality_this_community = nx.betweenness_centrality(community_subgraph)
        cycle_score = weighted_cycle_score(community_subgraph)

        total_score_within_this_community = {
            node: dict_betweenness_centrality_this_community.get(node, 0) + cycle_score.get(node, 0)
            for node in community_nodes
        }

        sorted_nodes_by_score = sorted(total_score_within_this_community, key=total_score_within_this_community.get, reverse=True)
        ranked_communities_with_sorted_nodes[key] = sorted_nodes_by_score

        for item in sorted_nodes_by_score:
            if item in FVS_list:
                control_in_communities.setdefault(key, []).append(item)
                counter += 1
                break # Move to the next community after finding one FVS member

    # Continue adding FVS members if 'wanted' target is not met
    while counter < wanted:
        found_additional_fvs = False
        for c1 in ranked_communities_with_sorted_nodes.keys():
            current_fvs_in_community = control_in_communities.get(c1, [])
            nodes_to_consider = [x for x in ranked_communities_with_sorted_nodes[c1] if x not in current_fvs_in_community and x in FVS_list]
            
            if nodes_to_consider:
                control_in_communities.setdefault(c1, []).append(nodes_to_consider[0])
                counter += 1
                found_additional_fvs = True
                if counter >= wanted:
                    break
        if not found_additional_fvs: # No more FVS members to add
            break

    return control_in_communities


def form_network(contents):
    """
    Parses a Boolean network definition string (e.g., from a .bnet file)
    and constructs a NetworkX directed graph. It identifies nodes and
    edges, and marks edges as positive or negative based on 'not' operators.

    Args:
        contents (str): A string containing the Boolean network definition.
                        Example format: "Node1 *= (not Node2) or Node3"

    Returns:
        nx.DiGraph: A NetworkX directed graph representing the Boolean network,
                    with edges having a 'negative' attribute (True/False).
    """
    # Standardize input for easier parsing
    contents = contents.replace(',\t', '*=').replace('!', "not ").replace('|', 'or').replace('&', "and")
    lines = contents.splitlines()

    G = nx.DiGraph()
    rules = [line for line in lines if '*=' in line]

    for rule in rules:
        parts = rule.split('*=')
        target = parts[0].strip()
        expression = parts[1].strip()

        # Extract all potential regulators by splitting on logical operators and 'not'
        # Then filter out logical operators and empty strings
        regulators_raw = re.split(r'\s*(?:not|and|or|\(|\))\s*', expression)
        regulators = list(set([r for r in regulators_raw if r and not r.isdigit()]))

        # Identify negative regulators explicitly
        negative_regulators = list(set(re.findall(r"not\s+(\w+)", expression)))

        for r in regulators:
            if not G.has_edge(r, target):
                G.add_edge(r, target)
            
            # Set 'negative' attribute based on whether the regulator is negated
            if r in negative_regulators:
                G[r][target]['negative'] = True
            else:
                G[r][target]['negative'] = False
    return G