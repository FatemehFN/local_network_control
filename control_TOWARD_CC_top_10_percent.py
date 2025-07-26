import pickle
import networkx as nx
import copy
import pystablemotifs as psm
from pyboolnet.trap_spaces import compute_trap_spaces
import BooleanDOI_processing as BDOIp  # Assuming this is another custom module
import operator
import classification_operations as CLO  # Assuming this is the cleaned-up code from previous turns
import FVS  # Assuming this is a custom module for Feedback Vertex Set calculation
from iteration_utilities import deepflatten
import os
from statistics import mean

# --- Configuration Parameters ---
STEPS = 60  # Number of steps for label propagation (if used)
# List of Boolean network models to process
MODELS = [
    'Senescence Associated Secretory Phenotype.txt',
    'Lymphopoiesis Regulatory Network.txt',
    'MAPK Cancer Cell Fate Network.txt',
    'Signaling in Macrophage Activation.txt',
    'IL-1 Signaling.txt',
    'IL-6 Signalling.txt',
    'HGF Signaling in Keratinocytes.txt',
    'Bortezomib Responses in U266 Human Myeloma Cells.txt',
    'Yeast Apoptosis.txt',
    'Colitis-associated colon cancer.txt',
    'IGVH mutations in chronic lymphocytic leukemia.txt',
    'Differentiation of T lymphocytes.txt',
    'CD4 T cell signaling.txt',
    'B bronchiseptica and T retortaeformis coinfection.txt',
    'T Cell Receptor Signaling.txt',
    'EGFR & ErbB Signaling.txt',
    'Glucose Repression Signaling 2009.txt',
    'Signaling Pathway for Butanol Production in Clostridium beijerinckii NRRL B-598.txt'
]

FOLDER_PATH = "corrected_models/"
RESULTS_DIR = FOLDER_PATH # Results are written directly into FOLDER_PATH as per original code
NUM_PHENOTYPES = 2  # Number of phenotypes for hierarchical clustering

# Ensure the results directory exists (which is FOLDER_PATH in this case)
os.makedirs(RESULTS_PATH, exist_ok=True)


# --- Output Files ---
# File to store percentage coverage and control size
coverage_control_file = os.path.join(RESULTS_DIR, 'la_percent_coverage_control_size.txt')
f1 = open(coverage_control_file, 'w')

# File to store model name and number of communities
communities_file = os.path.join(RESULTS_DIR, 'la_name_number_of_communities.txt')
f3 = open(communities_file, 'w')


# --- Main Processing Loop ---
for model_name in MODELS:
    print(f"Processing model: {model_name}")

    # --- Load Graph and Find FVS ---
    graphml_path = os.path.join(FOLDER_PATH, 'graphml_files', model_name.replace('.txt', '') + '.graphml')
    if not os.path.exists(graphml_path):
        print(f"GraphML file not found: {graphml_path}. Skipping model.")
        continue
    G = nx.read_graphml(graphml_path)

    # Calculate Feedback Vertex Set (FVS) size
    fvs_size = len(FVS.FVS(G))
    print(f"FVS size: {fvs_size}")

    # --- Community Detection ---
    # The original code had several commented-out label propagation methods.
    # The last uncommented one (leiden_partitions) is chosen here.
    # If other methods are desired, uncomment the relevant line.
    # G = CLO.label_propagation(G, steps=STEPS, write_graphml=False)
    # G = CLO.label_propagation_unsigned(G, steps=STEPS, write_graphml=False)
    # G = CLO.label_propagation_incoming_edges(G, steps=STEPS, write_graphml=False)
    # G = CLO.label_propagation_incoming_edges_unweighted(G, steps=STEPS, write_graphml=False)
    G = CLO.leiden_partitions(G)  # Using Leiden algorithm for partitioning

    # Get the unique community labels after partitioning
    labels = list(set([node_data['label'] for _, node_data in G.nodes(data=True)]))
    num_communities = len(labels)
    f3.write(f"{model_name.replace('.txt', '')} {num_communities}\n")
    print(f"Number of communities after partitioning: {num_communities}")

    # --- Identify Influential Nodes in Communities (Rank-based) ---
    # This function identifies influential nodes based on a combined rank
    # of betweenness centrality and weighted cycle score within each community.
    # It returns a dictionary where keys are community labels and values are lists of influential nodes.
    control_in_communities_dict, _, _, _, _ = CLO.influential_nodes_in_communities_rank_based(G)
    
    # Flatten the list of influential nodes from all communities into a single list
    control_nodes = list(deepflatten(control_in_communities_dict.values()))
    
    print('Selected structural control nodes:', control_nodes)

    # --- Load Boolean Model and Calculate Trapping Spaces ---
    boolean_model_path = os.path.join(FOLDER_PATH, model_name)
    if not os.path.exists(boolean_model_path):
        print(f"Boolean model file not found: {boolean_model_path}. Skipping LDOI calculation.")
        f1.write(f"{model_name.replace('.txt','')} 0.0 0\n") # Write default values if model file is missing
        continue

    with open(boolean_model_path, 'r') as f:
        contents = f.read()

    # Pre-process content for pyboolnet (standardizing logical operators)
    contents = contents.replace('||', '|').replace('&&', '&')

    # Create prime implicants from the Boolean model definition
    primes = psm.format.create_primes(contents)

    # Compute minimal trapping spaces (stable motifs)
    min_trap_spaces = compute_trap_spaces(primes, 'min')

    if not min_trap_spaces:
        print(f"No minimal trapping spaces found for {model_name}. Skipping LDOI calculation.")
        f1.write(f"{model_name.replace('.txt','')} 0.0 0\n") # Write default values if no trap spaces
        continue

    # Hierarchically cluster the minimal trapping spaces into phenotypes
    clustered_mints = CLO.hierarchial_cluster_mints(
        mints=min_trap_spaces,
        list_of_nodes=list(G.nodes()),  # Pass as a list
        no_of_clusters=NUM_PHENOTYPES
    )

    all_percentages = []  # To store percentage coverage for each phenotype
    for group_indices in clustered_mints:
        # Determine the common part (fixed states) for the current cluster/phenotype
        if len(group_indices) > 0:  # Ensure group is not empty
            if len(group_indices) == 1:
                common_part = min_trap_spaces[int(group_indices[0])]
            else:
                common_part = CLO.common_part(min_trap_spaces, group_indices)
        else:
            common_part = {}  # Empty group, empty common part

        print('Common part (phenotype fixed states):', common_part)

        if not common_part:
            print("Empty common part for this cluster. Skipping LDOI calculation.")
            continue # Skip to the next phenotype if the common part is empty

        # Find control nodes that are part of the common_part
        common_part_keys = common_part.keys()
        mutual_with_control = CLO.mutual_items(list(common_part_keys), control_nodes)

        # Create a dictionary of the fixed states for the effective control nodes
        dict_control = {c_n: common_part[c_n] for c_n in mutual_with_control}

        if dict_control:
            # Calculate the Logical Domain of Influence (LDOI) for the control nodes
            LDOI = psm.drivers.logical_domain_of_influence(dict_control, primes)

            # Combine LDOI fixed states with the control node fixed states
            resultant_fixed_states = {**LDOI[0], **dict_control}
            print('Constants (LDOI + control):', resultant_fixed_states)

            # Calculate how many of the phenotype's fixed states are covered by the LDOI
            number_coverage = CLO.number_covered_in_phenotype(
                resultant_fixed_states,
                common_part
            )

            # Calculate percentage coverage for this phenotype
            percentage = number_coverage / len(common_part)
            print('Percentage coverage for phenotype:', percentage)
            all_percentages.append(percentage)
        else:
            print("No effective control nodes for this common part. Skipping LDOI calculation for this phenotype.")

    # Calculate average percentage coverage across all phenotypes for the current model
    if all_percentages:
        avg_percentage_coverage = mean(all_percentages)
    else:
        avg_percentage_coverage = 0.0  # If no phenotypes or no effective control, coverage is 0

    # Write results to file1
    f1.write(f"{model_name.replace('.txt','')} {avg_percentage_coverage:.4f} {len(control_nodes)}\n")

# --- Close Files ---
f1.close()
f3.close()

print("\nAll models processed. Results saved to:")
print(f"- {coverage_control_file}")
print(f"- {communities_file}")