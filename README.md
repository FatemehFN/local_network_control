# Local Network Control

This repository contains Python implementations for analyzing and controlling Boolean networks.  The code focuses on identifying influential nodes and computing control sets using approaches such as feedback vertex set (FVS) analysis, betweenness centrality and stable motif (trap space) calculations.  The scripts can be used to process randomly generated networks as well as a collection of curated biological models.

## Features

- **Feedback Vertex Set algorithms** – `FVS.py` and `FVS_localsearch_10_python.py` implement a simulated annealing search to approximate minimum FVS sets in directed graphs.
- **Boolean network utilities** – `BooleanDOI_processing.py` provides functions for generating Boolean rules, expanded networks and other pre–processing steps.
- **Community and motif analysis** – `classification_operations.py` contains helpers for clustering minimal trapping spaces and selecting influential nodes inside communities.
- **Control strategies**
  - `BC_control.py` and `top_10_percent_control.py` evaluate control nodes chosen by betweenness centrality.
  - `control_TOWARD_CC_one_fvs.py` and `control_TOWARD_CC_top_10_percent.py` analyze biological models using FVS members or top‑ranked nodes in each community.
  - `one_fvs_from_each_module_control.py` applies a single FVS node per community approach across generated networks.
- **Random model generation** – `dynamics_operations.py` includes routines to construct networks and create nested canalyzing Boolean rules.

## Requirements

The scripts require Python 3 and the following packages (tested with typical recent versions):

- `networkx`
- `numpy`
- `pystablemotifs`
- `pyboolnet`
- `scipy`
- `scikit-learn`
- `igraph` and `leidenalg`
- `matplotlib`, `seaborn` and `iteration_utilities`

Install the dependencies with `pip`:

```bash
pip install networkx numpy pystablemotifs pyboolnet scipy scikit-learn igraph leidenalg matplotlib seaborn iteration_utilities
```

## Usage

Many scripts expect pre‑generated networks and Boolean models stored in directories such as `corrected_models/` or parameterized folders (`N <n> -k <k> -maxk <m> -t1 <t>/`).  Typical workflow:

1. Generate or load a network in GraphML format.
2. Use `FVS.FVS(G)` to compute a feedback vertex set.
3. Cluster minimal trapping spaces with functions from `classification_operations.py`.
4. Select control nodes and evaluate coverage of phenotypes using one of the control scripts.

Example – computing an approximate FVS for a graph:

```python
import networkx as nx
import FVS

G = nx.gnm_random_graph(10, 20, directed=True)
control_nodes = FVS.FVS(G)
print(control_nodes)
```

Refer to the individual scripts for the full pipeline of experiments.

## Repository Contents

- `BC_control.py` – Processes generated networks, selecting top nodes by betweenness centrality and measuring coverage.
- `BooleanDOI_processing.py` – Helper functions to construct Boolean rules and expanded networks.
- `FVS.py` / `FVS_localsearch_10_python.py` – Implementation of the simulated annealing algorithm for FVS calculation.
- `classification_operations.py` – Clustering and community analysis utilities used throughout the workflows.
- `control_TOWARD_CC_one_fvs.py` – Applies a single FVS member per community strategy on curated biological models.
- `control_TOWARD_CC_top_10_percent.py` – Uses top ranked nodes per community to drive the model toward desired cell cycles.
- `dynamics_operations.py` – Functions for generating random Boolean networks and rule sets.
- `one_fvs_from_each_module_control.py` – Runs FVS‑based control for each module on generated networks.
- `top_10_percent_control.py` – Compares Leiden vs. label propagation community detection with control from the top 10% of nodes.

The repository does not currently include example data or a dedicated `requirements.txt`.  Depending on the experiment you wish to reproduce, additional files such as Boolean rule sets or network graphs must be placed in the expected folder structure.

## License

No explicit license file is provided.  Please consult the repository owner before redistributing or using the code for other projects.

