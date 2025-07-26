# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 13:34:55 2023

@author: Eli
"""

import numpy as np
import pystablemotifs as sm
import os
from pyboolnet.trap_spaces import compute_trap_spaces
import networkx as nx


def addNegativeEdges(G, pNeg = 0.25, seed = 0):
    """
    Function to add negative edges to graph G
    inputs:
        G = Graph to add negative edges to
        pNeg = Probability that an edge is negative
        seed = Random number generator seed
    """


    rng = np.random.default_rng(seed)

    for E in range(len(G.edges(data=True))):
        isNeg = rng.random()
        if(isNeg < pNeg):
            G.edges(data=True)[E][-1].update({'negative':True})
        else:
            G.edges(data=True)[E][-1].update({'negative':False})


    return G


def generateNestedCanalyzingFunctionRoF(G, node, rng, bias = 0.5):

    """
    Function to write a nested canalyzing function for a node as a read-once function (RoF)
    inputs:
        G = Graph
        node = Node to generate rule for
        rng = Random number generator
        bias = Probability that the function output is 1
    output:
        String of nested canalyzing function for given node
    """
    inputs = [e[0] for e in G.in_edges(node)]
    negative = [G.get_edge_data(x, node)['negative'] for x in inputs]
    n = len(inputs)
    if(n == 0):
        rand = rng.random()
        val = 0
        if(rand < bias):
            val = 1
        return node + " *= " + str(val)
    order = list(rng.choice(inputs, n, replace = False))
    funct = str(node)+" *= "
    for i in range(len(order)):
        rand = rng.random()
        if(rand < bias):
            if(negative[inputs.index(order[i])]):
                funct += "not " + str(order[i]) + " and ("
            else:
                funct += str(order[i]) + " or ("
        else:
            if(negative[inputs.index(order[i])]):
                funct += "not " + str(order[i]) + " or ("
            else:
                funct += str(order[i]) + " and ("
    funct = funct[:-5].rstrip()
    funct += ")"*(len(inputs)-1)
    return funct


def generateRulesets(G, numRulesets, directoryPath, fewestAtts = 2, bias = 0.5, seed = 0):
    """
    Function to generate Boolean Nested Canalyzing Rule-sets for a given graph
    inputs:
        G = graph
        numRulesets = number of rule-sets to generate
        directoryPath = path to folder to save rule-sets in
            if the given directory does not exist this function will create a new folder
            this folder will be populated with numbered booleannet files for each rule-set
        fewestAtts = Fewest number of attractors that each rule-set must have
            if this value is set to 0, every generated rule-set will be output without checking attractor number
        bias = Probability that the function output is 1
        seed = Random number generator seed
    """

    numGood = 0
    rng = np.random.default_rng(seed)
    while(numGood < numRulesets):
        print('numgood')
        print(numGood)
        rules = ""
        for node in G.nodes():
            rules+=generateNestedCanalyzingFunctionRoF(G, node, rng)+'\n'
        if(fewestAtts == 0):
            writePath = directoryPath+str(numGood)+'.booleannet'
            isExist = os.path.exists(directoryPath)
            if not isExist:
              os.makedirs(directoryPath)       
            f = open(writePath,'w')
            f.write(rules)
            f.close()
            numGood += 1
        else:
            primes = sm.format.create_primes(rules)
            #max_simulate_size = 0
            #ar = sm.AttractorRepertoire.from_primes(primes, max_simulate_size=max_simulate_size)
            min_trap_spaces = compute_trap_spaces(primes, 'min')
            if len(min_trap_spaces) > fewestAtts and len(min_trap_spaces)<=50:
                writePath = directoryPath+str(numGood)+'.booleannet'
                isExist = os.path.exists(directoryPath)
                if not isExist:
                  os.makedirs(directoryPath)       
                f = open(writePath,'w')
                f.write(rules)
                f.close()
                numGood += 1
                
