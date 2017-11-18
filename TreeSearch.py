#!/usr/bin/python3
# -*- coding: utf-8 -*-
""" 
This is the module of general tree search algorithms. 

Author: Xian Lai
Date: Sep.14, 2017
"""
import numpy as np
import json
from pprint import pprint

import TreeVisual as visual


class TreeSearch():
    """ this class implements the basic operations of tree search, like expand 
    downward, trace backup etc, and different search strategies like BFS, DFS,
    A* etc. It should be used as parent class for specific problem instance.
    And it requires this instance to have the following methods:

    - ProblemInstance._transition(state)
        The transition model takes in state and return possible actions,  
        result states and corresponding step costs.
    - ProblemInstance._heuristicCost(state)
        Calculate and return the heuristic cost given a state.
    - ProblemInstance._isGoal(state)
        Check whether given state is goal state or one of goal states.

    Some abbreviations used in this script:
    gnrt   : Nodes in the same generation or level of tree.
    clst   : Nodes in the same cluster(children of one parent) of tree.
    sibl   : A node in a cluster.
    peerIdx: The index of clst and sibl
    cousin : A node in other cluster in the same level.
    niece  : A node in next generation but is not current node's child

    We store the tree in form of a nested list of dictionaries:
    tree   = [gnrt_0, gnrt_1, gnrt_2, ...]
    gnrt_# = [clst_0, clst_1, clst_2, ...]
    clst_# = [sibl_0, sibl_1, sibl_2, ...]
    sibl_# = {
        'state'      : state of current node, 
        'pathCost'   : the cost of path up to current node, 
        'heurist'    : the heurist cost from current node to goal node,
        'prevAction' : the action transform parent state to this state,
        'expanded'   : whether this node has been expanded,
        'gnrt'       : the generation or level in the tree of current node,
        'clst'       : the cluster index of this node in current generation,
        'sibl'       : the sibling index of this node in current cluster,
        'children'   : the indices of children in next generation,
        'parent'     : [the family index of parent in last gnrt,
                        the sibling index of parent in last gnrt]
    }
    """
    def __init__(self, initState=None, show_progress=False):
        """ initiallize the search tree with initial state.
        """
        self.initState = initState # the initial state
        self.paths     = []        # collection of paths to goal node found
        self.n_nodes   = 1         # the total number of nodes in the tree
        self.n_gnrt    = 0         # the total number of levels in the tree
        self.show = show_progress  # whether to show progress when searching
        self.root      = {         # the root node
            'state':self.initState, 
            'pathCost':0,
            'heuristic':99999,
            'prevAction':'initialize',
            'expanded':False,
            'gnrt':0,
            'clst':0,
            'sibl':0,
            'children':None,
            'parent':[]
        }
        self.tree = [[[self.root]]] # the search tree keeping track of paths


    def _reset(self):
        """ Reset the tree as an unsearched one.
        """
        self.root = {
            'state':self.initState, 
            'pathCost':0,
            'heuristic':99999,
            'prevAction':'initialize',
            'expanded':False,
            'gnrt':0,
            'clst':0,
            'sibl':0,
            'children':None,
            'parent':[]
        }

        self.tree    = [[[self.root]]]
        self.paths   = []
        self.n_nodes = 1
        self.n_gnrt  = 0


    def _compare_list(self, list_0, list_1):
        """ compare the 2 lists by first comparing the 1st element, if equal 
        then comparing the 2nd one and so forth. 
        Return true if list_0 < list_1. Otherwise return false.
        """
        for e0, e1 in zip(list_0, list_1):
            if e0 == e1: continue
            else: return e0 < e1


    def _sort_list(self, values):
        """ take in a list of values, return the rank of values in ascending 
        order as well as the order to pickup elements in the original list to  
        form a new ascending list.
        e.g.
        values  : [0, 0.8, 0.5, 1.0, 1.2, 1.7, 0.3]
        sortedV : [0, 0.3, 0.5, 0.8, 1.0, 1.2, 1.7]
        rank    : [0,   3,   2,   4,   5,   6,   1]
        pickup  : [0,   6,   2,   1,   3,   4,   5]
        """
        sortedV = sorted(values)
        rank    = [sortedV.index(v) for v in values]
        pickup  = [rank.index(i) for i in range(len(values))]
        return rank, pickup


    def _insert_cluster(self, current):
        """ take in a current node to be expanded, return the appropriate clst 
        index for its children cluster. We get this clst index by finding out  
        the closest cousins(left and right) who have been expanded and what  
        their children's clst indices are. And then take the average of these 
        2 indices

        inputs:
        -------
        - current: The current node to be expanded.

        output:
        -------
        - childClstIdx: The index of cluster of current node's children.
        """
        # default indices of clusters of niece on the left and right.
        leftNieceClstIdx, rightNieceClstIdx = -10, 999999

        # the generation index of current node
        currGnrtIdx = current['gnrt']
        currPeerIdx = [current['clst'], current['sibl']]
            
        # if this children cluster is the first one in next gnrt, return the 
        # cluster index as 0.
        if len(self.tree[currGnrtIdx+1]) == 0: return 0 
        else:
            # searching from left, find the left closest niece cluster
            for cluster in self.tree[currGnrtIdx+1]:
                if self._compare_list(cluster[0]['parent'], currPeerIdx):
                    leftNieceClstIdx = cluster[0]['clst']

            # searching from right, find the right closest niece cluster
            for cluster in self.tree[currGnrtIdx+1][::-1]:
                if not self._compare_list(cluster[0]['parent'], currPeerIdx):
                    rightNieceClstIdx = cluster[0]['clst']

            # return the average of left and right cluster index.
            return (leftNieceClstIdx + rightNieceClstIdx) / 2


    def _update_family(self, childGnrtIdx):
        """ after appending children nodes to the tree, the cluster index of  
        newly appended nodes is possibly a float(because it's the average of 
        left and right clst's indices) and the cluster is at the end of list 
        of generation. 

        So we need to sort the clusters by cluster index and integerize them. 
        Further we should update their parent node's 'chidlren' data and their
        children's 'parent' data.

        Assume current gnrt has suffix 'I', chilren gnrt has suffix 'II', and 
        grandchildren has suffix 'III'.

        inputs:
        -------
        - childGnrtIdx: The index of children generation.
        """
        # sort children grnt
        old_clstIdx = [cluster[0]['clst'] for cluster in self.tree[childGnrtIdx]]
        int_clstIdx, new_clstIdx = self._sort_list(old_clstIdx)
        self.tree[childGnrtIdx] = \
            [self.tree[childGnrtIdx][idx] for idx in new_clstIdx]

        # integerize the clst indices
        for idx, cluster in enumerate(self.tree[childGnrtIdx]):
            for node in cluster:
                node['clst'] = idx

        # 'children' data of current layer:
        for clstI in self.tree[childGnrtIdx-1]:
            for siblI in clstI:
                old_clstII = siblI['children']
                if old_clstII != None: # if this sibling has children
                    siblI['children'] = int_clstIdx[old_clstIdx.index(old_clstII)]

        # 'parent' data of grandchildren layer if the tree has that gnrt:
        if len(self.tree) >= childGnrtIdx+2:
            for clstIII in self.tree[childGnrtIdx+1]:
                for siblIII in clstIII:
                    old_clstII = siblIII['parent'][0]
                    siblIII['parent'][0] = \
                        int_clstIdx[old_clstIdx.index(old_clstII)]


    def _expand(self, current):
        """ Expand the given node(add new generated nodes to the tree) and  
        return the new nodes should be added to fringe.

        inputs:
        -------
        - current: The node to be expanded.
        output:
        -------
        - children: The nodes in children cluster.
        """
        if self.show: self._print_progress(current)

        children = [] # the list of children nodes result from this expansion

        # find result children states and turn these states into children nodes
        actions, childStates, stepCosts = self._transition(current['state'])
        n_children = len(childStates)

        # if current state has children states
        if n_children > 0: 
            # if children generation didn't exist, add new generation 
            childGnrtIdx = current['gnrt']+1
            if childGnrtIdx == len(self.tree): self.tree.append([])    
            
            childClstIdx = self._insert_cluster(current)
            childSiblIdx = 0

            for i in range(n_children):
                child = {
                    'state': childStates[i], 
                    'pathCost': (current['pathCost'] + stepCosts[i]),
                    'heuristic':self._heuristicCost(childStates[i]),
                    'prevAction': actions[i],
                    'expanded':False,
                    'gnrt': childGnrtIdx, 
                    'clst':childClstIdx,
                    'sibl':childSiblIdx,
                    'children':None,
                    'parent': [current['clst'], current['sibl']]
                }
                children.append(child)
                childSiblIdx += 1

            # append these children nodes to the tree
            self.tree[childGnrtIdx].append(children) 
            # update current node's data
            current['children'] = childClstIdx
            current['expanded'] = True
            # clean the children generation's cluster indices and update 
            # corresponding nodes's data
            self._update_family(childGnrtIdx)
            self.n_nodes += len(childStates) # update number of nodes
                
        return children


    def _search(self, fringe, maxNodes=np.inf, maxLayers=np.inf):
        """ perform search based on given fringe and stop criteria like maximal 
        nodes and maximal generations.

        The search procedure will stop in any of 3 conditions below:
            - maxNodes is set and reached.
            - maxLayers is set and reached.
            - Neither maxNodes or maxLayers is set and goal is found.
        """
        print("%s search begins:" % self.searchType)
        self._print_stopCriterion(maxNodes, maxLayers)
        
        while self.n_nodes < maxNodes:
            # if the program has nothing to expand, end searching
            if fringe.isEmpty(): break
            # pop the node to be expanded out of fringe 
            currentNode = fringe.pop() 

            # if maxLayers is set and reached, skip expanding this node
            if (maxLayers != np.inf) and (currentNode['gnrt'] == maxLayers): 
                continue
            else:
                childrenNodes = self._expand(currentNode)
                goalNodes     = self._check_goal(childrenNodes)
                # stop searching when goal is found if neither maxNodes or 
                # maxGnrt are set. Otherwise, keep searching until hit maxNodes.
                if (maxNodes == np.inf) and (maxLayers == np.inf):
                    if len(goalNodes) != 0: 
                        break

                self.n_gnrt = len(self.tree) # update tree depth
                fringe.push(childrenNodes) # add new children nodes into fringe

        print("Search ends")
    

    def _check_goal(self, childrenNodes):
        """ check whether there is goal node in given list of nodes. If so, 
        trace the path and return the goal nodes. Otherwise return empty list.
        """
        goalNodes = []

        # test whether each child is goal
        for node in childrenNodes:
            isGoal = self._isGoal(node['state'])
            if isGoal: goalNodes.append(node)

        # if there is goal nodes, trace paths
        for goalNode in goalNodes: self._trace_path(goalNode)

        return goalNodes


    def _trace_path(self, goalNode):
        """ trace the path up to root with given goal node. And add this path 
        to path collection. Path is a list of nodes.
        """
        path = [goalNode]

        # while not root, identify the parent node and add to path
        while path[-1]['gnrt'] != 0: 
            parentGnrt = path[-1]['gnrt']-1
            parentClst = path[-1]['parent'][0]
            parentSibl = path[-1]['parent'][1]
            path.append(self.tree[parentGnrt][parentClst][parentSibl])

        # reverse the path found so it starts with root node.
        self.paths.append(path[::-1])


    def _print_stopCriterion(self, mNodes, mGnrts):
        """ print out the stoping criteria based on the setting of maximal  
        number of nodes and maximal number of generations.
        """
        if mNodes == np.inf: mNodes = "infinite"
        else: mNodes = str(mNodes)

        if mGnrts == np.inf: mGnrts = "infinite"
        else: mGnrts = str(mGnrts)

        print("the search will stop when %s nodes or \
            \nevery nodes in %s generations are discovered"% (mNodes, mGnrts))


    def _print_progress(self, current):
        """ print the progress showing current expanding information and general
        information

        inputs:
        -------
        - current: The node to be expanded.
        """
        print(
            "\nnow expanding: generation:(%d); clst:(%d); sibl:(%d); state:(%s)"
            %(current['gnrt'], current['clst'], current['sibl'], current['state'])
        )

        print(
            "total number of nodes : %d, \ntotal number of layers : %d,\
            \nnumber of paths found : %d" 
            %(self.n_nodes, self.n_gnrt, len(self.paths))
        )
                

    def print_paths(self):
        """ print out the information extracted from paths in format:
        [path_0, path_1, ...]
        path_# = [
            {'action':action_0, 'state':state_0, 'pathCost':pathCost_0}, 
            {'action':action_1, 'state':state_1, 'pathCost':pathCost_1}, 
            ...
        ]
        """
        pathsInfo = []
        for path in self.paths:
            pathInfo = []
            for node in path:
                nodeInfo = {
                    'action':node['prevAction'], 
                    'state':node['state'], 
                    'pathCost':node['pathCost']
                }
                pathInfo.append(nodeInfo)
            pathsInfo.append(pathInfo)

        print('\n%d paths found:' % len(pathsInfo))
        pprint(pathsInfo)
        

    def plot_tree(self, diameter=50, background='dark', title='search tree',
                  ls='-', a=0.8):
        """ plot out the search tree in a polar fig.
        """
        params = {'diameter':diameter, 'background':background}
        plot   = visual.PolarPlot(**params)

        plot.plot_tree(self.tree, self.paths, title=title, ls=ls, a=a)


    def export(self):
        """ export tree nodes and paths as JSON files.
        """
        with open('tree.json', 'w') as outfile:
            json.dump(self.tree, outfile)
        with open('paths.json', 'w') as outfile:
            json.dump(self.paths, outfile)


    #------------------------- SEARCHING STRATEGIES --------------------------
    # uninformed search
    def breadthFirstSearch(self, maxNodes=np.inf, maxLayers=np.inf):
        """ perform breadth first search using FIFO as fringe.
        """
        self.searchType = "BFS"
        fringe = Queue(self.root, q_type='FIFO')
        self._search(fringe, maxNodes=maxNodes, maxLayers=maxLayers)


    def depthFirstSearch(self, maxNodes=np.inf, maxLayers=np.inf):
        """ perform Depth First Search using LIFO as fringe.
        """
        self.searchType = "DFS"
        fringe = Queue(self.root, q_type='LIFO')
        self._search(fringe, maxNodes=maxNodes, maxLayers=maxLayers)


    def uniformCostSearch(self, maxNodes=np.inf, maxLayers=np.inf):
        """ perform Uniform Cost Search using priority queue as fringe and  
        path costs as weight in fringe.
        """
        self.searchType = "UCS"
        fringe = Queue(self.root, q_type='g')
        self._search(fringe, maxNodes=maxNodes, maxLayers=maxLayers)


    def iterativeDeepeningSearch(self, maxDepth=5):
        """ perform Iterative Deepening Search by iteratively perform DFS with 
        maximal generation increasing.
        """
        self.searchType = "IDS"
        for i in range(1, maxDepth+1): 
            self.depthFirstSearch(maxLayers=i)
            if len(self.paths) > 0: break
            self._reset()


    # informed search
    def bestFirstSearch(self, maxNodes=np.inf, maxLayers=np.inf):
        """ perform best first search using priority queue as fringe and  
        heuristic cost as weight in fringe.
        """
        self.searchType = "bestFS"
        fringe = Queue(self.root, q_type='h')
        self._search(fringe, maxNodes=maxNodes, maxLayers=maxLayers)


    def aStarSearch(self, maxNodes=np.inf, maxLayers=np.inf):
        """ perform A star Search using priority queue as fringe and path costs 
        plus heuristic cost as weight in fringe.
        """
        self.searchType = "aStar"
        fringe = Queue(self.root, q_type='g+h')
        self._search(fringe, maxNodes=maxNodes, maxLayers=maxLayers)

    
class Queue():

    """ This class implements the queue data type used as the fringe in the 
    search tree. This queue can be turned into FIFO, LIFO and priority queue
    by specifying the q_type argument:
    
    - LIFO queue : 'lifo'
    - FIFO queue : 'fifo'
    - Priority queue : 
        - Using path cost as weight : 'g'
        - Using heuristic cost as weight : 'h'
        - Using path cost and heuristic cost as weight : 'g+h'
    """

    def __init__(self, rootNode, q_type='lifo'):
        """ initialize the queue with specific type and root node of tree.
        inputs:
        -------
        - rootNode: The starting fringe always has the root node in it.
        - q_type: The type of queue.
        """
        self.type     = q_type
        self.elements = []
        self.push(rootNode)

    def isEmpty(self):
        """ Check whether the queue is empty.
        """
        return self.elements == []


    def size(self):
        """ Return the number of elements in the queue.
        """
        return len(self.elements)


    def update_weights(self):
        """ Update the weights of elements based on queue type. The element 
        with smallest weight will be pop first.
        """
        size = self.size()

        # if queue is a lifo, update weights to range(size, 0, -1)
        # if queue is a fifo, update weights to range(0,size)
        if self.type == 'lifo':
            weights = list(range(0, size))[::-1]
        elif self.type == 'fifo':
            weights = list(range(0, size))
        # if queue is a priority queue, then weights are equal to path costs 
        # or heuristic or the combination of each node element.
        elif self.type == 'g+h':
            weights = [i['pathCost']+i['heuristic'] for i in self.elements]
        elif self.type == 'h':
            weights = [i['heuristic'] for i in self.elements]
        else:
            weights = [i['pathCost'] for i in self.elements]

        self.weights = weights


    def push(self, elements):
        """ push input elements into the queue and update their weights. 
        """
        # if the input is just one element, wrap it with a list
        if type(elements) != list: elements = [elements]
        self.elements.extend(elements)
        self.update_weights() # update weights for each elements after pushing

    def pop(self, weights=None):
        """ pop the element in queue with smallest weight.
        """
        self.update_weights()
        idx = self.weights.index(min(self.weights))
        return self.elements.pop(idx)


    
    