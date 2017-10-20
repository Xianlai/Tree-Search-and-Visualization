#!/usr/bin/python
# -*- coding: utf-8 -*-
""" 
This script implements general tree search algorithms. The TreeSearch instance 
should be used as parent class for specific problem instance. And it requires 
the specific problem instance have following methods implemented:
    - Problem._transition(state)
        The transition model takes in state and return possible actions, result 
        states and corresponding step costs.
    - Problem._heuristicCost(state)
        Calculate and return the heuristic cost given a state.
    - Problem._isGoal(state)
        Check whether given state is goal state or one of goal states.
Author: Xian Lai
Date: Sep.14, 2017
"""
import numpy as np
import json
from pprint import pprint

import TreeVisual as visual

class Queue():
    """ this class implements the queue data type which can turn into FIFO, LIFO and 
    priority queue by specifying the q_type argument:
    - LIFO queue : 'lifo'
    - FIFO queue : 'fifo'
    - Priority queue : 
        - Using path cost as weight : 'g'
        - Using heuristic cost as weight : 'h'
        - Using path cost and heuristic cost as weight : 'g+h'
    """
    def __init__(self, rootNode, q_type='lifo'):
        """ initialize the queue with specific type and root node of tree.
        """
        self.type=q_type
        self.elements = []
        self.push(rootNode)

    def isEmpty(self):
        return self.elements == []

    def size(self):
        return len(self.elements)

    def update_weights(self):
        """ update the weights of elements based on queue type.
        """
        size = self.size()

        # if type is a lifo or fifo, change weights to range(0,size) or range(size, 0, -1)
        if self.type == 'lifo':
            weights = list(range(0, size))[::-1]
        elif self.type == 'fifo':
            weights = list(range(0, size))

        # if type is a priority queue, then weights are equal to path costs of each node.
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
        if type(elements) != list:
            elements = [elements]

        self.elements.extend(elements)
        self.update_weights() # update weights for each elements after pushing

    def pop(self, weights=None):
        """ pop the element in queue with smallest weight.
        """
        self.update_weights()
        idx = self.weights.index(min(self.weights))
        return self.elements.pop(idx)


class TreeSearch():
    """ this class implements the basic operations of tree search, like expand 
    downward, trace backup etc, and the different search strategies like BFS,
    DFS, A* etc. 

    We store the tree as a nested list of dictionaries:
    tree = [generation_0, generation_1, generation_2, ...]
    generation_# = [node_0, node_1, ...]
    node_# = {'state':state of current node, 
              'pathCost':the cost of path up to current node, 
              'heurist': the heurist cost from current node to goal node,
              'prevAction':the action transform parent state to this state,
              'expanded': whether this node has been expanded,
              'gnrt':the generation or level in the tree of current node,
              'clst': the cluster index of this node in current generation,
              'sibl': the sibling index of this node in current family,
              'children':the indices of children in next generation,
              'parent':[the family index of parent in last gnrt,
                        the sibling index of parent in last gnrt]}
    """
    def __init__(self, initState=None, show_progress=False):
        """ initiallize the search tree with initial state.
        """
        self.initState = initState
        self.root = {'state':self.initState, 
                     'pathCost':0,
                     'heuristic':99999,
                     'prevAction':'initialize',
                     'expanded':False,
                     'gnrt':0,
                     'clst':0,
                     'sibl':0,
                     'children':None,
                     'parent':[]}

        self.tree = [[[self.root]]]
        self.paths = [] # collection of paths to goal node found
        self.n_nodes = 1 # number of nodes in the tree
        self.n_gnrt = 0 # number of generations in the tree
        self.show_prog = show_progress

    def _reinit(self):

        self.root = {'state':self.initState, 
                     'pathCost':0,
                     'heuristic':99999,
                     'prevAction':'initialize',
                     'expanded':False,
                     'gnrt':0,
                     'clst':0,
                     'sibl':0,
                     'children':None,
                     'parent':[]}

        self.tree = [[[self.root]]]
        self.paths = [] # collection of paths to goal node found
        self.n_nodes = 1 # number of nodes in the tree
        self.n_gnrt = 0 # number of generations in the tree

    def _compare_list(self, list_0, list_1):
        """ compare the 2 lists by first compare their 1st element, if equal then 
        comapre the 2nd element and so forth. Return true if list_0 < list_1.
        """
        for e0, e1 in zip(list_0, list_1):
            if e0 == e1: continue
            else: return e0 < e1

    def _sort_list(self, values):
        """ take in values, return the rank of values in ascending order as well
        as pickup orders in the original value list.
        e.g.
        values  : [0, 0.8, 0.5, 1.0, 1.2, 1.7, 0.3]
        sortedV : [0, 0.3, 0.5, 0.8, 1.0, 1.2, 1.7]
        rank    : [0,   3,   2,   4,   5,   6,   1]
        pickup  : [0,   6,   2,   1,   3,   4,   5]
        """
        sortedV = sorted(values)
        rank = [sortedV.index(v) for v in values]
        pickup = [rank.index(i) for i in range(len(values))]
        return rank, pickup

    def _insert_cluster(self, current):
        """ take in a current node to be expanded, return the appropriate clst 
        index for its children cluster. We get this clst index by finding out the 
        closest cousins(left and right) who have been expanded and what their 
        children's clst indices are. And then take the average of these 2 indices
        """
        # default leftPeer and rightPeer and will remain when currPeer is the only one.
        leftNiece, rightNiece = -10, 999999

        currGnrt, currPeer = current['gnrt'], [current['clst'], current['sibl']]
            
        # if this is the first children cluster in next gnrt
        if len(self.tree[currGnrt+1]) == 0: 
            childClst = 0 # children cluster set to 0
        else:
            # searching from left, find the left closest niece cluster
            for cluster in self.tree[currGnrt+1]:
                if self._compare_list(cluster[0]['parent'], currPeer):
                    leftNiece = cluster[0]['clst']
            
            # searching from right, find the right closest niece cluster
            for cluster in self.tree[currGnrt+1][::-1]:
                if not self._compare_list(cluster[0]['parent'], currPeer):
                    rightNiece = cluster[0]['clst']

            childClst = (leftNiece + rightNiece) / 2
        return childClst 

    def _update_family(self, childGnrt):
        """ after appending children nodes to the tree, the cluster index of new 
        appended nodes is possibly a float(because it's average of left and right 
        clst's) and the cluster is at the end of generation list. 

        So we need to sort the clusters by 'clst' and integerize them. And update
        their parent's chidlren attribute and their children's parent attribute.
        Assume current gnrt is named _I, chilren gnrt is name _II, and grandchildren
        gnrt is named _III.
        """
        # sort children grnt
        old_clst = [cluster[0]['clst'] for cluster in self.tree[childGnrt]]
        int_clst, new_clst = self._sort_list(old_clst)
        self.tree[childGnrt] = [self.tree[childGnrt][idx] for idx in new_clst]

        # integerize the clst indices
        for idx, cluster in zip(range(len(new_clst)), self.tree[childGnrt]):
            for node in cluster:
                node['clst'] = idx

        # children index of current layer:
        for clstI in self.tree[childGnrt-1]:
            for siblI in clstI:
                old_clstII = siblI['children']
                if old_clstII != None: # if this sibling has children
                    siblI['children'] = int_clst[old_clst.index(old_clstII)]

        # parent's cluster index of next2nd layer if the tree has that gnrt:
        if len(self.tree) >= childGnrt+2:
            for clstIII in self.tree[childGnrt+1]:
                for siblIII in clstIII:
                    old_clstII = siblIII['parent'][0]
                    siblIII['parent'][0] = int_clst[old_clst.index(old_clstII)]

    def _expand(self, current):
        """ expand the given node(add new generated nodes to the tree) and return 
        the new nodes should be added to fringe.
        """
        if self.show_prog: self._print_progress(current)

        children = [] # the list of children nodes result from this expansion

        # find result children states and turn these states into children nodes
        actions, childStates, stepCosts = self._transition(current['state'])
        n_children = len(childStates)

        if n_children > 0: 
            # if children generation didn't exist, add new generation 
            childGnrt = current['gnrt']+1
            if childGnrt == len(self.tree): self.tree.append([])    
            
            # look at existing next generation clusters to find right cluster indices
            clst = self._insert_cluster(current); sibl = 0

            for i in range(n_children):
                child = {'state': childStates[i], 
                         'pathCost': (current['pathCost'] + stepCosts[i]),
                         'heuristic':self._heuristicCost(childStates[i]),
                         'prevAction': actions[i],
                         'expanded':False,
                         'gnrt': childGnrt, 
                         'clst':clst,
                         'sibl':sibl,
                         'children':None,
                         'parent': [current['clst'], current['sibl']]
                        }
                children.append(child)
                sibl += 1

            # append these children nodes to the tree
            self.tree[childGnrt].append(children) 
            current['children'] = clst
            current['expanded'] = True

            self._update_family(childGnrt)
            
            self.n_nodes += len(childStates) # update number of nodes
                
        return children

    def _search(self, fringe, maxNodes=np.inf, maxLayers=np.inf):
        """ perform search based on given fringe and stop criteria like maximal 
        nodes and maximal generations.
        """
        print("%s search begins:" % self.searchType); 
        self._print_stopCriterion(maxNodes, maxLayers)
        
        while self.n_nodes < maxNodes:
            # if the program has nothing to expand, end searching
            if fringe.isEmpty(): break
            
            currentNode = fringe.pop() # pop the node out of fringe to be expand
            
            # if maximal generation is set and reached, skip expanding this node
            if (maxLayers != np.inf) and (currentNode['gnrt'] == maxLayers): 
                continue
            else:
                childrenNodes = self._expand(currentNode)
                goalNodes = self._check_goal(childrenNodes)
                
                # stop searching if goal is found when neither maxNodes and 
                # maxGnrt are set. Otherwise, keep searching until hit maxNodes.
                if (maxNodes == np.inf) and (maxLayers == np.inf):
                    if len(goalNodes) != 0: break

                self.n_gnrt = len(self.tree) # update tree depth
                fringe.push(childrenNodes) # add new children nodes into fringe

        print("search ends")
    
    def _check_goal(self, childrenNodes):
        """ check whether there is goal node in given list of nodes. If so, trace
        the path and return the goal nodes. Otherwise return empty list.
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
        """ trace the path up to root with given goal node. And add this path to 
        path collection. Path is a list of nodes.
        """
        path = [goalNode]

        # while not root, identify the parent node and add to path
        while path[-1]['gnrt'] != 0: 
            parentGnrt = path[-1]['gnrt']-1
            parentClst = path[-1]['parent'][0]
            parentSibl = path[-1]['parent'][1]
            path.append(self.tree[parentGnrt][parentClst][parentSibl])

        self.paths.append(path[::-1])

    def _print_stopCriterion(self, mNodes, mGnrts):
        """ print out the stoping criteria based on setting of maximal number of 
        nodes and maximal number of generations.
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
        """
        print("\nnow expanding: generation:(%d); clst:(%d); sibl:(%d); state:(%s)"
                %(current['gnrt'], current['clst'], current['sibl'], current['state']))

        print("total number of nodes : %d, \ntotal number of layers : %d,\
            \nnumber of paths found : %d" %(self.n_nodes, self.n_gnrt, len(self.paths)))
                


    def print_paths(self):
        """ print out the information extracted from paths in format:
        [path_0, path_1, ...]
        path_# = [{'action':action_0, 'state':state_0, 'pathCost':pathCost_0}, 
                  {'action':action_1, 'state':state_1, 'pathCost':pathCost_1}, 
                  ...]
        """
        pathsInfo = []
        for path in self.paths:
            pathInfo = []
            for node in path:
                nodeInfo = {'action':node['prevAction'], 
                            'state':node['state'], 
                            'pathCost':node['pathCost']}
                pathInfo.append(nodeInfo)
            pathsInfo.append(pathInfo)

        print('\n%d paths found:' % len(pathsInfo))
        pprint(pathsInfo)

    def print_tree(self):
        """ print out the fully grown tree
        """
        print('\nTree searched:')
        pprint(self.tree)
        
    def plot_tree(self, diameter=50, background='dark', title='search tree', 
                  ls='-', a=0.8):
        """ plot out the search tree in a polar fig.
        """
        params={'figDiameter':diameter, 'background':background}
        plot = visual.SinglePolarPlot(**params)
        plot.tree(self.tree, self.paths, title=title, ls=ls, a=a)

    def export(self):
        """ export tree nodes and paths as JSON files.
        """
        with open('tree.json', 'w') as outfile:
            json.dump(self.tree, outfile)
        with open('paths.json', 'w') as outfile:
            json.dump(self.paths, outfile)

    #---------------------------SEARCHING STRATEGIES----------------------------
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
        """ perform Uniform Cost Search using priority queue as fringe and path 
        costs as weight in fringe.
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
            self._reinit()

    # informed search
    def bestFirstSearch(self, maxNodes=np.inf, maxLayers=np.inf):
        """ perform best first search using priority queue as fringe and heuristic 
        cost as weight in fringe.
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

    



    
    