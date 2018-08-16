#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
This is the module for search tree visualization

Author: Xian Lai
Date: Sep.14, 2017
"""

from matplotlib import pyplot as plt
from copy import deepcopy
import math

sm_font   = {'fontsize':13, 'fontname':'Arial'}
md_font   = {'fontsize':17, 'fontname':'Arial', 'fontweight':'bold'}
lg_font   = {'fontsize':25, 'fontname':'Arial', 'fontweight':'bold'}

grey      = {'light':'#efefef', 'median':'#aaaaaa', 'dark':'#282828'}
discColor = {'blue':'#448afc', 'red':'#ed6a6a', 'green':'#80f442'}
cm        = {'light':plt.get_cmap('RdYlGn'), 'dark':plt.get_cmap('cool')}
green     = {'light':'#d4f442' , 'dark':'#2dbc32'}


class TreeVisual():
    
    """ This class implements the methods to parse the search tree to get plot
    data and the methods to plot the tree based on the attributes of its nodes
    like whether is goal node or whether is path. 
    """

    def __init__(self, diameter=20, background='dark'):
        """ set the fig size and background color.
        """
        # set up the color of background, edges, nodes and text
        if background == 'dark':
            self.bgc   = grey['dark']
            self.c     = grey['light']
            self.cm    = cm['light']
            self.green = green['light']
        else:
            self.bgc   = grey['light']
            self.c     = grey['dark']
            self.cm    = cm['dark']
            self.green = green['dark']
        plt.rcParams['axes.facecolor']   = self.bgc
        plt.rcParams['figure.facecolor'] = self.bgc

        self.fig    = plt.figure(figsize=(diameter, diameter))
        self.ax     = self.fig.add_subplot(111, projection='polar')
        self.radius = diameter/2
        self._set_polarAxParam(self.ax)


    @staticmethod
    def show():
        plt.show()


    def save(self, path='search_tree.png'):
        """ Save the fig to file. You can directly specify the format in the 
        file name. And you can specify the dpi and bbox_inches. 
        
        inputs:
        -------
        - path: The path you want to save the file to.
        """
        self.fig.savefig(path, dpi=None, bbox_inches='tight')


    def plot_tree(self, tree, paths, title='search tree', ls='-', a=0.5, 
            show=True):
        """ plot the tree in polar projection with correct spacing and:
        - initState marked in green text
        - goalState marked in green text
        - path leads to goal node marked in green
        - nodes colored by their pathcost

        inputs:
        -------
        - tree: The search tree in form described in TreeSearch.py
        - paths: The collection of paths found.
        - title: title of the fig.
        - ls: line style
        - a: alpha
        - show: whether to show the plot.
        """
        self.tree      = tree
        self.pathNodes = self._flatten(paths)
        if self.pathNodes: self.goal = paths[0][-1]['state']
        else: self.goal = "Not Found"

        # set up spacing:
        vDists     = [x**1.5 for x in range(len(self.tree))]
        self.vDist = [self.radius * 1500 * x / max(vDists) for x in vDists]
        self.hDist = [0] + [600/radius for radius in self.vDist[1:]]

        # parse the tree and plot
        self._parse_tree()

        for gnrt in range(len(self.parsedTree)):
            for cluster in self.parsedTree[gnrt]:
                for sibling in cluster:
                    self._plot_node(sibling, ls=ls, textColor=self.c, a=a)
                    if gnrt > 0:
                         self._plot_edge(
                            sibling, gnrt, ls=ls, edgeColor=self.c, a=a
                        )

        self._set_axTitle(self.ax, title)
        if show: plt.show()


    def _plot_edge(self, node, gnrtIdx, edgeColor, ls, a):
        """ plot the edge from given node to its parent node with given color, 
        line style, and alpha. If this edge is on path, then change the color
        to green.

        inputs:
        -------
        - node: The given node which is the children end of edge
        - gnrtIdx: The generation index of given node
        - edgeColor: the color of edge
        - ls: line style
        - a: alpha
        """
        paClstIdx = node['parent'][0] # cluster index of parent node
        paSiblIdx = node['parent'][1] # sibling index of parent node
        parent    = self.parsedTree[gnrtIdx-1][paClstIdx][paSiblIdx]
        edgeXs    = (node['x'], parent['x'])
        edgeYs    = (node['y'], parent['y'])

        if node['isPath'] == True:  edgeColor=self.green
        self.ax.plot(edgeXs, edgeYs, c=edgeColor, ls=ls, alpha=a, lw=1)


    def _plot_node(self, node, textColor, ls, a):
        """ plot the given node colored by its path cost. And plot its text 
        label with default text color and alpha. If this node is goal node, 
        plot the text in green.

        inputs:
        -------
        - node: The node to be plotted.
        - textColor: the default color of text
        - ls: line style
        - a: alpha
        """
        nodeColor = self.cm(1-node['z'])
        if node['nodeText'] == self.goal: textColor = self.green

        # if the text in on the left side of the plot, it's upside down. So  
        # rotate it by 180 degree
        textRotation = math.degrees(node['x'])
        if (textRotation > 90) and (textRotation < 270): 
            textRotation = textRotation - 180

        x_text  = node['x']
        y_text  = node['y'] + 800 # offset labels outward away from node
        content = node['nodeText']
        param   = {
            'color':textColor, 'rotation':textRotation, 'alpha':a, 
            'ha':'center', 'va':'center'
        }
        
        self.ax.plot(node['x'], node['y'], 'o', c=nodeColor, alpha=a)
        self.ax.text(x_text, y_text, content, **param)


    def _set_polarAxParam(self, ax):
        """ set the parameters of polar axes: turn off the axes ticks, spines
        and grid.
        """
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.spines["polar"].set_visible(False)
        ax.grid(False)
        

    def _tight_layout(self):
        self.fig.tight_layout()


    def _set_figTitle(self, title):
        self.fig.suptitle(title, c=grey['light'], **lg_font)  


    def _set_axTitle(self, ax, title):
        if self.radius < 12: font = md_font
        else:                font = lg_font
        ax.set_title(title, color=self.c, **font)


    #----------------------------- PARSE TREE --------------------------------
    def _parse_tree(self):
        """ To plot the nodes, edges and corresponding text labels of search 
        tree, we need to parse the tree data and extract information like the
        x, y values of each node, their parent node etc.

        The parsed result will in form of:
        parsedTree = [gnrt_0, gnrt_1, ...]
        gnrt_#     = [clst_0, clst_1, ...]
        clst_#     = [sibl_0, sibl_1, ...]
        sibl_#     = {
            'x' : x value of this node in plot, 
            'y' : y value of this node in plot which is the level of this node
            'z' : color of this node which is the path cost up to this node
            'nodeText': the state of this node
            'parent'  : the peerIdx of parent
            'edgeText': the name of edge which is the previous action
            'isPath'  : if this edge is part of path, 2xlinewidth
        }
        """
        pathCosts = self._extract_values(self.tree, 'pathCost')
        maxZ = max(pathCosts)
        clstEndX = -2 # default end of cluster

        self.parsedTree = [] # initialize the parsedTree

        # start parsing from the root layer by layer
        for gnrt in self.tree:
            # append an empty list as current generation
            self.parsedTree.append([]) 
            for clst in gnrt:
                # append an empty list to the current layer as current cluster
                self.parsedTree[-1].append([]) 
                # fix the starting and ending x values of current cluster.
                clstStartX, clstEndX = self._parse_clstX(clst, clstEndX)
                # adding desired information to each node in the cluster.
                for sibl in clst:
                    gnrtIdx, siblIdx = sibl['gnrt'], sibl['sibl']
                    parsedNode = {
                        'x':clstStartX + siblIdx * self.hDist[gnrtIdx],
                        'y':self.vDist[gnrtIdx], # y is its layer
                        'z':sibl['pathCost']/maxZ, # normalized pathCost
                        'nodeText':sibl['state'],
                        'children':sibl['children'],
                        'parent':sibl['parent'],
                        'edgeText':sibl['prevAction'],
                        'isPath':self._isPath(sibl),
                        'clst':sibl['clst'],
                        'gnrt':sibl['gnrt'],
                        'sibl':sibl['sibl']
                    }
                    self.parsedTree[-1][-1].append(parsedNode)

        # scale the x values of tree so it's within 2*pi
        self._normalize_tree()

        
    def _parse_clstX(self, clst, l_clstEndX):
        """ Calculate the starting and ending x values of given cluster. But  
        in order to plot a nice looking tree, we can't simply use the order of 
        nodes' cluster indices and sibling indices as their x values. Instead,  
        we have to add correct amount of spaces between nodes and clusters so:

            - All the parent nodes will be on the center of their childrens.
            - Neighboring clusters will be separated by a one-unit gap.
            - Nodes belong to same cluster will stay together.

        So we need to consider the x values of given cluster's neighbors and 
        parent.

        inputs:
        -------
        - cluster: The current cluster to calculate x values.
        - l_clstEndX: The ending x value of last cluster in the same layer.

        output:
        -------
        - clstStartX: The starting x value of current cluster.
        - clstEndX: The ending x value of current cluster.
        """
        node = clst[0]
        gnrtIdx, clstIdx, paIdx = node['gnrt'], node['clst'], node['parent']
        clstSize = len(clst) - 1

        # if currently at root, both clstStartX and clstEndX are just 0
        if gnrtIdx == 0: clstStartX, clstEndX = 0, 0

        # else, every cluster has a parent, the center of cluster equals to 
        # parent's x value. And the starting and ending x values are fixed.
        else:
            parentX    = self.parsedTree[gnrtIdx-1][paIdx[0]][paIdx[1]]['x']
            clstStartX = parentX - clstSize*self.hDist[gnrtIdx]/2
            clstEndX   = parentX + clstSize*self.hDist[gnrtIdx]/2

            # but we still need to make sure the current cluster is not over-
            # lapping with last cluster in the same generation. If current 
            # cluster is the first one in current generation, there won't be 
            # overlaping. Otherwise, we should check whether overlapping exist
            # by comparing l_clstEndX + 2*unit with clstStartX(we add 2 units 
            # here instead of 1 is because we want to leave the 1 unit gap 
            # between clusters). If clstStartX is greater then we are safe, 
            # otherwise we should shift the cluster onward by overlapping 
            # amount and adjust the x values of all nodes affected by this 
            # shift in the previous generations(because we still want to keep 
            # parent on the center of children cluster).
            if clstIdx != 0:
                shift = (l_clstEndX + 2*self.hDist[gnrtIdx]) - clstStartX
                if shift > 0:
                    clstStartX += shift; clstEndX += shift
                    self._adjust_upward(
                        gnrtIdx, clstIdx, paIdx, clstStartX, clstEndX
                    )

        return clstStartX, clstEndX


    def _adjust_upward(self, gnrtIdx, clstIdx, paIdx, clstStartX, clstEndX):
        """ if current expanding cluster is shifted, we need to adjust upward
        to update all the nodes being affected. 

        Assuming the parent, parent of parent and so forth of current cluster 
        are called direct relatives. Only the nodes following(has bigger 
        clstIdx) direct relatives in each higher generations are affected by 
        the shift.

        inputs:
        -------
        - gnrtIdx: The generation index of current cluster
        - clstIdx: The cluster index of current cluster
        - paIdx: [parent's cluster index, parent's sibling index]
        - clstStartX: the starting x value of shifted current cluster
        - clstEndX: the ending x value of shifted current cluster
        """
        # find the clstIdx and siblIdx of direct relatives in each generation
        dirClst = [paIdx[0]] # cluster indices of direct relatives
        dirSibl = [paIdx[1]] # sibling indices of direct relatives
        for generation in self.tree[gnrtIdx-1:1:-1]:
            prnt = generation[dirClst[-1]][dirSibl[-1]]['parent']
            dirClst.append(prnt[0])
            dirSibl.append(prnt[1])

        # update generation gnrtIdx - 1, in this generation, all nodes follow-
        # ing and including parent of current cluster simply should shift by 
        # the amount of (clstStartX + clstEndX)/2 - dir_parent['x'].
        dir_parent = self.parsedTree[-2][dirClst[0]][dirSibl[0]]
        shift      = (clstStartX + clstEndX)/2 - dir_parent['x']

        # update nodes following parent in the parent's cluster
        for sibl in self.parsedTree[-2][dirClst[0]][dirSibl[0]:]:
            sibl['x'] += shift

        # update nodes in the following cluster
        for clst in self.parsedTree[-2][dirClst[0]+1:]:
            for sibl in clst:
                sibl['x'] += shift

        # update all generations upward except root, from gnrtIdx - 2 to 1.
        # because this is polar plot, the root node is always in the center.
        for i, gnrt in enumerate(self.parsedTree[gnrtIdx-2:0:-1], start=1):

            # For the direct relative in each generation, it should be shift 
            # onto the center of their children nodes. And we keep record of 
            # the shift distance.
            dir_relative      = gnrt[dirClst[i]][dirSibl[i]]
            original          = deepcopy(dir_relative['x'])
            dir_relative['x'] = self._find_childrenCenter(dir_relative)
            shift             = dir_relative['x'] - original

            # For the nodes in the direct relative's cluster following direct 
            # relative, we compare the x value after shift and the x value of
            # the center of their children nodes. The correct x value should 
            # be the greater one. And we keep track of the shift distance 
            # because it will affect the following siblings and clusters.
            for sibl in gnrt[dirClst[i]][dirSibl[i]+1:]:
                shift = self._compare_shiftCenter(sibl, shift)

            # For the nodes in the following clusters
            for clst in gnrt[dirClst[i]+1:]:
                for sibl in clst:
                    shift = self._compare_shiftCenter(sibl, shift)


    def _compare_shiftCenter(self, node, shift):
        """ compare the shifted distance and the center of children cluster of
        given node, assign the larger one as the new x value of given node and 
        return the shifted amount.
        """
        center    = self._find_childrenCenter(node)
        original  = deepcopy(node['x'])
        shifted   = original + shift
        node['x'] = max(center, shifted)
        shift     = node['x'] - original
        return shift


    def _find_childrenCenter(self, node):
        """ Find the center of children cluster of gievn node if it has child-
        ren nodes.
        """
        # if it has no children, then return a negative value
        if node['children'] == None: return -999
        # else, find the children cluster, take the average of x values of 
        # this cluster's first and last nodes.
        else:
            children   = self.parsedTree[node['gnrt']+1][node['children']]
            clstStartX = children[0]['x']
            clstEndX   = children[-1]['x']
            return (clstStartX + clstEndX)/2


    def _normalize_tree(self):
        """ scale the x values of tree if the maximal x value of any layer 
        exceed 5/6 pi. 
        """
        X = self._extract_values(self.parsedTree, 'x')
        minX, maxX = min(X), max(X)
        if maxX >= 5/6 * math.pi:
            scale = 2*math.pi / (maxX - minX + 0.05)
            for gnrt in self.parsedTree:
                for clst in gnrt:
                    for sibl in clst:
                        sibl['x'] *= scale


    def _flatten(self, nestedList):
        """ given a list, possibly nested to any level, return it flattened.
        """
        flatten = []
        for item in nestedList:
            # if any element is a list, recursively apply this function.
            if type(item) == list: flatten.extend(self._flatten(item))
            else: flatten.append(item)
        return flatten


    def _isPath(self, node):
        """ check whether the given node is on the path to the goal node. To 
        do this, we simply check whether this node is in the path collection.
        """
        for pathNode in self.pathNodes: 
            if node == pathNode: return True
        return False


    def _extract_values(self, nested, key):
        """ extract the values of specified key from all dictionaries in a 
        nested list. We use this function to extract values of certain 
        attributes of all nodes in a tree or generation.

        inputs:
        -------
        - nested: A nested list of dictionaries. 
        - key: The key of interest.
        """
        values = [node[key] for node in self._flatten(nested)]
        return values


