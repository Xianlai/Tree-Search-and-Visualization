#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Thur Sep 21 08:07:00 2017
This class implements polar plot of tree generated by TreeSearch class.
@author: LAI
"""
#### 

from matplotlib import pyplot as plt
from copy import deepcopy
import math

sm_font = {'fontsize' : 13, 
           'fontname' : 'Arial'}

md_font = {'fontsize' : 17, 
           'fontweight' : 'bold', 
           'fontname' : 'Arial'}

lg_font = {'fontsize' : 25, 
            'fontweight' : 'bold', 
            'fontname' : 'Arial'}
        
grey = {'light':'#efefef', 
        'median':'#aaaaaa', 
        'dark':'#282828'}

discColor = {'blue':'#448afc', 
             'red':'#ed6a6a', 
             'green':'#80f442'}

cm = plt.get_cmap('RdYlGn')

class BasePlot():
    
    def __init__(self, background='light'):
        """ set the background color for fig and axes
        """
        self.bgc = grey[background]
        if self.bgc == grey['dark']: self.c = grey['light']
        else: self.c = grey['dark']
        plt.rcParams['axes.facecolor'] = self.bgc
        plt.rcParams['figure.facecolor'] = self.bgc

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
        ax.set_title(title, color=grey['light'], **lg_font)

    def show(self):
        plt.show()

    #--------------------------------PARSE TREE---------------------------------
    def _flatten(self, nestedList):
        """ given a list, possibly nested to any level, return it flattened.
        """
        flatten = []
        for item in nestedList:
            if type(item) == list: flatten.extend(self._flatten(item))
            else: flatten.append(item)
        return flatten

    def _isPath(self, node):
        """ test whether the edge between this node and its parent are on the path
        """
        for pathNode in self.pathNodes: 
            if node == pathNode: return True
        return False

    def _extract_values(self, nested, key):
        """ extract the values of specified key for all dictionaries in nested list.
        """
        values = [node[key] for node in self._flatten(nested)]
        return values

    def _find_childrenCenter(self, node):
        """ given a node, find the center of children cluster of this node
        """
        if node['children'] == None: return -999
        else:
            children = self.parsedTree[node['gnrt']+1][node['children']]
            leftEnd, rightEnd = children[0]['x'], children[-1]['x']
            return (leftEnd + rightEnd)/2

    def _compare_shiftCenter(self, sibling, shift):
            """ compare the shifted x value and the x value of center of children
            cluster, assign the larger one as the new x value of sibling and return
            the shifted amount.
            """
            center = self._find_childrenCenter(sibling)
            original = deepcopy(sibling['x'])
            shifted = original + shift
            sibling['x'] = max(center, shifted)
            shift = sibling['x'] - original
            return shift

    def _adjust_backward(self, gnrt, clst, parent, shift, c_clst_a, c_clst_e,):
        """ if current expanding cluster is shifted, we need to adjust backward
        to update the nodes being affected. Assuming the parent, parent of parent
        and so forth of current cluster are called direct relatives. All the nodes 
        affected are the ones following direct relative in each generation. 
        """
        # find the clst and sibl of direct relatives in each generation
        dirClst = [parent[0]]
        dirSibl = [parent[1]]
        for generation in self.tree[gnrt-1:1:-1]:
            prnt = generation[dirClst[-1]][dirSibl[-1]]['parent']
            dirClst.append(prnt[0])
            dirSibl.append(prnt[1])

        # update generation currGnrt-1, in this generation, all nodes following
        # and including parent of current cluster simply shift by overlap amount.
        # in the direct cluster
        dir_parent = self.parsedTree[-2][dirClst[0]][dirSibl[0]]
        shift = (c_clst_a + c_clst_e)/2 - dir_parent['x']
        dir_parent['x'] += shift

        for sibling in self.parsedTree[-2][dirClst[0]][dirSibl[0]+1:]:
            sibling['x'] += shift

        # in the following cluster
        for cluster in self.parsedTree[-2][dirClst[0]+1:]:
            for sibling in cluster:
                sibling['x'] += shift

        # update following generations backward except root (currGnrt-2:0:-1).
        for i, generation in enumerate(self.parsedTree[gnrt-2:0:-1], start=1):
            # every direct relative should be on the center of their children nodes.
            dir_parent = generation[dirClst[i]][dirSibl[i]]
            original = deepcopy(dir_parent['x'])
            dir_parent['x'] = self._find_childrenCenter(dir_parent)
            shift = dir_parent['x'] - original

            # in the direct cluster following direct relative, we compare the x 
            # value after shift and the x value of center of their children nodes.
            # the real x value should be the bigger one.
            for sibling in generation[dirClst[i]][dirSibl[i]+1:]:
                shift = self._compare_shiftCenter(sibling, shift)

            # in the following cluster
            for cluster in generation[dirClst[i]+1:]:
                for sibling in cluster:
                    shift = self._compare_shiftCenter(sibling, shift)

    def _parse_clstX(self, cluster, l_clst_e):
        """ 
        In order to plot a nice looking tree, we have to parse the x value of 
        each node so that:
        - Parent should sit on the center line of children cluster
        - Nodes belong to same cluster should stick together
        - Neighboring clusters should be separated by a one-unit gap.

        this function takes in current cluster and the ending x value of last 
        cluster, parse and calculate the starting and ending x values of current 
        cluster based on its neighbors and parent's x value.
        """
        node = cluster[0]
        gnrt, clst, parent = node['gnrt'], node['clst'], node['parent']
        clstSize = len(cluster) - 1

        # if currently at root, x is just 0
        if gnrt == 0: clst_a, clst_e = 0, 0

        # else, every cluster has parent, its center equals to parent's x
        else:
            parentX = self.parsedTree[gnrt-1][parent[0]][parent[1]]['x']
            clst_a = parentX - clstSize*self.hDist[gnrt]/2
            clst_e = parentX + clstSize*self.hDist[gnrt]/2

            # but we still need to make sure the current cluster is not overlapping
            # with last cluster in the same generation. If current cluster is the 
            # first one in current generation, there won't be overlaping. Otherwise, 
            # we calculate overlapped space and shift the current cluster if
            # overlap > 0 and adjust the x values of all nodes affected by this
            # shift in the previous generations.
            if clst != 0:
                shift = (l_clst_e + 2*self.hDist[gnrt]) - clst_a
                if shift > 0:
                    clst_a += shift; clst_e += shift
                    self._adjust_backward(gnrt, clst, parent, shift, clst_a, clst_e)

        return clst_a, clst_e

    def _parse_tree(self):
        """ parse the tree and return result
        parsedTree = [gnrt_0, gnrt_1, ...]
        gnrt_# = [cluster_0, cluster_1, ...]
        cluster_# = [sibling_0, sibling_1, ...]
        sibling_# = {'x':x value of this node in plot, 
                     'y':y value of this node in plot which is the level of this node,
                     'z':color value of this node which is the path cost up to this node,
                     'nodeText': the state of this node,
                     'parent': the peerIdx of parent
                     'edgeText':the name of edge which is the previous action,
                     'isPath': if this edge is part of path, 2xlinewidth}
        """
        pathCosts = self._extract_values(self.tree, 'pathCost')
        maxZ = max(pathCosts)
        clst_e = -2 # default end of cluster

        self.parsedTree = []
        # start parsing from the root
        for generation in self.tree: 
            self.parsedTree.append([]) # append an empty list as current generation

            for cluster in generation:
                clst_a, clst_e = self._parse_clstX(cluster, clst_e)
                self.parsedTree[-1].append([]) # append an empty list as current cluster

                for sibling in cluster:
                    gnrt, sibl = sibling['gnrt'], sibling['sibl']
                    parsedNode = {'x':clst_a + sibl * self.hDist[gnrt],
                                  'y':self.vDist[gnrt], # y is its layer
                                  'z':sibling['pathCost']/maxZ, # normalized pathCost
                                  'nodeText':sibling['state'],
                                  'children':sibling['children'],
                                  'parent':sibling['parent'],
                                  'edgeText':sibling['prevAction'],
                                  'isPath':self._isPath(sibling),
                                  'clst':sibling['clst'],
                                  'gnrt':sibling['gnrt'],
                                  'sibl':sibling['sibl']}

                    self.parsedTree[-1][-1].append(parsedNode)
        
        # normalize tree so it's within 2*pi
        self._normalize_tree()

    def _normalize_tree(self, ):
        """
        """
        X = self._extract_values(self.parsedTree, 'x')
        minX, maxX = min(X), max(X)
        if maxX >= 5/6 * math.pi:
            scale = 2*math.pi / (maxX - minX + 0.05)
            for generation in self.parsedTree:
                for cluster in generation:
                    for sibling in cluster:
                        sibling['x'] *= scale


    def _plot_edge(self, sibling, gnrt, ls, c_edge, a):
        """ plot one edge
        """
        parentClst = sibling['parent'][0]
        parentSibl = sibling['parent'][1]
        parent = self.parsedTree[gnrt-1][parentClst][parentSibl]
        edgeXs = (sibling['x'], parent['x'])
        edgeYs = (sibling['y'], parent['y'])

        if sibling['isPath'] == True:  c_edge='#d4f442'
        self.axes.plot(edgeXs, edgeYs, c=c_edge, ls=ls, alpha=a, lw=1)

    def _plot_node(self, sibling, ls, c_text, a):
        """ plot one node and its text
        """
        c_node = cm(1-sibling['z'])
        if sibling['nodeText'] == self.goal: 
            c_text = '#80f442'

        r_text = math.degrees(sibling['x'])
        if (r_text > 90) and (r_text < 270): r_text = r_text - 180


        x_text, content = sibling['x'], sibling['nodeText']
        y_text = sibling['y'] + 800 #self.vDist[sibling['gnrt']]*0.1
        param = {'color':c_text, 'alpha':a, 'rotation':r_text, 
                 'ha':'center', 'va':'center'}
        
        self.axes.plot(sibling['x'], sibling['y'], 'o', c=c_node, alpha=a)
        self.axes.text(x_text, y_text, content, **param)

    def _plot_tree(self, ls, c, a):
        """
        """
        for gnrt in range(len(self.parsedTree)):
            for cluster in self.parsedTree[gnrt]:
                for sibling in cluster:
                    self._plot_node(sibling, ls=ls, c_text=c, a=a)
                    if gnrt > 0:
                         self._plot_edge(sibling, gnrt, ls=ls, c_edge=c, a=a)



class SinglePolarPlot(BasePlot):

    def __init__(self, figDiameter=20, background='light'):
        """ initialize the single plot object
        """
        BasePlot.__init__(self, background=background)
        self.fig, self.axes = self._plot_base(figDiameter)
        self.figR = figDiameter/2
        

    def _plot_base(self, diameter):
        """ set up the fig and axes for single plot
        """
        fig = plt.figure(figsize=(diameter, diameter))
        ax = fig.add_subplot(111, projection='polar')
        self._set_polarAxParam(ax)
        return fig, ax

    def tree(self, tree, paths, title='search tree', ls='-', a=0.5, show=True):
        """ plot the tree in polar projection with initState, goalState and 
        optimal path marked.
        """
        self.tree = tree
        self.goal = paths[0][-1]['state']
        self.pathNodes = self._flatten(paths)

        # set up spacing:
        vDists = [x**1.5 for x in range(len(self.tree))]
        self.vDist = [self.figR * 1500 * x / max(vDists) for x in vDists]
        self.hDist = [0] + [600/radius for radius in self.vDist[1:]]

        # parse the tree and plot
        self._parse_tree()
        self._plot_tree(ls=ls, c=self.c, a=a)

        # plot notations
        self._set_axTitle(self.axes, title)

        if show: plt.show()
