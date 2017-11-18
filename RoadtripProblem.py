#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
This is the module of example problem to show the functions of its parent 
class--TreeSearch class.

Author: Xian Lai
Date: Sep.14, 2017
"""
from TreeSearch import TreeSearch
import numpy as np

# collect the useful information from the problem and the maps including:
#   - Environment: Romania
#   - States: All the cities in the map
#   - Step costs: Distances between cities
#   - Actions: The agent can choose to go to any one of the cities directly 
#     connected to current city in one step
#   - Heuristics: The straight line distance between each city and Bucharest
cities = sorted([
    'Rimnicu Vilcea','Sibiu','Craiova','Fagaras',
    'Pitesti', 'Bucharest','Oradea','Zerind','Arad',
    'Timisoara','Lugoj','Mehadia','Drobeta','Giurgiu',
    'Urziceni','Hirsova','Eforie','Vaslui','Iasi','Neamt'
])

distances = [
    ['Vaslui', 'Iasi', 92],
    ['Iasi', 'Neamt', 87],
    ['Sibiu', 'Fagaras', 99],
    ['Sibiu', 'Rimnicu Vilcea', 80],
    ['Craiova', 'Rimnicu Vilcea', 146],
    ['Pitesti', 'Rimnicu Vilcea', 97],
    ['Craiova', 'Pitesti', 138],
    ['Pitesti', 'Bucharest', 101],
    ['Bucharest', 'Fagaras', 211],
    ['Mehadia', 'Drobeta', 75],
    ['Drobeta', 'Craiova', 120],
    ['Giurgiu', 'Bucharest', 90],
    ['Urziceni', 'Bucharest', 85],
    ['Urziceni', 'Vaslui', 142],
    ['Urziceni', 'Hirsova', 98],
    ['Hirsova', 'Eforie', 86],
    ['Oradea', 'Zerind', 71],
    ['Sibiu', 'Oradea', 151],
    ['Zerind', 'Arad', 75],
    ['Arad', 'Sibiu', 140],
    ['Arad', 'Timisoara', 118],
    ['Timisoara', 'Lugoj', 111],
    ['Lugoj', 'Mehadia', 211]]  

SLDs = {
    'Arad':366, 'Bucharest':0, 'Craiova':160, 'Drobeta':242,
    'Eforie':161, 'Fagaras':176, 'Giurgiu':77, 'Hirsova':151,
    'Iasi':226, 'Lugoj':244, 'Mehadia':241, 'Pitesti':100, 
    'Oradea':380, 'Neamt':234, 'Rimnicu Vilcea':193, 'Sibiu':253,
    'Timisoara':329, 'Urziceni':80, 'Vaslui':199, 'Zerind':374
}

class RoadtripProblem(TreeSearch):

    """ 
    This class implements the specific example problem of finding the best 
    route in Romania. 

    It needs to be the child class of TreeSearch class and needs to have
    following 3 methods to be used in tree searching:
    - self._transition(state)
        The transition model takes in state and return possible actions,  
        result states and corresponding step costs.
    - self._heuristicCost(state)
        Calculate and return the heuristic cost given a state.
    - self._isGoal(state)
        Check whether given state is goal state or one of goal states.

    All the other attributes and methods are flexible depends on the problem.
    """

    def __init__(self, initState=None, goalState=None, states=cities, 
            stepCosts=distances, heuristics=SLDs, show_progress=False):
        """ initiate the TreeSearch class and current class. 

        inputs:
        -------
        - initState: The starting city
        - goalState: The destination
        - states: All cities
        - stepCosts: The distances between city pairs
        - heuristics: The straight line distance from each city to destination
        - show_progress: whether to show progress when searching.
        """
        TreeSearch.__init__(
            self, initState=initState, show_progress=show_progress
        )
        self.states     = states
        self.size       = len(states)
        self.goalState  = goalState
        self.heuristics = heuristics
        self._coding()
        self._update_stepCosts(stepCosts)


    def _coding(self):
        """ encode the state with integers because we are using matrix to 
        record the step costs between each pair of cities. self.encoding will 
        translate state to code and self.decoding will translate code to state.
        """
        self.encoding = {}; self.decoding = {}
        for i in range(self.size):
            self.encoding.update({self.states[i]:i})
            self.decoding.update({i:self.states[i]})
          

    def _update_stepCosts(self, stepCosts):
        """ use adjacency matrix to represent the step costs. Each element of  
        matrix represent the distance between row city and column city. The 
        pair of cities without direct connect will have infinite distance.

        inputs:
        -------
        - stepCosts: The step cost between each pair of states. 
        """
        # we set up the matrix with all cities unconnected to each other.
        self.stepCosts = np.full((self.size, self.size), np.inf)
        
        # and then we change the value of diagnal elements to 0 because the 
        # corresponding row and column are the same city.
        np.fill_diagonal(self.stepCosts, 0)
        
        # and then we update the connected cities. The matrix is symmetric 
        # along diagonal.
        for dist in stepCosts:
            city_0 = self.encoding[dist[0]]
            city_1 = self.encoding[dist[1]]
            self.stepCosts[city_0, city_1] = dist[2]
            self.stepCosts[city_1, city_0] = dist[2]


    def print_encoding(self):
        """ print out the encodings
        """
        print('The encoding of states:')
        for i in range(self.size): print('%d : %s' % (i, self.states[i]))


    def _transition(self, state):
        """ the transition model that takes in a state, return the possible 
        actions, children states and corresponding step costs.

        inputs:
        -------
        - state: The state to be expanded.

        output:
        -------
        - actions: The actions possible to be applied in current state.
        - childStates: The successor states result from the actions.
        - stepcosts: The step cost of each action.
        """
        childStates, actions, stepCosts= [], [], []
        
        s = self.encoding[state]
        # iterate through all cities, if a city is connected to current city 
        # and it is not the current city itself, it is one of the successors.
        for i in range(self.size):
            if (self.stepCosts[s,i] != np.inf) & (self.stepCosts[s,i] != 0):
                childStates.append(self.decoding[i])
                actions.append("go " + self.decoding[i])
                stepCosts.append(self.stepCosts[s,i])

        return actions, childStates, stepCosts
    

    def _heuristicCost(self, state):
        """ return the heuristic cost of given state

        inputs:
        -------
        - state: The state to be calculated.
        """
        if self.heuristics == None: return 0
        else: return self.heuristics[state]
        

    def _isGoal(self, state):
        """ check whether given state is goal state.

        inputs:
        -------
        - state: The state to be checked.
        """
        if state == self.goalState: return True







