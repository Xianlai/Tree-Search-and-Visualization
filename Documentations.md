# Documentation for Modules in this project

-----
## TreeSearch class     
**class** ```TreeSearch(initState=None, show_progress=False)```     

#### Parameters:     
- **initState**: 
        the initial state as the root of searching tree.    

- **show_process**:   
        a boolean value. If true, the algorithm will print the intermediate search process on the screen.     
  
#### Attributes:     
- **``self.initState``**: 
    The initial state    

- **``self.n_nodes``**: 
    The number of nodes in the search tree    

- **``self.n_gnrt``**: 
    The number of generations/levels in the search tree.  

- **``self.show``**: 
    Whether to show progress when searching.  

- **``self.root``**: 
    The root node of search tree.  

- **``self.searchType``**: 
    The search strategy you choose.  

- **``self.tree``**: 
    The whole search tree as a nested list of dictionaries.  

- **``self.paths``**: 
    All the paths found as a nested list of dictionaries.  


#### Methods:  
```python
- self.breadthFirstSearch(maxNodes=np.inf, maxLayers=np.inf)
- self.depthFirstSearch**(maxNodes=np.inf, maxLayers=np.inf)
- self.uniformCostSearch**(maxNodes=np.inf, maxLayers=np.inf)
- self.iterativeDeepeningSearch**(maxDepth=5)
- self.bestFirstSearch**(maxNodes=np.inf, maxLayers=np.inf)
- self.aStarSearch**(maxNodes=np.inf, maxLayers=np.inf) 
- self.print_paths**()  
- self.plot_tree**(diameter=10, background='dark', title='search tree', ls='-', a=0.8)  
- self.export**()  
```

**``__init__(initState=None, show_progress=False)``**  
    
**``breadthFirstSearch(maxNodes=np.inf, maxLayers=np.inf)``**
    The breadth first search that always push and expand the first node saved in the frontier list.  

    ----Parameters----: 
    - maxNodes: 
        The maximum nodes to be explored before stopping the searching.  
    - maxLayers: 
        The maximum layers to be explored before stopping the searching.  
        If neither of these is set, then the search will stop when the first path is found.  


**``depthFirstSearch(maxNodes=np.inf, maxLayers=np.inf)``**    
    The depth first search that always push and expand the last node saved in the frontier list.

    *----Parameters----*:  
    - maxNodes: 
        The maximum nodes to be explored before stopping the searching.  
    - maxLayers: 
        The maximum layers to be explored before stopping the searching.     
        If neither of these is set, then the search will stop when the first path is found.  


**``uniformCostSearch(maxNodes=np.inf, maxLayers=np.inf)``**    
    The uniform cost search that always push and expand the node with least path cost in the frontier list.

    *----Parameters----*:  
    - maxNodes: 
        The maximum nodes to be explored before stopping the searching.  
    - maxLayers: 
        The maximum layers to be explored before stopping the searching.  
        If neither of these is set, then the search will stop when the first path is found.  


**``iterativeDeepeningSearch(maxDepth=5)``**    
    The iterative deepening search that iteratively performs DFS with maximum layers increasing. The search will stop when it finds the first goal or finished the last iteration with maxLayer equals maxDepth.  

    *----Parameters----*:  
    - maxDepth: 
        The maximum layers of last iteration of DFS.  


**``bestFirstSearch(maxNodes=np.inf, maxLayers=np.inf)``**    
    The best first search that always push and expand the node with least heuristic in the frontier list.  

    *Parameters*:  
    - maxNodes: 
        The maximum nodes to be explored before stopping the searching.  
    - maxLayers: 
        The maximum layers to be explored before stopping the searching.

    If neither of these is set, then the search will stop when the first path is found.  


**``aStarSearch(maxNodes=np.inf, maxLayers=np.inf)``**    
    The A* search that uses the sum of path cost and heuristic to determine which node in the frontier to push and expand.  

    *----Parameters----*:  
    - maxNodes: 
        The maximum nodes to be explored before stopping the searching.  
    - maxLayers: 
        The maximum layers to be explored before stopping the searching.

    If neither of these is set, then the search will stop when the first path is found.  


**``print_paths()``**    
    Print out the the paths found.  


**``plot_tree(diameter=50, background='dark', ls='-', a=0.8, title='Search Tree')``**    
    Plot out the search tree in a polar figure.  

    *----Parameters----*:  
    - diameter: 
        The nodes of tree has to spread out in the plot without overlapping, so the bigger diameter, more sparse the nodes are.  
    - background: 
        The background color of fig, could be either 'light' or 'dark'.  
    - ls: 
        line style.  
    - a: 
        alpha of nodes and edges.  
    - title: 
        The title of this plot.  


**``export()``**     
    Write tree and paths as JSON files into current directory.  


-----
## RoadtripProblem class  
\*This class is the child class of TreeSearch class. So it has all the attributes as well as methods shown above.  

**class** `SpecificProblem(states=None, initState=None, goalState=None, stepCosts=None, heuristics=None, show_progress=False)` 

#### Parameters:  
- **initState**: 
    The initial state as the root 

- **goalState**: 
    The goal state  

- **states**: 
    All the states in the problem environment  

- **stepCosts**: 
    The step costs from a state to another state  

- **heuristics**: 
    The heuristics from a state to goal state  

- **show_process**: 
    A boolean value. If true, the algorithm will print the intermediate search process on the screen.  
  

#### Attributes:  
- **``self.states``**: 
    All the states in the problem environment  

- **``self.size``**: 
    The number of all possible states  

- **``self.goalState``**: 
    The goal state.  

- **``self.heuristics``**: 
    The heuristics of each state  

- **``self.encoding``**: 
    The encoding from states to integers  

- **``self.decoding``**: 
    The decoding from integers to states.  

- **``self.stepCosts``**: 
    The step cost between each pair of states.   

#### Methods: 
```python
- self.print_encoding()
```

**``__init__(states=None, initState=None, goalState=None, stepCosts=None, heuristics=None, show_progress=False)``**  

**``self.print_encoding()``**      
    Print out the encodings from states to integers.  


-----
##TreeVisual Class

**class** `PolarPlot(diameter=20, background='dark')`   

#### Parameters:  
- **diameter**: 
    The diameter of the polar fig. 

- **background**: 
    The background color.  


#### Attributes:  
- **``self.states``**: 
    All the states in the problem environment  

- **``self.bgc``**: 
    The background color of self.fig  

- **``self.c``**: 
    The default color for edges and labels  

- **``self.cm``**: 
    The color map for nodes' color  

- **``self.green``**: 
    The color green used to mark path and goal node labels  
 
- **``self.fig``**: 
    The figure object of plotting.  

- **``self.ax``**: 
    The ax object of plotting.  

- **``self.radius``**: 
    The radius of polar fig.   

- **``self.tree``**: 
    The search tree to be parsed and plot.   

- **``self.pathNodes``**: 
    All the nodes on the path as a flat list.  

- **``self.goal``**: 
    The goal state.  

- **``self.vDist``**: 
    The unit distance in radius direction  

- **``self.hDist``**: 
    The unit distance in the tangent direction.  

- **``self.parsedTree``**: 
    The parsed tree.  

#### Methods: 
```python
- self.show()   
- self.save()  
- self.plot_tree(tree, paths, title='search tree', ls='-', a=0.5, show=True)  
```

**``__init__(diameter=20, background='dark')``**  


**``show()``**    
    A static method to be used outside class to show plots.  


**``save(path='search_tree.png')``**    
    Save the fig to file. You can directly specify the format in the file name. And you can specify the dpi and bbox_inches.  

    *----Parameters----*:  
    - path: The path you want to save the file to.  


**``plot_tree(tree, paths, title='search tree', ls='-', a=0.5, show=True)``**  
    Plot the tree in polar projection with correct spacing and mark the initial state, goal state and path edges in green and color every node by their path costs.  

    *----Parameters----*:  
    - tree: The search tree in form described in TreeSearch.py  
    - paths: The collection of paths found.  
    - title: title of the fig.  
    - ls: line style  
    - a: alpha  
    - show: whether to show the plot.  




