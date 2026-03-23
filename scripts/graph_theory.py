import networkx as nx
import matplotlib.pyplot as plt

nodes = list(range(10))
edges = [(1,0), (2,1), (3,2), (4,1), (5,0),
         (0,5), (6,3), (7,3), (3,0), (8,0), (9,8)]

gx = nx.DiGraph()
gx.add_nodes_from(nodes)
