import networkx as nx
import matplotlib.pyplot as plt

# Create a directed graph
G = nx.DiGraph()

# Add edges
G.add_edges_from([
    ('X', 'Y'), ('Y', 'Z'), ('Z', 'X'), ('X', 'W'),
    ('W', 'Z'), ('Y', 'W'), ('Z', 'V'), ('V', 'Y'), 
    ('W', 'V')
])

# Draw the graph
plt.figure(figsize=(10, 10))
nx.draw_networkx(G, with_labels=True)
plt.title("Directed Graph with New Values")
plt.show()

# Calculate HITS scores
hubs, authorities = nx.hits(G, max_iter=50, normalized=True)

# Print Hub and Authority Scores
print("Hub Scores:", hubs)
print("Authority Scores:", authorities)