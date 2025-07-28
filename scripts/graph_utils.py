import matplotlib.pyplot as plt
import networkx as nx
import csrgraph as cg
import numpy as np
import random
import string


def get_graph(nodes = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]):

    G = nx.DiGraph()
    east_pairs = [(nodes[0], nodes[1]), (nodes[1], nodes[2]), (nodes[3], nodes[4]),
                  (nodes[4], nodes[5]), (nodes[6], nodes[7]), (nodes[7], nodes[8])]
    south_pairs = [(nodes[0], nodes[3]), (nodes[3], nodes[6]), (nodes[1], nodes[4]),
                   (nodes[4], nodes[7]), (nodes[2], nodes[5]), (nodes[5], nodes[8])]
    north_pairs = [(i[1], i[0]) for i in south_pairs]
    west_pairs = [(i[1], i[0]) for i in east_pairs]

    for n in nodes:
        G.add_node(n)

    for tple in east_pairs:
        G.add_edge(tple[0], tple[1], direction='EAST')
    for tple in north_pairs:
        G.add_edge(tple[0], tple[1], direction='NORTH')
    for tple in west_pairs:
        G.add_edge(tple[0], tple[1], direction='WEST')
    for tple in south_pairs:
        G.add_edge(tple[0], tple[1], direction='SOUTH')

    return G

def get_random_walks(G, n_walks=1):
    csr_G = cg.csrgraph(G, threads=12)
    node_names = csr_G.names
    walks = csr_G.random_walks(walklen=50, # length of the walks
                    epochs=n_walks,
                    # start_nodes=list(range(0, 9)),
                    return_weight=1.,
                    neighbor_weight=1.)
    walks = np.vectorize(lambda x: node_names[x])(walks)
    return walks

def generate_n_random_walks(G, n_walks, walk_length):
    walks = []
    nodes = list(G.nodes)
    
    for _ in range(n_walks):
        walk = []
        # Start from a random node
        current_node = random.choice(nodes)
        walk.append(current_node)
        
        while len(walk) < walk_length:
            neighbors = list(G.successors(current_node))
            if not neighbors:
                break  # If the current node has no out-edges, end the walk
            next_node = random.choice(neighbors)
            walk.append(next_node)
            current_node = next_node
        
        # Convert walk to a string describing the path
        walk_str = walk_to_string(walk, G)
        walks.append(walk_str)
        
    return walks


def walk_to_string(walk, G):
    walk_string = ""
    for i in range(len(walk)-1):
        node1 = walk[i]
        node2 = walk[i+1]
        direc = G.edges[(node1, node2)]['direction']
        walk_string += str(node1) + " "+ str(direc) + " "
    walk_string += walk[-1]
    return walk_string

def generate_name() -> str:
    return ''.join(random.choices(string.ascii_lowercase, k=2))

def get_walks_as_strings(n_graphs=1000, n_walks=10, walk_length=50):
    entities_for_graphs =[[generate_name() for j in range(9)] for i in range(n_graphs)]
    
    all_graphs = []
    walks_as_strings = []
    for nodes in entities_for_graphs:
        G = get_graph(nodes=nodes)
        walks = generate_n_random_walks(G, n_walks, walk_length)
        walks_as_strings.extend(walks)
        all_graphs.append(G)
    return walks_as_strings, all_graphs

def plot_path(input_string):
    directions = {'NORTH': (0, 1), 'EAST': (1, 0), 'SOUTH': (0, -1), 'WEST': (-1, 0)}
    steps = input_string.split(' ')

    # Initialize position and label
    x, y = 0, 0
    label = steps[0]

    # List to store the trajectory (includes positions, labels, and directions)
    trajectory = [(x, y, label)]

    # Update position and label based on each step
    for i in range(1, len(steps), 2):
        movement = steps[i]
        label = steps[i + 1]
        dx, dy = directions[movement]
        x, y = x + dx, y + dy
        trajectory.append((x, y, label))

    # Plot the trajectory
    fig, ax = plt.subplots(figsize=(5, 5))

    # Plot each step in the trajectory
    for i in range(len(trajectory) - 1):
        x, y, label = trajectory[i]
        dx = trajectory[i + 1][0] - x
        dy = trajectory[i + 1][1] - y
        ax.scatter(x, y, marker='x', color='red')
        ax.text(x, y, label, fontsize=12, ha='right')

        # Draw the arrows at the middle of each line
        ax.annotate('', xy=(x + dx, y + dy), xytext=(x, y),
                     arrowprops=dict(arrowstyle="->", color='blue', connectionstyle="arc3,rad=.2"))

    # Add label and scatter for the last position
    x, y, label = trajectory[-1]
    ax.scatter(x, y, marker='x', color='red')
    ax.text(x, y, label, fontsize=12, ha='right')

    # Hide the axes but keep the grid
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_xticks(range(min(x for x, _, _ in trajectory), max(x for x, _, _ in trajectory) + 1), minor=False)
    ax.set_yticks(range(min(y for _, y, _ in trajectory), max(y for _, y, _ in trajectory) + 1), minor=False)
    ax.grid(True)

    plt.show()
