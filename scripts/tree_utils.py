import random
import networkx as nx
import csrgraph
import string

def generate_name() -> str:
    return ''.join(random.choices(string.ascii_lowercase, k=2))

def create_nuclear_family(num_children, child_names):
    # Generate parent names
    parent1 = generate_name()
    parent2 = generate_name()
    
    # Ensure parent names are unique
    while parent2 == parent1:
        parent2 = generate_name()

    # Generate additional children names if necessary
    additional_names_needed = num_children - len(child_names)
    additional_children = [generate_name() for _ in range(additional_names_needed)]

    # Combine provided child names with generated ones
    children = child_names + additional_children

    # Define relationships
    relationships = {
        parent1: {"SPOUSE_OF": [parent2], "PARENT_OF": children},
        parent2: {"SPOUSE_OF": [parent1], "PARENT_OF": children},
    }

    # Add child and sibling relationships
    for child in children:
        relationships[child] = {
            "CHILD_OF": [parent1, parent2],
            "SIBLING_OF": [sibling for sibling in children if sibling != child]  # Exclude self from siblings
        }
    
    # Define roles
    roles = {parent1: "parent", parent2: "parent"}
    roles.update({child: "child" for child in children})

    return relationships, roles

def create_extended_family_tree(base_num_children: int, grandparent_num_children: int):
    # Step 1: Create the base family
    base_family_children_names = []
    base_family_relationships, base_family_roles = create_nuclear_family(base_num_children, base_family_children_names)
    
    # Extract parent names from the base family to use as children for the grandparent families
    parents_names = [name for name, role in base_family_roles.items() if role == 'parent']
    
    # Step 2: Create grandparent families for each parent in the base family
    grandparent_families = {}
    for parent_name in parents_names:
        grandparent_family_relationships, grandparent_family_roles = create_nuclear_family(grandparent_num_children, [parent_name])
        grandparent_families[parent_name] = {"relationships": grandparent_family_relationships, "roles": grandparent_family_roles}
    
    # Combine all families into a comprehensive family structure
    # This structure will include the base family and both sets of grandparents, aunts, and uncles
    combined_relationships = base_family_relationships
    for gp_family in grandparent_families.values():
        # Merge relationships, ensuring not to overwrite existing ones
        for name, relations in gp_family["relationships"].items():
            if name in combined_relationships:
                for rel_type, rel_names in relations.items():
                    if rel_type in combined_relationships[name]:
                        combined_relationships[name][rel_type] = list(set(combined_relationships[name][rel_type] + rel_names))
                    else:
                        combined_relationships[name][rel_type] = rel_names
            else:
                combined_relationships[name] = relations
    
    return combined_relationships

def infer_grandparent_edges(relationships):
    # Temporarily store grandparent relationships to avoid modifying the dictionary while iterating
    temp_relationships = {}

    for person, rels in relationships.items():
        if 'PARENT_OF' in rels:
            for child in rels['PARENT_OF']:
                if 'PARENT_OF' in relationships[child]:
                    for grandchild in relationships[child]['PARENT_OF']:
                        # Ensure initialization of 'GRANDPARENT_OF' list for person
                        if person not in temp_relationships:
                            temp_relationships[person] = {}
                        if 'GRANDPARENT_OF' not in temp_relationships[person]:
                            temp_relationships[person]['GRANDPARENT_OF'] = []
                        temp_relationships[person]['GRANDPARENT_OF'].append(grandchild)
                        
                        # Ensure initialization of 'GRANDCHILD_OF' list for grandchild
                        if grandchild not in temp_relationships:
                            temp_relationships[grandchild] = {}
                        if 'GRANDCHILD_OF' not in temp_relationships[grandchild]:
                            temp_relationships[grandchild]['GRANDCHILD_OF'] = []
                        temp_relationships[grandchild]['GRANDCHILD_OF'].append(person)

    # Update the original relationships with the inferred grandparent relationships
    for person, rels in temp_relationships.items():
        if person in relationships:
            for key, value in rels.items():
                if key in relationships[person]:
                    relationships[person][key].extend(value)
                else:
                    relationships[person][key] = value
        else:
            relationships[person] = rels

    return relationships


def create_family_tree_digraph(relationships):
    G = nx.DiGraph()
    for person, rels in relationships.items():
        G.add_node(person)
        for rel_type, related_individuals in rels.items():
            for related in related_individuals:
                G.add_edge(person, related, relationship=rel_type)
    return G

def generate_random_walks(G, n=5, walk_length=50):
    walks = []
    for _ in range(n):
        # Start from a random node
        current_node = random.choice(list(G.nodes))
        walk = [current_node]
        for _ in range(walk_length):
            neighbors = list(G.successors(current_node))
            if not neighbors:
                break  # If the current node has no out-edges, end the walk
            next_node = random.choice(neighbors)
            # Get the relationship type for the edge
            edge_data = G.get_edge_data(current_node, next_node)
            relationship = edge_data['relationship']
            walk.append(relationship)
            walk.append(next_node)
            current_node = next_node
        walks.append(' '.join(walk))
    return walks

def get_walks_for_n_trees(n_graphs=2000, 
                          n_walks=10, 
                          base_num_children=2, 
                          grandparent_num_children=2,
                          walk_length=50):
    all_walks = []
    all_graphs = []
    for i in range(n_graphs):
        r = create_extended_family_tree(base_num_children=base_num_children, 
                                        grandparent_num_children=grandparent_num_children)
        r = infer_grandparent_edges(r)
        G = create_family_tree_digraph(r)
        random_walks = generate_random_walks(G, n=n_walks, walk_length=walk_length)
        all_walks.extend(random_walks)
        all_graphs.append(G)
    return all_walks, all_graphs
