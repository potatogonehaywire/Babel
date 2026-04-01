import math
import random
import networkx as nx
import matplotlib.pyplot as plt


def save_png(graph, pos, width, height, path, dpi=150):
    scale = 8 / max(width, height)  # fit longest side to 8 inches
    fig, ax = plt.subplots(figsize=(width * scale, height * scale))

    nx.draw(graph, pos=pos, with_labels=True, ax=ax,
            node_color="#534AB7", font_color="white", edge_color="#aaa")

    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect("equal")   # prevent any remaining distortion
    ax.set_clip_on(False)
    for artist in ax.get_children():
        artist.set_clip_on(False)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def repulsive_f(num_nodes, ideal_dist, positions, displacement):
    for first_node in range(num_nodes):
        for other_node in range(num_nodes):
            if first_node != other_node:
                dist_x = positions[first_node][0] - positions[other_node][0]
                dist_y = positions[first_node][1] - positions[other_node][1]
                dist = math.sqrt(dist_x ** 2 + dist_y ** 2)

                displacement[first_node][0] += (dist_x/dist) * (ideal_dist**2 /dist)
                displacement[first_node][1] += (dist_y/dist) * (ideal_dist**2 /dist)
    
    return displacement


def attractive_f(edges, ideal_dist, positions, displacement):

    for first_node, other_node in edges:

        dist_x = positions[first_node][0] - positions[other_node][0]
        dist_y = positions[first_node][1] - positions[other_node][1]
        dist = math.sqrt(dist_x ** 2 + dist_y ** 2)

        displacement[first_node][0] -= (dist_x/dist) * (dist**2 / ideal_dist)
        displacement[first_node][1] -= (dist_y/dist) * (dist**2 / ideal_dist)
        displacement[other_node][0] += (dist_x/dist) * (dist**2 / ideal_dist)
        displacement[other_node][1] += (dist_y/dist) * (dist**2 / ideal_dist)
    return displacement


def fruchterman_reingold(edges, num_nodes, width, height, iterations):

    positions = {}
    displacement = {}

    for i in range(num_nodes):
        positions[i] = (random.uniform(0, width), random.uniform(0, height))
    
    F = nx.Graph()
    F.add_edges_from(edges)
    save_png(F, positions, 500, 200, "graph_final.png", 150)
    
    area = width * height
    ideal_dist = math.sqrt(area/num_nodes)

    temperature = width / 10.0
    cooling = temperature / (iterations + 1)

    for _ in range(iterations):
        for i in range(num_nodes):
            displacement[i] = [0.0, 0.0]
    
        displacement = repulsive_f(num_nodes, ideal_dist, positions, displacement)
        displacement = attractive_f(edges, ideal_dist, positions, displacement)

        for node in range(num_nodes):
            dx, dy = displacement[node]
            magnitude = math.sqrt(dx**2 + dy**2)
            clamped = min(magnitude, temperature)
            new_x = positions[node][0] + (dx/magnitude) * clamped
            new_y = positions[node][1] + (dy/magnitude) * clamped
            
            new_x = max(0.0, min(width, new_x))
            new_y = max(0.0, min(height, new_y))
            positions[node] = (int(new_x), int(new_y))
        
        temperature = max(temperature - cooling, 1e-6)
    
    return positions


def angles(positions, edges, num_nodes):
    
    for first_node, other_node in edges:
        dist_x = positions[first_node][0] - positions[other_node][0]
        dist_y = positions[first_node][1] - positions[other_node][1]
        dist = math.sqrt(dist_x ** 2 + dist_y ** 2)
        if dist_y == 0:
            angle = math.pi/2
        elif dist_x == 0:
            angle = math.pi
        else:
            angle = math.atan(abs(dist_x) / abs(dist_y))
        
        if angle < math.pi/12:
            ideal_angle = 0
        elif math.pi/12 < angle <= math.pi/4:
            ideal_angle = math.pi/6
        elif math.pi/4 < angle <= 5 * math.pi/12:
            ideal_angle = math.pi/3
        elif 5 * math.pi/12 < angle:
            ideal_angle = math.pi/2
            
        print(first_node, other_node, math.degrees(angle), math.degrees(ideal_angle))


def gravity(positions, num_nodes, width, height, iterations):
    area = width * height
    ideal_dist = math.sqrt(area / num_nodes)
    temperature = width / 10.0
    cooling = temperature / (iterations + 1)
    stopped_nodes = set()
    too_close = []
    rand_node = random.randint(0, num_nodes - 1)
    stopped_nodes.add(rand_node)
    print(stopped_nodes)
    for _ in range(iterations):
        displacement = {i: [0.0, 0.0] for i in range(num_nodes)}

        for other_node in range(num_nodes):
            if other_node not in stopped_nodes:
                dist_x = positions[rand_node][0] - positions[other_node][0]
                dist_y = positions[rand_node][1] - positions[other_node][1]
                dist = math.sqrt(dist_x ** 2 + dist_y ** 2) or 1e-6

                if dist < 50.0:
                    stopped_nodes.add(other_node)
                else:
                    force = dist ** 2 / ideal_dist
                    displacement[other_node][0] += (dist_x / dist) * force
                    displacement[other_node][1] += (dist_y / dist) * force 
                    displacement[rand_node][0]  -= (dist_x / dist) * force
                    displacement[rand_node][1]  -= (dist_y / dist) * force 

        for node in range(num_nodes):
            dx, dy = displacement[node]
            magnitude = math.sqrt(dx ** 2 + dy ** 2) or 1e-6
            clamped = min(magnitude, temperature)
            new_x = positions[node][0] + (dx / magnitude) * clamped
            new_y = positions[node][1] + (dy / magnitude) * clamped
            new_x = max(0.0, min(width, new_x))
            new_y = max(0.0, min(height, new_y))
            positions[node] = (new_x, new_y)

        positions, too_close = separate(positions, num_nodes, 50.0)
        for item in too_close:
            stopped_nodes.add(item)
        temperature = max(temperature - cooling, 1e-6)

    return positions


def separate(positions, num_nodes, min_dist):
    too_close = []
    for node in range(num_nodes):
        for other_node in range(node+1, num_nodes):
            dx = positions[node][0] - positions[other_node][0]
            dy = positions[node][1] - positions[other_node][1]
            dist = math.sqrt(dx**2 + dy**2)

            if dist < min_dist:
                too_close.append(node)
                too_close.append(other_node)
                overlap = (min_dist - dist)/2
                push_x = (dx / dist) * overlap
                push_y = (dx/dist) * overlap
                node_x, node_y = positions[node]
                other_node_x, other_node_y = positions[other_node]
                positions[node] = (node_x + push_x, node_y + push_y)
                positions[other_node] = (other_node_x - push_x, other_node_y - push_y)
    
    return positions, too_close


def main():
    edges = [(0,1),(1,2), (2,3), (3,4), (4,5), (5,6), (6,7),(4,6), (4,2), (3,6), (2,5), (1,5)]
    num_nodes = 8
    positions = fruchterman_reingold(edges, num_nodes, 500, 300, 100)
    print(positions)
    G = nx.Graph()
    G.add_edges_from(edges)
    save_png(G, positions, 500, 300, "graph_final.png", 150)
    
    angles(positions, edges, num_nodes)
           
    positions = gravity(positions, num_nodes, 500, 300, 50)
    print(positions)
    H = nx.Graph()
    H.add_edges_from(edges)
    save_png(H, positions, 500, 300, "graph_gravity.png", 150)


main()
