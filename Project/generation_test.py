import math
import random
import networkx as nx
import matplotlib.pyplot as plt
from itertools import permutations

def center_force(positions, displacement, width, height, strength=0.01):
    cx = width / 2
    cy = height / 2

    for i, (x, y) in positions.items():
        displacement[i][0] += (cx - x) * strength
        displacement[i][1] += (cy - y) * strength


def edge_spring(edges, positions, displacement, target_len, strength=0.05):
    for u, v in edges:
        dx = positions[v][0] - positions[u][0]
        dy = positions[v][1] - positions[u][1]
        dist = math.sqrt(dx*dx + dy*dy) or 1e-6

        force = (dist - target_len) * strength
        fx = (dx / dist) * force
        fy = (dy / dist) * force

        displacement[u][0] += fx
        displacement[u][1] += fy
        displacement[v][0] -= fx
        displacement[v][1] -= fy


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
        dist = math.sqrt(dist_x ** 2 + dist_y ** 2) or 1e-6

        displacement[first_node][0] -= (dist_x/dist) * (dist**2 / ideal_dist)
        displacement[first_node][1] -= (dist_y/dist) * (dist**2 / ideal_dist)
        displacement[other_node][0] += (dist_x/dist) * (dist**2 / ideal_dist)
        displacement[other_node][1] += (dist_y/dist) * (dist**2 / ideal_dist)
    return displacement


def gravity_f(num_nodes, positions, displacement, width, height, gravity=1.0):
    center_x = width / 2
    center_y = height / 2
    for node in range(num_nodes):
        dx = center_x - positions[node][0]
        dy = center_y - positions[node][1]
        displacement[node][0] += dx * gravity
        displacement[node][1] += dy * gravity
    return displacement


def fruchterman_reingold(edges, num_nodes, width, height, iterations):

    positions = {}
    displacement = {}
    for i in range(num_nodes):
        positions[i] = (random.uniform(0, width), random.uniform(0, height))
        
    E = nx.Graph()
    E.add_edges_from(edges)
    save_png(E, positions, 500, 300, "graph_original.png", 150)    
    
    area = width * height
    ideal_dist = math.sqrt(area/(num_nodes*2))

    temperature = width / 10.0
    cooling = temperature / (iterations + 1)

    for _ in range(iterations):
        for i in range(num_nodes):
            displacement[i] = [0.0, 0.0]
    
        displacement = repulsive_f(num_nodes, ideal_dist, positions, displacement)
        displacement = attractive_f(edges, ideal_dist, positions, displacement)
        displacement = gravity_f(num_nodes, positions, displacement, width, height, gravity=0.03)

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
                if dist != 0:
                    push_x = (dx / dist) * overlap
                    push_y = (dy /dist) * overlap
                else:
                    push_x = 80
                    push_y = 80
                node_x, node_y = positions[node]
                other_node_x, other_node_y = positions[other_node]
                positions[node] = (node_x + push_x, node_y + push_y)
                positions[other_node] = (other_node_x - push_x, other_node_y - push_y)
    
    return positions


def segments_intersect(p1, p2, p3, p4):
    """Returns True if segment p1-p2 intersects segment p3-p4 (ignoring shared endpoints)."""
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)

    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False


def edges_cross(e1, e2, positions):
    """Returns True if two edges cross (shared nodes are not considered crossings)."""
    a, b = e1
    c, d = e2
    if len({a, b, c, d}) < 4:  # shared endpoint → not a crossing
        return False
    return segments_intersect(positions[a], positions[b], positions[c], positions[d])


def reconnect(positions, num_nodes):
    # Build all candidate edges sorted by distance
    candidates = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            dx = positions[i][0] - positions[j][0]
            dy = positions[i][1] - positions[j][1]
            dist = math.sqrt(dx**2 + dy**2)
            candidates.append((dist, i, j))
    candidates.sort()

    accepted = []
    degree = {i: 0 for i in range(num_nodes)}

    # --- Pass 1: Kruskal-style spanning tree (guarantees connectivity) ---
    parent = list(range(num_nodes))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        parent[find(x)] = find(y)

    for dist, i, j in candidates:
        if find(i) == find(j):
            continue  # already connected, skip
        if degree[i] >= 2 or degree[j] >= 2:
            continue  # can't use this edge without violating degree cap

        new_edge = (i, j)
        if not any(edges_cross(new_edge, existing, positions) for existing in accepted):
            accepted.append(new_edge)
            degree[i] += 1
            degree[j] += 1
            union(i, j)

    # Check if spanning tree was achievable — warn if not
    roots = set(find(i) for i in range(num_nodes))
    if len(roots) > 1:
        print(f"Warning: could not fully connect graph ({len(roots)} components). "
              f"Degree-2 + no-crossing constraints may be too strict for this layout.")

    # --- Pass 2: Add remaining non-crossing edges up to degree cap ---
    for dist, i, j in candidates:
        if degree[i] >= 2 or degree[j] >= 2:
            continue
        new_edge = (i, j)
        if new_edge not in accepted:
            if not any(edges_cross(new_edge, existing, positions) for existing in accepted):
                accepted.append(new_edge)
                degree[i] += 1
                degree[j] += 1

    return accepted


def main():
    edges = [(0,1),(1,2), (2,3), (3,4), (4,5), (5,6), (6,7),(4,6), (4,2), (3,6), (2,5), (1,5), (7,8), (8,9), (9,10), (10,11), (11,12),(12,13), (13,14), (14,15), (15,16),(4,6), (4,2), (3,6), (2,5), (1,5), (14,3), (15, 7), (13, 9), (8, 3), (10, 13), (10, 12), (14, 11), (15, 1), (11, 3), (11, 7), (11, 9), (8, 5)]
    #, (7,8), (8,9), (9,10), (10,11), (11,12),(12,13), (13,14), (14,15), (15,16),(4,6), (4,2), (3,6), (2,5), (1,5), (14,3), (15, 7), (13, 9), (8, 3), (10, 13), (10, 12), (14, 11), (15, 1), (11, 3), (11, 7), (11, 9), (8, 5)
    num_nodes = 17

    positions = fruchterman_reingold(edges, num_nodes, 500, 500, 100)
    G = nx.Graph()
    G.add_edges_from(edges)
    save_png(G, positions, 500, 500, "graph_force.png", 150)
    
    # positions = change_angles(edges, positions, num_nodes, 500, 300, 100)
    # M = nx.Graph()
    # M.add_edges_from(edges)
    # save_png(M, positions, 500, 300, "graph_angles.png", 150)

    edges = reconnect(positions, num_nodes)
    
    H = nx.Graph()
    H.add_edges_from(edges)
    save_png(H, positions, 500, 500, "graph_reorder.png", 150)

    positions
    

    
main()