import math
import random
import networkx as nx
import matplotlib.pyplot as plt
from itertools import permutations


def assign_angles(positions, edges, num_nodes):
    from collections import defaultdict

    SLOTS = {
        0 : math.pi/2,
        math.pi/2 : 0,
        math.pi / 3 : math.pi/6,
        math.pi / 6 : math.pi/3,
        2 * math.pi / 3 : 5 * math.pi/6,
        5 * math.pi / 6 : 2 * math.pi / 3,
        7 * math.pi / 6 : 4 * math.pi / 3,
        4 * math.pi / 3 : 7 * math.pi / 6,
        5 * math.pi / 3 : 11 * math.pi / 6,
        11 * math.pi / 6 : 5 * math.pi / 3
    }    


    adjacency = defaultdict(list)
    for u, v in edges:
        adjacency[u].append(v)
        adjacency[v].append(u)

    ideal_angles = {}
    used_slots = defaultdict(set)

    slot_list = list(SLOTS.keys())

    for u, v in edges:
        # skip if edge already handled
        if (u, v) in ideal_angles:
            continue

        # find a free slot for u
        chosen_slot = None
        for slot in slot_list:
            if slot not in used_slots[u] and SLOTS[slot] not in used_slots[v]:
                chosen_slot = slot
                break

        if chosen_slot is None:
            raise ValueError(f"No available angle slot for edge ({u}, {v})")

        paired = SLOTS[chosen_slot]

        ideal_angles[(u, v)] = chosen_slot
        ideal_angles[(v, u)] = paired

        used_slots[u].add(chosen_slot)
        used_slots[v].add(paired)

    return ideal_angles


#def angle_diff(a, b):
    # smallest angular distance between two angles
    #diff = abs(a - b) % (2 * math.pi)
    #return min(diff, 2 * math.pi - diff)


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


def angles(positions, edges, num_nodes, width, height, ideal_angles, step=0.1):
    displacement = {i: [0.0, 0.0] for i in range(num_nodes)}

    for first_node, other_node in edges:
        dist_x = positions[other_node][0] - positions[first_node][0]
        dist_y = positions[other_node][1] - positions[first_node][1]
        dist = math.sqrt(dist_x**2 + dist_y**2) or 1e-6

        for src, dst in [(first_node, other_node), (other_node, first_node)]:
            target_angle = ideal_angles.get((src, dst))
            if target_angle is None:
                continue
            ideal_x = positions[src][0] + math.cos(target_angle) * dist
            ideal_y = positions[src][1] + math.sin(target_angle) * dist
            displacement[dst][0] += (ideal_x - positions[dst][0]) * step
            displacement[dst][1] += (ideal_y - positions[dst][1]) * step

    for node in range(num_nodes):
        dx, dy = displacement[node]
        new_x = max(0.0, min(width,  positions[node][0] + dx))
        new_y = max(0.0, min(height, positions[node][1] + dy))
        positions[node] = (new_x, new_y)

    return positions


def gravity(edges, positions, num_nodes, width, height, iterations):
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

        #for other_node in range(num_nodes):
            #if other_node not in stopped_nodes:
                #dist_x = positions[rand_node][0] - positions[other_node][0]
                #dist_y = positions[rand_node][1] - positions[other_node][1]
                #dist = math.sqrt(dist_x ** 2 + dist_y ** 2) or 1e-6

                #if dist < 10.0:
                    #stopped_nodes.add(other_node)
                #else:
                    #force = dist ** 2 / ideal_dist
                    #displacement[other_node][0] += (dist_x / dist) * force
                    #displacement[other_node][1] += (dist_y / dist) * force 
                    #displacement[rand_node][0]  -= (dist_x / dist) * force
                    #displacement[rand_node][1]  -= (dist_y / dist) * force 
        
        
        #for node in range(num_nodes):
            #dx, dy = displacement[node]
            #magnitude = math.sqrt(dx ** 2 + dy ** 2) or 1e-6
            #clamped = min(magnitude, temperature)
            #new_x = positions[node][0] + (dx / magnitude) * clamped
            #new_y = positions[node][1] + (dy / magnitude) * clamped
            #new_x = max(0.0, min(width, new_x))
            #new_y = max(0.0, min(height, new_y))
            #positions[node] = (new_x, new_y)

        # recompute slot assignments based on current positions
        ideal_angles = assign_angles(positions, edges, num_nodes)
        print(ideal_angles)
    
        positions = separate(positions, num_nodes, 50.0)
        positions = angles(positions, edges, num_nodes, width, height, ideal_angles)


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
                if dist != 0:
                    push_x = (dx / dist) * overlap
                    push_y = (dx/dist) * overlap
                else:
                    push_x = 50
                    push_y = 50
                node_x, node_y = positions[node]
                other_node_x, other_node_y = positions[other_node]
                positions[node] = (node_x + push_x, node_y + push_y)
                positions[other_node] = (other_node_x - push_x, other_node_y - push_y)
    
    return positions


def reconnect(positions, edges, num_nodes):
    connectable_nodes = { i: i for i in range(num_nodes)}
    edges = [(i, 0) for i in range(num_nodes)]
    distance = {i: {a: 0 for a in range(num_nodes)} for i in range(num_nodes)}
    closest = {i: [0,5000] for i in range(num_nodes)}
    
    for node, coords in positions.items():
        for other_node in range(node + 1, num_nodes):
            dist_x = coords[0] - positions[other_node][0]
            dist_y = coords[1] - positions[other_node][1]
            dist_total = math.sqrt(dist_x ** 2 + dist_y ** 2)
            
            distance[node][other_node] = dist_total
            
    node = 0
    connectable_nodes.pop(node)
    for i in range(num_nodes):
        print("current node", node)
        #connectable_nodes.remove(node)
        for other_node in connectable_nodes:
            if distance[node][other_node] < closest[node][1]:
                closest[node][0] = other_node
                closest[node][1] = distance[node][other_node]
        print("closest node", closest[node][0])

        if len(connectable_nodes) > 1 :
            print("before pop", connectable_nodes)
            node = connectable_nodes.pop(closest[node][0])
            print("after pop", connectable_nodes)
                
    #for node in connectable_nodes:
        #connectable_nodes.remove(node)
        #for other_node in connectable_nodes:
            #if distance[node][other_node] < closest[node][1]:
                #closest[node][0] = other_node
                #closest[node][1] = distance[node][other_node]
                #print(closest[node][0])
                #print(connectable_nodes)

            #if closest[node][0] in connectable_nodes:
                #connectable_nodes.pop(closest[node][0])

    print(distance)
    for i in range(num_nodes):
        edges[i] = (i, closest[i][0])

    edges.remove((closest[node][0], 0))
    print(edges)
    return edges


def main():
    edges = [(0,1),(1,2), (2,3), (3,4), (4,5), (5,6), (6,7),(4,6), (4,2), (3,6), (2,5), (1,5)]
    #, (7,8), (8,9), (9,10), (10,11), (11,12),(12,13), (13,14), (14,15), (15,16),(4,6), (4,2), (3,6), (2,5), (1,5), (14,3), (15, 7), (13, 9), (8, 3), (10, 13), (10, 12), (14, 11), (15, 1), (11, 3), (11, 7), (11, 9), (8, 5)
    num_nodes = 8
    positions = fruchterman_reingold(edges, num_nodes, 500, 300, 100)
    print(f"fruchterman positions {positions}")
    G = nx.Graph()
    G.add_edges_from(edges)
    save_png(G, positions, 500, 300, "graph_final.png", 150)
    
    edges = reconnect(positions, edges, num_nodes)
    
    H = nx.Graph()
    H.add_edges_from(edges)
    save_png(H, positions, 500, 300, "graph_reorder.png", 150)
    
    positions = gravity(edges, positions, num_nodes, 500, 300, 100)
    M = nx.Graph()
    M.add_edges_from(edges)
    save_png(M, positions, 500, 300, "graph_gravity.png", 150)
    #print(f"gravity + angles positions {positions}")

    


main()
