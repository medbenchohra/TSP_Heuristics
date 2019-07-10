import networkx as nx
import networkx as nx2
import networkx as nxo
import matplotlib.pyplot as plt
import time
import random


# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
#  Needed variables
# ------------------

# estimated_cycle = [0]
# optimal_cycle = []
# current_cycle = []

# random_start_node = 0


# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
#  Visuals
# ---------

def update_cycle_node_list_by_labels_name(a_list):
    for i in range(1, len(a_list)):
        a_list[i] += 1
    return a_list


def create_graph(nt):
    return nt.Graph()


def color_cycle_edges(graph, to_color):
    nodes = len(to_color)
    for i in range(nodes - 2):
        graph[to_color[i+1]][to_color[i+2]]['color'] = 'g'
    graph[to_color[1]][to_color[nodes-1]]['color'] = 'g'


def draw_graph(g, cycle, nt):
    color_cycle_edges(g, cycle)
    pos = nt.circular_layout(g)
    edges_labels = dict([((u, v,), d['weight']) for u, v, d in g.edges(data=True)])
    nodes_labels = dict([(i,i+1) for i in range(g.number_of_nodes())])
    nt.draw_networkx_labels(g, pos, labels=nodes_labels)
    nt.draw_networkx_edge_labels(g, pos, edge_labels=edges_labels, rotate=False, label_pos=0.2)
    nt.draw_networkx(g, pos, with_labels= False)
    edges_colors = [g[u][v]['color'] for u, v in g.edges]
    nt.draw(g, pos, width=3, node_size=500, node_color='#A0CBE2', edges=g.edges,
            edge_color=edges_colors, with_labels=False, font_weight='bold')
    plt.show()


# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
#  Helpers
# ---------

def read_graphs(g1, g2):
    nodes = int(input("\n\n\nNumber of nodes : "))
    print("")
    for i in range(nodes):
        g1.add_node(i)
        g2.add_node(i)
    for i in range(nodes):
        for k in range(i+1, nodes):
            weight = int(input("\t - Weight between nodes (" + str(i+1) + ") & (" + str(k+1) + ") : "))
            g1.add_edge(i, k, color='#ECEAE1', weight=weight)
            g2.add_edge(i, k, color='#ECEAE1', weight=weight)


# def initialize_variables():
#     global estimated_cycle
#     global current_cycle
#     global optimal_cycle
#     global visited_nodes
#
#     estimated_cycle = [0]
#     current_cycle = [0]
#     optimal_cycle = [0]
#     visited_nodes = 0


def initialize_graphs(g1, g2):
    nodes = int(input("\n\n\nNumber of nodes : "))
    print("")
    for i in range(nodes):
        g1.add_node(i)
        g2.add_node(i)
    for i in range(nodes):
        for k in range(i+1, nodes):
            weight = random.randint(2, 20)
            g1.add_edge(i, k, color='#ECEAE1', weight=weight)
            g2.add_edge(i, k, color='#ECEAE1', weight=weight)


# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
#  Bruteforce
# ------------

def bruteforce(g, i):

    visited_nodes = 0

    def recursive_bruteforce(g, i):
        global visited_nodes
        visited_nodes = visited_nodes + 1
        current_cycle.append(i)
        for k in range(g.number_of_nodes()):
            if g.has_edge(i, k) and (current_cycle.count(k) == 0 or (current_cycle.count(k) == 1 and current_cycle[0] == k)):
                current_cycle[0] = current_cycle[0] + g[i][k]['weight']
                recursive_bruteforce(g, k)
                add_to_min(g, current_cycle, optimal_cycle)
                current_cycle.pop()
                current_cycle[0] = current_cycle[0] - g[i][k]['weight']
                visited_nodes = visited_nodes - 1


    visited_nodes = 0
    current_cycle = [0]
    optimal_cycle = [0]

    time_begin = time.process_time()
    recursive_bruteforce(g, i)
    time_end = time.process_time()
    time_exec = round(time_end - time_begin, 6)

    cost = current_cycle[0]
    cycle = current_cycle
    cycle.pop(0)

    return cycle, cost, time_exec


def add_to_min(g, current_conf, ideal_conf):

    nbr_nodes = g.number_of_nodes()
    if len(current_conf) == nbr_nodes + 1:
        last_edge_weight = g[current_conf[len(current_conf)-1]][current_conf[1]]['weight']
        if len(ideal_conf) == 1:
            ideal_conf[0] = current_conf[0] + last_edge_weight
            for i in range(nbr_nodes):
                ideal_conf.append(current_conf[i + 1])
        if (current_conf[0] + last_edge_weight) < ideal_conf[0]:
            ideal_conf[0] = current_conf[0] + last_edge_weight
            for i in range(nbr_nodes):
                ideal_conf[i+1] = current_conf[i+1]




# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
#  Heuristics
# ------------

# --------------------------------------------------------------------------------------
# -------------------
#  Nearest Neighboor
# -------------------

def heuristic_nearest_neighboor(g, i):

    estimated_cycle = []

    def recursive_nearest_neighboor(g, i):
        estimated_cycle.append(i)
        if len(estimated_cycle) == g.number_of_nodes() + 1:
            estimated_cycle[0] = estimated_cycle[0] + g[estimated_cycle[1]][i]['weight']
            return
        estimated_cycle[0] = estimated_cycle[0] + min_adj_cost(g, i)[0]
        heuristic_nearest_neighboor(g, min_adj_cost(g, i)[1])


    time_begin = time.process_time()
    recursive_nearest_neighboor(g, i)
    time_end = time.process_time()
    time_exec = round(time_end - time_begin, 6)

    cost = estimated_cycle[0]
    estimated_cycle.pop(0)

    return estimated_cycle, cost, time_exec


def min_adj_cost(g, node):
    adj_weights = []
    neighbors = [n for n in g.neighbors(node)]

    min_cost = -1
    min_cost_node = -1

    for i in neighbors:
        estimated_cycle[0] = estimated_cycle[0] - 500
        if estimated_cycle.count(i) == 0:
            weight = g[node][i]['weight']
            adj_weights.append(weight)
            if min_cost == -1 or weight < min_cost:
                min_cost = weight
                min_cost_node = i

        estimated_cycle[0] = estimated_cycle[0] + 500

    min_cost = min(adj_weights)

    return min_cost, min_cost_node


# --------------------------------------------------------------------------------------
# ------------------
#  Random Selection
# ------------------

def heuristic_random_selection(g):
    global estimated_cycle
    estimated_cycle = []
    for i in range(g.number_of_nodes()):
        estimated_cycle.append(i)
    random.shuffle(estimated_cycle)
    cost = 0
    for i in range(g.number_of_nodes()):
        cost += g[estimated_cycle[i-1]][estimated_cycle[i]]['weight']
    estimated_cycle.insert(0, cost)    # First position is for the cost)



# --------------------------------------------------------------------------------------
# -------------
#  Random Walk
# -------------

def heuristic_random_walk(g, i):
    global estimated_cycle
    estimated_cycle.append(i)
    last_city = g.number_of_nodes() - 1
    if len(estimated_cycle) == g.number_of_nodes() + 1:
        estimated_cycle[0] = estimated_cycle[0] + g[estimated_cycle[1]][i]['weight']
        return
    next_random_city = random_other_than(estimated_cycle[1:], last_city)
    estimated_cycle[0] = estimated_cycle[0] + g[i][next_random_city]['weight']
    heuristic_random_walk(g, next_random_city)


def random_other_than(alist, top):
    n = random.randint(0, top)
    while alist.__contains__(n):
        n = random.randint(0, top)

    return n


# --------------------------------------------------------------------------------------
# --------------
#  Christofides
# --------------

def heuristic_christofides(G):
    T = nxo.minimum_spanning_tree(G)
    return T

# --------------------------------------------------------------------------------------
# --------
#
# --------



# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
#  Main Program
# --------------

main_graph_bruteforce = create_graph(nx)
main_graph_heuristic = create_graph(nx2)
# read_graphs(main_graph_bruteforce, main_graph_heuristic)
initialize_graphs(main_graph_bruteforce, main_graph_heuristic)
# initialize_variables()
random_start_node = random.randrange(main_graph_bruteforce.number_of_nodes())



    # -----------------
    #  Exact execution
    # -----------------

time_begin_bruteforce = time.process_time()

bruteforce(main_graph_bruteforce, 0)

time_end_bruteforce = time.process_time()
time_bruteforce = round(time_end_bruteforce - time_begin_bruteforce, 6)



    # ---------------------
    #  Heuristic execution
    # ---------------------

time_begin_heuristic = time.process_time()

# heuristic_nearest_neighboor(main_graph_heuristic, random_start_node)
# heuristic_random_selection(main_graph_heuristic)
# heuristic_random_walk(main_graph_heuristic, random_start_node)

time_end_heuristic = time.process_time()
time_heuristic = round(time_end_heuristic - time_begin_heuristic, 6)


    # ------------------
    #  Printing results
    # ------------------

draw_graph(heuristic_christofides(main_graph_bruteforce), optimal_cycle, nx)

# draw_graph(main_graph_bruteforce, optimal_cycle, nx)
# draw_graph(main_graph_heuristic, estimated_cycle, nx2)

# print("\n\nOptimal Cycle : " + str(update_cycle_node_list_by_labels_name(optimal_cycle)[1:]))
# print("Cost : " + str(optimal_cycle[0]))
# print("Time : " + str(round(1000*time_bruteforce, 1)) + " ms")
#
# print("\n\nEstimated Cycle : " + str(update_cycle_node_list_by_labels_name(estimated_cycle)[1:]))
# print("Cost : " + str(estimated_cycle[0]))
# print("Time : " + repr(round(1000*time_heuristic, 1)) + " ms")
#
# print("\n\n - Estimation is " + str(round(estimated_cycle[0]/optimal_cycle[0], 1))
#       + " further from optimal\n\n")



# ---------------------------------------------------------------------------------------
