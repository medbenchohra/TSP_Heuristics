import networkx as nx
import networkx as nx2
import networkx as nxo
import matplotlib.pyplot as plt
import time
import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from operator import itemgetter

from scipy.spatial import distance
from pytsp import christofides_tsp
import collections

# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
#  Needed variables
# ------------------

estimated_cycle = [0]
cycle_optimal = []
current_cycle = []
visited_nodes = 0
random_start_node = 0



# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
#  Visuals
# ---------

def create_graph(nt):
    return nt.Graph()


def color_cycle_edges(graph, to_color):
    nodes = len(to_color)
    for i in range(nodes):
        graph[to_color[i-1]][to_color[i]]['color'] = 'g'


def draw_graph(g, cycle, nt):
    color_cycle_edges(g, cycle)
    pos = nt.circular_layout(g)
    edges_labels = dict([((u, v,), d['weight']) for u, v, d in g.edges(data=True)])
    nodes_labels = dict([(i,i) for i in range(g.number_of_nodes())])
    nt.draw_networkx_labels(g, pos, labels=nodes_labels)
    nt.draw_networkx_edge_labels(g, pos, edge_labels=edges_labels, rotate=False, label_pos=0.2)
    nt.draw_networkx(g, pos, with_labels= False)
    edges_colors = [g[u][v]['color'] for u, v in g.edges]
    nt.draw(g, pos, width=3, node_size=500, node_color='#A0CBE2', edges=g.edges,
            edge_color=edges_colors, with_labels=False, font_weight='bold')
    plt.show()


# ------------------------------------------------------------------------------------------
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


def initialize_variables():
    global estimated_cycle
    global current_cycle
    global cycle_optimal
    global visited_nodes

    estimated_cycle = [0]
    current_cycle = [0]
    cycle_optimal = [0]
    visited_nodes = 0


def initialize_graphs(g1, g2):
    nodes = int(input("\n\nNumber of nodes : "))
    print("")
    for i in range(nodes):
        g1.add_node(i)
        g2.add_node(i)
    for i in range(nodes):
        for k in range(i+1, nodes):
            weight = random.randint(2, 100)
            g1.add_edge(i, k, color='#ECEAE1', weight=weight)
            g2.add_edge(i, k, color='#ECEAE1', weight=weight)


def calculate_cost_from_path(G, path):
    # Path is noncyclic

    cost = 0
    for i in range(G.number_of_nodes()):
        cost += G[path[i-1]][path[i]]['weight']

    return cost


def coord_to_adj_matrix(coord):
    num_nodes = coord.shape[0]
    dist_mat = np.zeros((num_nodes, num_nodes))
    for _ in range(0, num_nodes):
        for i in range(_ + 1, num_nodes):
            a = (coord[_][0], coord[_][1])
            b = (coord[i][0], coord[i][1])
            dist_mat[_][i] = int(distance.euclidean(a, b))

    dist_mat += dist_mat.T

    return dist_mat



# ------------------------------------------------------------------------------------------
# -------------------------------------- Bruteforce ----------------------------------------
# ------------------------------------------------------------------------------------------

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



def bruteforce(g, i):
    global visited_nodes
    global current_cycle

    visited_nodes = visited_nodes + 1
    current_cycle.append(i)
    for k in range(g.number_of_nodes()):
        if g.has_edge(i, k) and (current_cycle.count(k) == 0 or (current_cycle.count(k) == 1 and current_cycle[0] == k)):
            current_cycle[0] = current_cycle[0]+g[i][k]['weight']
            bruteforce(g, k)
            add_to_min(g, current_cycle, cycle_optimal)
            current_cycle.pop()
            current_cycle[0] = current_cycle[0] - g[i][k]['weight']
            visited_nodes = visited_nodes - 1



# ------------------------------------------------------------------------------------------
# -------------------------------------- Heuristics ----------------------------------------
# ------------------------------------------------------------------------------------------

# -------------------
#  Nearest Neighboor
# -------------------

def heuristic_nearest_neighboor(g, i):
    time_begin = time.process_time_ns()  # Starting timer
    heuristic_exec = heuristic_nearest_neighboor_recursive(g, i)
    path = heuristic_exec[0]
    cost = heuristic_exec[1]
    time_end = time.process_time_ns()  # Stopping timer

    time_exec = time_end - time_begin

    return path, cost, time_exec


def heuristic_nearest_neighboor_recursive(g, i):
    global estimated_cycle
    estimated_cycle.append(i)
    if len(estimated_cycle) == g.number_of_nodes() + 1:
        estimated_cycle[0] = estimated_cycle[0] + g[estimated_cycle[1]][i]['weight']
        return estimated_cycle[1:], estimated_cycle[0]
    estimated_cycle[0] = estimated_cycle[0] + min_adj_cost(g, i)[0]
    return heuristic_nearest_neighboor_recursive(g, min_adj_cost(g, i)[1])


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
    path = []

    time_begin = time.process_time_ns()  # Starting timer
    for i in range(g.number_of_nodes()):
        path.append(i)
    random.shuffle(path)
    cost = calculate_cost_from_path(g, path)
    time_end = time.process_time_ns()  # Stopping timer

    time_exec = time_end - time_begin

    return path, cost, time_exec


# --------------------------------------------------------------------------------------
# -------------
#  Random Walk
# -------------

def heuristic_random_walk(g, i):
    time_begin = time.process_time_ns()  # Starting timer
    heuristic_execution = heuristic_random_walk_recursive(g, i)
    time_end = time.process_time_ns()  # Stopping timer

    path = heuristic_execution[0]
    cost = heuristic_execution[1]
    time_exec = time_end - time_begin

    return path, cost, time_exec


def heuristic_random_walk_recursive(g, i):
    global estimated_cycle
    estimated_cycle.append(i)
    last_city = g.number_of_nodes() - 1

    if len(estimated_cycle) == g.number_of_nodes() + 1:
        estimated_cycle[0] = estimated_cycle[0] + g[estimated_cycle[1]][i]['weight']
        return estimated_cycle[1:], estimated_cycle[0]

    next_random_city = random_other_than(estimated_cycle[1:], last_city)
    estimated_cycle[0] = estimated_cycle[0] + g[i][next_random_city]['weight']
    return heuristic_random_walk_recursive(g, next_random_city)


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

    mat = np.array(create_adjacency_matrix(G))

    time_begin = time.process_time_ns()  # Starting timer
    path = christofides_tsp.christofides_tsp(mat)
    time_end = time.process_time_ns()  # Stopping timer

    cost = calculate_cost_from_path(G, path)
    time_exec = time_end - time_begin

    return path, cost, time_exec


def to_upper_matrix(matrix):
    m = matrix
    n = len(m)

    for i in range(n):
        for j in range(i):
            m[i][j] = 0

    return m


def create_adjacency_matrix(graph):
    matrix = []
    nb_nodes = graph.number_of_nodes()

    for i in range(nb_nodes):
        line = []
        for j in range(nb_nodes):
            if i==j :
                line.append(0)
            else :
                line.append(graph[i][j]['weight'])
        matrix.append(line)

    return matrix



# --------------------------------------------------------------------------------------
# ---------
#  Two-Opt
# ---------

def cost_change(cost_mat, n1, n2, n3, n4):
    return cost_mat[n1][n3] + cost_mat[n2][n4] - cost_mat[n1][n2] - cost_mat[n3][n4]


def heuristic_two_opt(route, cost_mat):
    best = route
    cost = 0
    improved = True
    while improved:
        improved = False
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route)):
                if j - i == 1: continue
                if cost_change(cost_mat, best[i - 1], best[i], best[j - 1], best[j]) < 0:
                    best[i:j] = best[j - 1:i - 1:-1]
                    improved = True
        route = best
    for _ in range(0, len(best) - 1):
        cost += cost_mat[best[_]][best[_ + 1]]

    return best, cost




# ------------------------------------------------------------------------------------------
# ------------------------------------ Metaheuristics --------------------------------------
# ------------------------------------------------------------------------------------------

# -------------------
#  Genetic Algorithm
# -------------------

# Class City which represents the coord of a city. The city will represent the gene in our GA
class City:
    def __init__(self, node, x, y):
        self.node = node
        self.x = x
        self.y = y

    def distance(self, city):
        xDis = abs(self.x - city.x)
        yDis = abs(self.y - city.y)
        distance = np.sqrt((xDis ** 2) + (yDis ** 2))
        return distance

    def __repr__(self):
        return str(self.node)



# Class Fitness tells us how good each route is. In our case the fitness is 1/global_dist
class Fitness:
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness = 0.0

    def routeDistance(self):
        if self.distance == 0:
            pathDistance = 0
            for i in range(0, len(self.route)):
                fromCity = self.route[i]
                toCity = None
                if i + 1 < len(self.route):
                    toCity = self.route[i + 1]
                else:
                    toCity = self.route[0]
                pathDistance += fromCity.distance(toCity)
            self.distance = pathDistance
        return self.distance

    def routeFitness(self):
        if self.fitness == 0:
            self.fitness = 1 / float(self.routeDistance())
        return self.fitness



# First step of our GA for TSP. Create a population
def createRoute(cityList):
    route = random.sample(cityList, len(cityList))
    return route



def initialPopulation(popSize, cityList):
    population = []

    for i in range(0, popSize):
        population.append(createRoute(cityList))
    return population



# Rank routes according to their Fitness
def rankRoutes(population):
    fitnessResults = {}
    for i in range(0,len(population)):
        fitnessResults[i] = Fitness(population[i]).routeFitness()
    return sorted(fitnessResults.items(), key = itemgetter(1), reverse = True)



def selection(popRanked, eliteSize):
    selectionResults = []
    df = pd.DataFrame(np.array(popRanked), columns=["Index", "Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100 * df.cum_sum / df.Fitness.sum()

    for i in range(0, eliteSize):
        selectionResults.append(popRanked[i][0])
    for i in range(0, len(popRanked) - eliteSize):
        pick = 100 * random.random()
        for i in range(0, len(popRanked)):
            if pick <= df.iat[i, 3]:
                selectionResults.append(popRanked[i][0])
                break
    return selectionResults



# Create the mating pool : a mating pool is a collection of parents to create the next generation
def matingPool(population, selectionResults):
    matingpool = []
    for i in range(0, len(selectionResults)):
        index = selectionResults[i]
        matingpool.append(population[index])
    return matingpool



def breed(parent1, parent2):
    child = []
    childP1 = []
    childP2 = []

    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1))

    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)

    for i in range(startGene, endGene):
        childP1.append(parent1[i])

    childP2 = [item for item in parent2 if item not in childP1]

    child = childP1 + childP2
    return child


def breedPopulation(matingpool, eliteSize):
    children = []
    length = len(matingpool) - eliteSize
    pool = random.sample(matingpool, len(matingpool))

    for i in range(0, eliteSize):
        children.append(matingpool[i])

    for i in range(0, length):
        child = breed(pool[i], pool[len(matingpool) - i - 1])
        children.append(child)
    return children



# Mutate: is a way to introduce variation in our population randomly by swapping two cities in a route
def mutate(individual, mutationRate):
    for swapped in range(len(individual)):
        if random.random() < mutationRate :
            swapWith = int(random.random() * len(individual))

            city1 = individual[swapped]
            city2 = individual[swapWith]

            individual[swapped] = city2
            individual[swapWith] = city1
    return individual


def mutatePopulation(population, mutationRate):
    mutatedPop = []

    for ind in range(0, len(population)):
        mutatedInd = mutate(population[ind], mutationRate)
        mutatedPop.append(mutatedInd)
    return mutatedPop



# Prepration of the next generation
def nextGeneration(currentGen, eliteSize, mutationRate):
    popRanked = rankRoutes(currentGen)
    selectionResults = selection(popRanked, eliteSize)
    matingpool = matingPool(currentGen, selectionResults)
    children = breedPopulation(matingpool, eliteSize)
    nextGeneration = mutatePopulation(children, mutationRate)
    return nextGeneration



# The main algorithm of GA
def geneticAlgorithm(population, popSize, eliteSize, mutationRate, generations):
    pop = initialPopulation(popSize, population)
    print("Initial distance: " + str(1 / rankRoutes(pop)[0][1]))

    for i in range(0, generations):
        pop = nextGeneration(pop, eliteSize, mutationRate)

    print("Final distance: " + str(1 / rankRoutes(pop)[0][1]))
    bestRouteIndex = rankRoutes(pop)[0][0]
    bestRoute = pop[bestRouteIndex]
    return bestRoute

# ------------------------------------------------------------------------------------------
# ---------------------
#  Simulated Annealing
# ---------------------



# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------- Main Program ---------------------------------------
# ------------------------------------------------------------------------------------------

main_graph_bruteforce = create_graph(nx)
main_graph_heuristic = create_graph(nx2)
# read_graphs(main_graph_bruteforce, main_graph_heuristic)
initialize_graphs(main_graph_bruteforce, main_graph_heuristic)
initialize_variables()
random_start_node = random.randrange(main_graph_bruteforce.number_of_nodes())


    # -----------------
    #  Exact execution
    # -----------------
time_begin_bruteforce = time.process_time_ns()

bruteforce(main_graph_bruteforce, random_start_node)

time_end_bruteforce = time.process_time_ns()
time_bruteforce = round(time_end_bruteforce - time_begin_bruteforce, 6)


    # ---------------------
    #  Heuristic execution
    # ---------------------
time_begin_heuristic = time.process_time_ns()  # Starting timer

# heuristic_execution = heuristic_nearest_neighboor(main_graph_heuristic, random_start_node)
# heuristic_execution = heuristic_random_selection(main_graph_heuristic)
# heuristic_execution = heuristic_random_walk(main_graph_heuristic, random_start_node)
heuristic_execution = heuristic_christofides(main_graph_heuristic)


cycle_heuristic = heuristic_execution[0]
cost_heuristic = heuristic_execution[1]
time_heuristic = heuristic_execution[2]


    # ------------------
    #  Printing results
    # ------------------
draw_graph(main_graph_bruteforce, cycle_optimal[1:], nx)
draw_graph(main_graph_heuristic, cycle_heuristic, nx2)

print("\n\nOptimal Cycle : " + str(cycle_optimal[1:]))
print("Cost : " + str(cycle_optimal[0]))
print("Time : " + str(round(time_bruteforce/10**6, 1)) + " ms = " +
                            str(round(time_bruteforce, 1)) + " ns")


print("\n\nEstimated Cycle : " + str(cycle_heuristic))
print("Cost : " + str(cost_heuristic))
print("Time : " + repr(round(time_heuristic/10**6, 1)) + " ms = " +
                                str(round(time_heuristic, 1)) + " ns")

print("\n\n - Estimation is " + str(round(cost_heuristic / cycle_optimal[0], 1))
      + " further from optimal\n\n")


# ---------------------------------------------------------------------------------------
