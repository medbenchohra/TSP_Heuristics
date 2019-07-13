import time
import random
import sys
import json



def heuristic_random_selection(g):
    path = []

    time_begin = time.time()  # Starting timer
    for i in range(g.number_of_nodes()):
        path.append(i)
    random.shuffle(path)
    cost = calculate_cost_from_path(g, path)
    time_end = time.time()  # Stopping timer

    time_exec = time_end - time_begin

    return path, cost, time_exec


def calculate_cost_from_path(G, path):
    # Path is noncyclic

    cost = 0
    for i in range(G.number_of_nodes()):
        cost += G[path[i-1]][path[i]]['weight']

    return cost

# --------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------

sys.path.append('scripts/tsplib-parser')
import TsplibParser
params=json.loads(sys.argv[1])

instance = TsplibParser.load_instance("scripts/tsp-dataset/"+params["fileName"],None)
graph = instance.get_nx_graph()


""" heuristic_execution = heuristic_random_selection(graph)
heuristic_path = heuristic_execution[0]
heuristic_cost = heuristic_execution[1]
heuristic_time = heuristic_execution[2]

print(json.dumps({'execTime': heuristic_time, 'pathCost': heuristic_cost}, separators=(',', ': '))) """
print(json.dumps({'execTime': 55, 'pathCost': 44,'solution':"4,3,2,1"}, separators=(',', ': '))) 
sys.stdout.flush()

