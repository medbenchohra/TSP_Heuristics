import matplotlib.pyplot as plt
import random
import os
import sys
import time
sys.path.append('scripts/tsplib-parser')
import json
import TsplibParser

params=json.loads(sys.argv[1])

coords=[]
instance=TsplibParser.load_instance("scripts/tsp-dataset/"+params["fileName"],None)
adj_matrix = instance.get_adj_matrix()
init_route = list(range(adj_matrix.shape[0]))
init_route.append(0)

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

    return cost

if __name__ == "__main__":
    t1=time.time()
    cost=heuristic_two_opt(init_route,adj_matrix)
    t2=time.time()
    print(json.dumps({'execTime': t2-t1, 'pathCost': cost, 'solution':"TooLongToShow"}, separators=(',', ': ')))
    sys.stdout.flush()



    