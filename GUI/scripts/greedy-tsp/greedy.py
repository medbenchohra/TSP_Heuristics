import json
import time
import collections
# --------------------------------------------------------------------------------------
# -----------
# 4 - Greedy
# -----------




def heuristic_greedy(G):
    def edge_weight(v):
        return G[v[0]][v[1]]['weight']

    def node_degree(n, edges):
        edges_list = []
        for i in edges:
            edges_list.append(i[0])
            edges_list.append(i[1])

        return edges_list.count(n)

    def cycle_inf(result):
        edges_list = []
        for i in result:
            edges_list.append(i[0])
            edges_list.append(i[1])
        a = collections.Counter(edges_list)
        if len(a.keys()) < len(G.nodes):
            x = 0
            for i in a.values():
                x = x + i
            if x == len(a.values()) * 2:
                return True

        return False

    time_begin = time.process_time()
    sorted_edges = []
    for e in G.edges:
        sorted_edges.append(e)

    sorted_edges.sort(key=edge_weight)

    result = []
    e = sorted_edges.pop(0)
    result.append(e)

    i = 1
    while i < len(G.nodes):
        e = sorted_edges.pop(0)
        result.append(e)
        if node_degree(e[0], result) > 2 or node_degree(e[1], result) > 2 or cycle_inf(result):
            e = result.pop()
        else:
            i = i + 1

    time_end = time.process_time()
    exec_time = round(time_end - time_begin, 6)

    cost = 0
    for v in result:
        cost += edge_weight(v)
    print(
        json.dumps({'execTime': exec_time, 'pathCost': cost, 'solution': "To large !! "}, separators=(',', ': ')))
    return exec_time, cost

