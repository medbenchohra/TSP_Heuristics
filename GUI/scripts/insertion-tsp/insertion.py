import json
import time
import collections


# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
# Insertion
# ----------

def heuristic_insertion(N):

    M = N.todense().tolist()
    time_begin = time.process_time()
    visited = []
    path = []
    min = None
    for i in range(1, len(M)):
        for j in range(i + 1, len(M)):
            if min is None:
                min = M[i][j]
                i_min = i
                j_min = j
            elif M[i][j] < min:
                min = M[i][j]
                i_min = i
                j_min = j
    visited.append(i_min)
    visited.append(j_min)
    path.append((i_min, j_min))

    for n in range(len(M)):
        min = None
        if not n in visited:
            for v in visited:
                if min is None:
                    min = M[n][v]
                    v1 = v
                elif M[n][v] < min:
                    min = M[n][v]
                    v1 = v
            min2 = None
            for v in visited:
                if (v, v1) in path or (v1, v) in path:
                    if min2 is None:
                        min2 = min + M[n][v]
                        v2 = v
                    elif M[n][v] + min < min2:
                        min2 = M[n][v] + min
                        v2 = v

            if (v1, v2) in path:
                path.remove((v1, v2))
            else:
                path.remove((v2, v1))
            path.append((v1, n))
            path.append((v2, n))
            visited.append(n)
    x = []
    for v in path:
        x.append(v[0])
        x.append(v[1])
    a = collections.Counter(x)
    final = []
    for k, v in a.items():
        if v == 1:
            final.append(k)

    path.append((final[0], final[1]))

    time_end = time.process_time()
    exec_time = round(time_end - time_begin, 6)
    s = 0
    for v in path:
        s = s + M[v[0]][v[1]]

    print(json.dumps({'execTime': exec_time, 'pathCost': s, 'solution': "To large !!"}, separators=(',', ': ')))
    return exec_time, s
