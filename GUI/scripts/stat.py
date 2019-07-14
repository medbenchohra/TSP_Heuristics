import os
import sys
sys.path.append('./greedy-tsp')
sys.path.append('scripts/tsplib-parser')
from greedy import heuristic_greedy
import TsplibParser
import time

instance_dir = "./stp-lib-copy/"

result = ['Greedy ;exec   ;cout   ;',]
for instance_file in os.listdir(instance_dir):
    instance = TsplibParser.load_instance(instance_dir + instance_file, None)
    G = instance.get_nx_graph()
    exec, cout = heuristic_greedy(G)
    result.append(instance_file+'   ;'+str(exec)+'   ;'+str(cout)+'   ;')

with open('greedy.csv', 'w') as f:
    for item in result:
        f.write("%s\n" % item)

#=============================================================

sys.path.append('./insertion-tsp')
from insertion import heuristic_insertion

result = ['Insertion ;exec   ;cout   ;',]
for instance_file in os.listdir(instance_dir):
    instance = TsplibParser.load_instance(instance_dir + instance_file, None)
    M = instance.get_adj_matrix()
    exec, cout = heuristic_insertion(M)
    result.append(instance_file+'   ;'+str(exec)+'   ;'+str(cout)+'   ;')

with open('insertion.csv', 'w') as f:
    for item in result:
        f.write("%s\n" % item)
#=============================================================

sys.path.append('./GA-tsp')
import ga


result = ['GA1 ;exec   ;cout   ;',]
for instance_file in os.listdir(instance_dir):
    instance = TsplibParser.load_instance(instance_dir + instance_file, None)
    coords = instance.get_nodes_coord()

    cityList = []

    for i in range(0, len(coords)):
        coord = coords[i]
        cityList.append(ga.City(node=i, x=coord[1], y=coord[2]))
    t1 = time.time()
    cout = ga.geneticAlgorithm(cityList, popSize=100,eliteSize = 30, mutationRate=0.01, generations =100)
    t2 = time.time()
    result.append(instance_file+'   ;'+str(t2-t1)+'   ;'+str(cout)+'   ;')

with open('ag2.csv', 'w') as f:
    for item in result:
        f.write("%s\n" % item)


result = ['GA2 ;exec   ;cout   ;',]
for instance_file in os.listdir(instance_dir):
    instance = TsplibParser.load_instance(instance_dir + instance_file, None)
    coords = instance.get_nodes_coord()

    cityList = []

    for i in range(0, len(coords)):
        coord = coords[i]
        cityList.append(ga.City(node=i, x=coord[1], y=coord[2]))
    t1 = time.time()
    cout = ga.geneticAlgorithm(cityList, popSize=50,eliteSize = 40, mutationRate=0.05, generations =50)
    t2 = time.time()
    result.append(instance_file+'   ;'+str(t2-t1)+'   ;'+str(cout)+'   ;')

with open('ag2.csv', 'w') as f:
    for item in result:
        f.write("%s\n" % item)
#==============================================================



result = ['Christofides ;exec   ;cout   ;',]
for instance_file in os.listdir(instance_dir):
    instance = TsplibParser.load_instance(instance_dir + instance_file, None)
    M = instance.get_adj_matrix()
    _, cout, exec = (M.todense().tolist())
    result.append(instance_file+'   ;'+str(exec)+'   ;'+str(cout)+'   ;')

with open('christofides.csv', 'w') as f:
    for item in result:
        f.write("%s\n" % item)

#================================================================