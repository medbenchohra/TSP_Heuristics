import os
import sys
sys.path.append('./greedy-tsp')
sys.path.append('scripts/tsplib-parser')
from greedy import heuristic_greedy
import TsplibParser

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

#==============================================================
