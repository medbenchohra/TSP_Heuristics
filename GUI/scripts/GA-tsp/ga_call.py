import ga
import random
import os
import sys
sys.path.append('scripts/tsplib-parser')
import json
import TsplibParser
import time

params=json.loads(sys.argv[1])

coords=[]
instance=TsplibParser.load_instance("scripts/tsp-dataset/"+params["fileName"],None)

coords = instance.get_nodes_coord()


cityList = []

for i in range(0,len(coords)):
    coord = coords[i]
    cityList.append(ga.City(node=i , x=coord[1] , y= coord[2] ))



if __name__ == "__main__":
    
    t1=time.time()
    cost=ga.geneticAlgorithm(cityList, popSize=int(params["popSize"]),eliteSize = int(params["eliteSize"]), mutationRate=float(params["mutationRate"]), generations = int(params["generations"]))

    t2=time.time()
    print(json.dumps({'execTime': t2-t1, 'pathCost': cost, 'solution':"TooLongToShow"}, separators=(',', ': ')))
    sys.stdout.flush()