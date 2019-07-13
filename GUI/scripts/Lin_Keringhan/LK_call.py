import Lin_Keringhan as LK
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

if __name__ == "__main__":
    #sa = SimAnneal(coords, stopping_iter=int(params["iterations"]),stopping_T=int(params["temperature"]))
    #sa.anneal()
    
    t1=time.time()
    cost=LK.Lin_Keringhan(coords,depth=3)
    t2=time.time()
    print(json.dumps({'execTime': t2-t1, 'pathCost': cost, 'solution':"TooLongToShow"}, separators=(',', ': ')))
    sys.stdout.flush()


