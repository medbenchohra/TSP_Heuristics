import Lin_Keringhan as LK
import random
import os
import sys
#sys.path.append('scripts/tsplib-parser')
import json
#import TsplibParser
import time
import subprocess
import re

params=json.loads(sys.argv[1])

#coords=[]
#instance=TsplibParser.load_instance("scripts/tsp-dataset/"+params["fileName"],None)
#instance=TsplibParser.load_instance("scripts/tsp-dataset/a280.tsp",None)
#coords = instance.get_nodes_coord()

if __name__ == "__main__":
    #sa = SimAnneal(coords, stopping_iter=int(params["iterations"]),stopping_T=int(params["temperature"]))
    #sa.anneal()
    
    t1=time.time()
    #cost=LK.Lin_Keringhan(coords,depth=3)
    with open('LKH_call.par','w') as myFile:
        myFile.write('PROBLEM_FILE = '+"../tsp-dataset/"+params["fileName"]")
	
    result = subprocess.check_output(['./LKH', 'LKH_call.par'])
    cost = re.search('Cost.min = (.*), Cost.avg =', str(result))
    #print(str(result))
    print(cost.group(1))
    t2=time.time()
    print(json.dumps({'execTime': t2-t1, 'pathCost': cost, 'solution':"TooLongToShow"}, separators=(',', ': ')))
    sys.stdout.flush()
    
    



    
         

