from anneal import SimAnneal
import matplotlib.pyplot as plt
import random
import os
import sys
sys.path.append('scripts/tsplib-parser')
import json
import TsplibParser

params=json.loads(sys.argv[1])

coords=[]
instance=TsplibParser.load_instance("scripts/tsp-dataset/"+params["fileName"],None)
coords = instance.get_nodes_coord()

if __name__ == "__main__":
    sa = SimAnneal(coords, stopping_iter=int(params["iterations"]),stopping_T=int(params["temperature"]))
    sa.anneal()
    sys.stdout.flush()


