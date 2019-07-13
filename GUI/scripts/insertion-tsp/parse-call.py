from insertion import heuristic_insertion
import sys
sys.path.append('scripts/tsplib-parser')
import json
import TsplibParser

params=json.loads(sys.argv[1])


instance=TsplibParser.load_instance("scripts/tsp-dataset/"+params["fileName"],None)
M = instance.get_adj_matrix()

if __name__ == "__main__":
    sa = heuristic_insertion(M)
    sys.stdout.flush()