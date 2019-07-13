from greedy import heuristic_greedy
import sys
sys.path.append('scripts/tsplib-parser')
import json
import TsplibParser

params=json.loads(sys.argv[1])


instance=TsplibParser.load_instance("scripts/tsp-dataset/"+params["fileName"],None)
G = instance.get_nx_graph()

if __name__ == "__main__":
    sa = heuristic_greedy(G)
    sys.stdout.flush()