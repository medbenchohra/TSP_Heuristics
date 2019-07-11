import TsplibParser_backend as tspb
import networkx

def load_instance(Problemefilenalme,SolutionFileName):
    if SolutionFileName==None:
        return Instance(tspb.utils.load_problem(Problemefilenalme), None )
    return Instance(tspb.utils.load_problem(Problemefilenalme), tspb.utils.load_solution(SolutionFileName))
class Instance:
    def __init__(self, problem, solution):
        self.problem = problem
        self.solution = solution
        self.nx_graph = problem.get_graph()
        self.adj_matrix = networkx.adjacency_matrix(problem.get_graph()).todense()
    def get_nx_graph(self):
        return self.nx_graph
    def get_adj_matrix(self):
        return self.adj_matrix
    def get_opt_tour(self):
        if self.solution==None:
            return None
        return self.solution.tours[0]
    def get_opt_cost(self):
        if self.solution==None:
            return None
        return self.problem.trace_tours(self.solution)[0]
    def is_metric(self):
        pass
    def get_nodes_coord(self):
        coords_tab = []
        coord_dict= self.problem.get_coords_dict()
#         print(coord_dict)
        for city, coords in coord_dict.items():
            coords_tab.append([city, coords[0], coords[1]])
        return coords_tab