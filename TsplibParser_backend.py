import collections
import functools
import math
import itertools
import networkx


VALUE_TYPES = {
    'NAME': str,
    'TYPE': str,
    'COMMENT': str,
    'DIMENSION': int,
    'CAPACITY': int,
    'EDGE_WEIGHT_TYPE': str,
    'EDGE_WEIGHT_FORMAT': str,
    'EDGE_DATA_FORMAT': str,
    'NODE_COORD_TYPE': str,
    'DISPLAY_DATA_TYPE': str,
}


class Stream:
    def __init__(self, lines):
        self.lines = iter(lines)
        self.line = self._get_next()

    def __next__(self):
        self.line = self._get_next()
        return self.line

    def _get_next(self):
        try:
            line = ''
            while not line:
                line = next(self.lines).strip()
        except StopIteration:
            return None
        return line


def get_next_tour(sequence):
    tour = []
    while sequence:
        index = sequence.pop(0)

        if index == -1:
            if sequence == [-1]:
                sequence.pop(0)
            return tour

        tour.append(index)

    raise Exception('all tours must end with -1')


def read_integer_sequence(stream):
    while True:
        try:
            yield from map(int, stream.line.split())
            next(stream)
        except (ValueError, AttributeError):
            break


def partition(values, lengths):
    edge_weights = []
    for n in lengths:
        if n > len(values):
            raise Exception('too few values')
        row, values = values[:n], values[n:]
        edge_weights.append(row)

    if values:
        raise Exception('too many values')

    return edge_weights


def read_input_file(filename):
    with open(filename) as f:
        lines = f.read().splitlines()
    return lines


def split_kv(line):
    k, v = line.split(':', 1)
    return k.strip(), v.strip()


def parse(filename):
    lines = read_input_file(filename)
    stream = Stream(lines)
    data = {}

    transition = start
    while transition:
        transition = transition(data, stream)

    return data


def start(data, stream):
    next(stream)
    return process_line


def finish(data, stream):
    return None


def process_line(data, stream):
    if stream.line is None or stream.line == 'EOF':
        return finish

    if ':' in stream.line:
        return process_key_value
    else:
        return process_key


def process_key_value(data, stream):
    key, value = split_kv(stream.line)
    data[key] = VALUE_TYPES[key](value)
    next(stream)
    return process_line


def process_key(data, stream):
    key = stream.line
    next(stream)
    return {
        'NODE_COORD_SECTION': parse_node_coords,
        'DEPOT_SECTION': parse_depots,
        'DEMAND_SECTION': parse_demands,
        'EDGE_DATA_SECTION': parse_edge_data,
        'FIXED_EDGES_SECTION': parse_fixed_edges,
        'DISPLAY_DATA_SECTION': parse_display_data,
        'TOUR_SECTION': parse_tours,
        'EDGE_WEIGHT_SECTION': parse_edge_weights,
    }[key]


def parse_node_coords(data, stream):
    section = data['NODE_COORD_SECTION'] = collections.OrderedDict()

    while True:
        if stream.line is None:
            break

        index, *reals = stream.line.split()
        try:
            index = int(index)
        except ValueError:
            break

        if len(reals) not in (2, 3):
            raise Exception('invalid node coord')

        coord = tuple(map(float, reals))
        section[index] = coord
        next(stream)

    return process_line


def parse_depots(data, stream):
    section = data['DEPOT_SECTION'] = []

    while True:
        if stream.line is None:
            raise Exception('depot section must end with -1')

        try:
            depot = int(stream.line)
        except ValueError:
            raise Exception('invalid depot')

        if depot == -1:
            break

        section.append(depot)
        next(stream)

    next(stream)
    return process_line


def parse_demands(data, stream):
    section = data['DEMAND_SECTION'] = {}

    while True:
        if stream.line is None:
            break

        try:
            index, demand = stream.line.split()
        except ValueError:
            break

        try:
            index, demand = int(index), int(demand)
        except ValueError:
            break

        section[index] = demand
        next(stream)

    return process_line


def parse_edge_data(data, stream):
    edge_format = data['EDGE_DATA_FORMAT']
    return {
        'EDGE_LIST': parse_edge_list,
        'ADJ_LIST': parse_adj_list,
    }[edge_format]


def parse_edge_list(data, stream):
    section = data['EDGE_DATA_SECTION'] = []

    while True:
        if stream.line is None:
            raise Exception('edge list must end with a -1')

        try:
            u, v = stream.line.split()
        except ValueError:
            break

        try:
            edge = int(u), int(v)
        except ValueError:
            raise Exception('bad edge')

        section.append(edge)
        next(stream)

    if stream.line != '-1':
        raise Exception('edge list must end with a -1')

    next(stream)
    return process_line


def parse_adj_list(data, stream):
    section = data['EDGE_DATA_SECTION'] = collections.OrderedDict()

    while True:
        if stream.line is None:
            raise Exception('entire adjacency list must end with a -1')

        *values, end = stream.line.split()
        if end != '-1':
            raise Exception('adjacency list must end with a -1')
        if not values:
            break

        node, *neighbors = map(int, values)
        section[node] = neighbors
        next(stream)

    next(stream)
    return process_line


def parse_fixed_edges(data, stream):
    section = data['FIXED_EDGES_SECTION'] = []

    while True:
        if stream.line is None:
            raise Exception('fixed edges must end with a -1')

        try:
            u, v = stream.line.split()
        except ValueError:
            break

        try:
            edge = int(u), int(v)
        except ValueError:
            raise Exception('bad fixed edge')

        section.append(edge)
        next(stream)

    if stream.line != '-1':
        raise Exception('fixed edges must end with a -1')

    next(stream)
    return process_line


def parse_display_data(data, stream):
    section = data['DISPLAY_DATA_SECTION'] = collections.OrderedDict()

    while True:
        if stream.line is None:
            break

        index, *reals = stream.line.split()
        try:
            index = int(index)
        except ValueError:
            break

        if len(reals) not in (2, 3):
            raise Exception('invalid display data')

        coord = tuple(map(float, reals))
        section[index] = coord
        next(stream)

    return process_line


def parse_tours(data, stream):
    section = data['TOUR_SECTION'] = []

    sequence = list(read_integer_sequence(stream))
    while sequence:
        tour = get_next_tour(sequence)
        section.append(tour)

    return process_line


def parse_edge_weights(data, stream):
    data['EDGE_WEIGHT_SECTION'] = list(read_integer_sequence(stream))
    return process_line



class utils:
    def load_problem(filepath, special=None):
        """Load a problem at the given filepath.

        :param str filepath: path to a TSPLIB problem file
        :param callable special: special/custom distance function
        :return: problem instance
        :rtype: :class:`~Problem`
        """
        data = parse(filepath)
#         print(data)
        return models.Problem(special=special, **data)


    def load_solution(filepath):
        """Load a solution at the given filepath.

        :param str filepath: path to a TSPLIB solution file
        :return: solution instance
        :rtype: :class:`~Solution`
        """
        data = parse(filepath)
        return models.Solution(**data)


    def load_unknown(filepath):
        """Load a TSPLIB file.

        :param str filepath: path to a TSPLIB problem file
        :return: either a problem or solution instance
        """
        data = parse(filepath)
        if data['TYPE'] == 'TOUR':
            return models.Solution(**data)
        return models.Problem(**data)


    def parse_degrees(coord):
        degrees = utils.nint(coord)
        minutes = coord - degrees
        return degrees + minutes * 5 / 3


    def nint(x):
        return int(x + 0.5)


    def icost(x):
        return int(100 * x + 0.5)


    def deltas(start, end):
        return (e - s for e, s in zip(end, start))


    class RadianGeo:
        def __init__(self, coord):
            x, y = coord
            self.lat = self.__class__.parse_component(x)
            self.lng = self.__class__.parse_component(y)

        @staticmethod
        def parse_component(component):
            return math.radians(utils.parse_degrees(component))


    def _int_sum(n, memo={}):
        if n not in memo:
            s = n * (n + 1) // 2
            memo[n] = s
        return memo[n]


    def integer_sum(n, m=None):
        s = utils._int_sum(n)
        if m:
            s -= utils._int_sum(m)
        return s


    def pairwise(indexes):
        starts = list(indexes)
        ends = list(indexes)
        ends += [ends.pop(0)]
        return zip(starts, ends)

class matrix:
        class Matrix:
            """A square matrix created from a list of numbers.

            Elements are accessible using matrix notation. Negative indexing is not
            allowed.

            :param list numbers: the elements of the matrix
            :param int size: the width (also height) of the matrix
            :param int min_index: the minimum index
            """

            def __init__(self, numbers, size, min_index=0):
                self.numbers = list(numbers)
                self.size = size
                self.min_index = min_index

            def __getitem__(self, key):
                return self.value_at(*key)

            def value_at(self, i, j):
                """Get the element at row *i* and column *j*.

                :param int i: row
                :param int j: column
                :return: value of element at (i,j)
                """
                i -= self.min_index
                j -= self.min_index
                if not self.is_valid_row_column(i, j):
                    raise IndexError(f'({i}, {j}) is out of bonuds')
                index = self.get_index(i, j)
                return self.numbers[index]

            def is_valid_row_column(self, i, j):
                """Return True if (i,j) is a row and column within the matrix.

                :param int i: row
                :param int j: column
                :return: whether (i,j) is within the bounds of the matrix
                :rtype: bool
                """
                return 0 <= i < self.size and 0 <= j < self.size

            def get_index(self, i, j):
                """Return the linear index for the element at (i,j).

                :param int i: row
                :param int j: column
                :return: linear index for element (i,j)
                :rtype: int
                """
                raise NotImplementedError()

        class FullMatrix(Matrix):
            """A complete square matrix.

            :param list numbers: the elements of the matrix
            :param int size: the width (also height) of the matrix
            :param int min_index: the minimum index
            """

            def get_index(self, i, j):
                return i * self.size + j

        class HalfMatrix(Matrix):
            """A triangular half-matrix.

            :param list numbers: the elements of the matrix
            :param int size: the width (also height) of the matrix
            :param int min_index: the minimum index
            """

            #: True if the half-matrix includes the diagonal
            has_diagonal = True

            def value_at(self, i, j):
                if i == j and not self.has_diagonal:
                    return 0
                i, j = self._fix_indices(i, j)
                return super().value_at(i, j)

        class UpperDiagRow(HalfMatrix):
            """Upper-triangular matrix that includes the diagonal.

            :param list numbers: the elements of the matrix
            :param int size: the width (also height) of the matrix
            :param int min_index: the minimum index
            """

            has_diagonal = True

            def _fix_indices(self, i, j):
                i, j = (j, i) if i > j else (i, j)
                if not self.has_diagonal:
                    j -= 1
                return i, j

            def get_index(self, i, j):
                n = self.size - int(not self.has_diagonal)
                return utils.integer_sum(n, n - i) + (j - i)

        class LowerDiagRow(HalfMatrix):
            """Lower-triangular matrix that includes the diagonal.

            :param list numbers: the elements of the matrix
            :param int size: the width (also height) of the matrix
            :param int min_index: the minimum index
            """

            has_diagonal = True

            def _fix_indices(self, i, j):
                i, j = (j, i) if i < j else (i, j)
                if not self.has_diagonal:
                    i -= 1
                return i, j

            def get_index(self, i, j):
                return utils.integer_sum(i) + j

        class UpperRow(UpperDiagRow):
            """Upper-triangular matrix that does not include the diagonal.

            :param list numbers: the elements of the matrix
            :param int size: the width (also height) of the matrix
            :param int min_index: the minimum index
            """

            has_diagonal = False

        class LowerRow(LowerDiagRow):
            """Lower-triangular matrix that does not include the diagonal.

            :param list numbers: the elements of the matrix
            :param int size: the width (also height) of the matrix
            :param int min_index: the minimum index
            """

            has_diagonal = False

        class UpperCol(LowerRow):
            pass

        class LowerCol(UpperRow):
            pass

        class UpperDiagCol(LowerDiagRow):
            pass

        class LowerDiagCol(UpperDiagRow):
            pass

        TYPES = {
            'FULL_MATRIX': FullMatrix,
            'UPPER_DIAG_ROW': UpperDiagRow,
            'UPPER_ROW': UpperRow,
            'LOWER_DIAG_ROW': LowerDiagRow,
            'LOWER_ROW': LowerRow,
            'UPPER_DIAG_COL': UpperDiagCol,
            'UPPER_COL': UpperCol,
            'LOWER_DIAG_COL': LowerDiagCol,
            'LOWER_COL': LowerCol,
        }

class distances:
    def euclidean(start, end, round=utils.nint):
        """Return the Euclidean distance between start and end.

        :param tuple start: *n*-dimensional coordinate
        :param tuple end: *n*-dimensional coordinate
        :param callable round: function to use to round the result
        :return: rounded distance
        """
        if len(start) != len(end):
            raise ValueError('dimension mismatch between start and end')

        square_distance = sum(d * d for d in utils.deltas(start, end))
        distance = math.sqrt(square_distance)

        return round(distance)


    def manhattan(start, end, round=utils.nint):
        """Return the Manhattan distance between start and end.

        :param tuple start: *n*-dimensional coordinate
        :param tuple end: *n*-dimensional coordinate
        :param callable round: function to use to round the result
        :return: rounded distance
        """
        if len(start) != len(end):
            raise ValueError('dimension mismatch between start and end')

        distance = sum(abs(d) for d in utils.deltas(start, end))

        return round(distance)


    def maximum(start, end, round=utils.nint):
        """Return the Maximum distance between start and end.

        :param tuple start: *n*-dimensional coordinate
        :param tuple end: *n*-dimensional coordinate
        :param callable round: function to use to round the result
        :return: rounded distance
        """
        if len(start) != len(end):
            raise ValueError('dimension mismatch between start and end')

        distance = max(abs(d) for d in utils.deltas(start, end))

        return round(distance)


    def geographical(start, end, round=utils.nint, diameter=6378.388):
        """Return the geographical distance between start and end.

        :param tuple start: *n*-dimensional coordinate
        :param tuple end: *n*-dimensional coordinate
        :param callable round: function to use to round the result
        :param float diameter: the diameter of the Earth
        :return: rounded distance
        """
        if len(start) != len(end):
            raise ValueError('dimension mismatch between start and end')

        start = utils.RadianGeo(start)
        end = utils.RadianGeo(end)

        q1 = math.cos(start.lng - end.lng)
        q2 = math.cos(start.lat - end.lat)
        q3 = math.cos(start.lat + end.lat)
        distance = diameter * math.acos(0.5 * ((1 + q1) * q2 - (1 - q1) * q3)) + 1

        return round(distance)


    def pseudo_euclidean(start, end, round=utils.nint):
        """Return the pseudo-Euclidean distance between start and end.

        :param tuple start: *n*-dimensional coordinate
        :param tuple end: *n*-dimensional coordinate
        :param callable round: function to use to round the result
        :return: rounded distance
        """
        if len(start) != len(end):
            raise ValueError('dimension mismatch between start and end')

        square_sum = sum(d * d for d in utils.deltas(start, end))
        value = math.sqrt(square_sum / 10)

        # with nint does this not equate to ceil? and what about other cases?
        distance = round(value)
        if distance < value:
            distance += 1
        return distance


    def xray(start, end, sx=1, sy=1, sz=1, round=utils.icost):
        """Return x-ray crystallography distance.

        :param tuple start: 3-dimensional coordinate
        :param tuple end: 3-dimensional coordinate
        :param float sx: x motor speed
        :param float sy: y motor speed
        :param float sz: z motor speed
        :return: distance
        """
        if len(start) != len(end) or len(start) != 3:
            raise ValueError('start and end but be 3-dimensional')

        dx = min(abs(start[0] - end[0]), abs(abs(start[0] - end[0])) - 360)
        dy = abs(start[1] - end[1])
        dz = abs(start[2] - end[2])
        distance = max(dx / sx, dy / sy, dz / sz)

        return round(distance)


    TYPES = {
        'EUC_2D': euclidean,
        'EUC_3D': euclidean,
        'MAX_2D': maximum,
        'MAX_3D': maximum,
        'MAN_2D': manhattan,
        'MAN_3D': manhattan,
        'CEIL_2D': functools.partial(euclidean, round=math.ceil),
        'GEO': euclidean,
        'ATT': euclidean,
        'XRAY1': xray,
        'XRAY2': functools.partial(xray, sx=1.25, sy=1.5, sz=1.15),
    }

class models:
    class File:
        """Base file format type.

        This class isn't meant to be used directly. It contains the common keyword
        values common among all formats. Note that all information is optional. In
        that case the value will be None. See the official TSPLIB_ documentation
        for more details.

        .. _TSPLIB: https://www.iwr.uni-heidelberg.de/groups/comopt/software/TSPLIB95/index.html
        """

        def __init__(self, **kwargs):
            self.name = kwargs.get('NAME')
            self.comment = kwargs.get('COMMENT')
            self.type = kwargs.get('TYPE')
            self.dimension = kwargs.get('DIMENSION')


    class Solution(File):
        """A TSPLIB solution file containing one or more tours to a problem.

        The length of a solution is the number of tours it contains.
        """

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.tours = kwargs.get('TOUR_SECTION')

        def __len__(self):
            return len(self.tours)


    class Problem(File):
        """A TSPLIB problem file.

        For problems that require a special distance function, you must set the
        special function in one of two ways:

        .. code-block:: python

            >>> problem = Problem(special=func, ...)  # at creation time
            >>> problem.special = func                # on existing problem

        Special distance functions are ignored for explicit problems but are
        required for some.

        Regardless of problem type or specification, the weight of the edge between
        two nodes given by index can always be found using ``wfunc``. For example,
        to get the weight of the edge between nodes 13 and 6:

        .. code-block:: python

            >>> problem.wfunc(13, 6)
            87

        The length of a problem is the number of nodes it contains.
        """

        def __init__(self, special=None, **kwargs):
            super().__init__(**kwargs)
            self.capacity = kwargs.get('CAPACITY')

            # specification
            self.edge_weight_type = kwargs.get('EDGE_WEIGHT_TYPE')
            self.edge_weight_format = kwargs.get('EDGE_WEIGHT_FORMAT')
            self.edge_data_format = kwargs.get('EDGE_DATA_FORMAT')
            self.node_coord_type = kwargs.get('NODE_COORD_TYPE')
            self.display_data_type = kwargs.get('DISPLAY_DATA_TYPE')

            # data
            self.depots = kwargs.get('DEPOT_SECTION')
            self.demands = kwargs.get('DEMAND_SECTION')
            self.node_coords = kwargs.get('NODE_COORD_SECTION')
            #print(self.node_coords)
            self.edge_weights = kwargs.get('EDGE_WEIGHT_SECTION')
            self.display_data = kwargs.get('DISPLAY_DATA_SECTION')
            self.edge_data = kwargs.get('EDGE_DATA_SECTION')
            self.fixed_edges = kwargs.get('FIXED_EDGES_SECTION', set())

            self.wfunc = None
            self.special = special
        def get_coords_dict(self):
            return self.node_coords
        def __len__(self):
            return self.dimension

        @property
        def special(self):
            """Special distance function"""
            return self._special

        @special.setter
        def special(self, func):
            """Set the special distance function.

            Special/custom distance functions must accept two coordinates of
            appropriate dimension and return the distance between them.

            Note that this has no effect if the problem defines weights explicitly.

            :param callable func: custom distance function
            """
            self._special = func
            self.wfunc = self._create_wfunc(special=func)

        def is_explicit(self):
            """Return True if the problem specifies explicit edge weights.

            :rtype: bool
            """
            return self.edge_weight_type == 'EXPLICIT'

        def is_full_matrix(self):
            """Return True if the problem is specified as a full matrix.

            :rtype: bool
            """
            return self.edge_weight_format == 'FULL_MATRIX'

        def is_weighted(self):
            """Return True if the problem has weighted edges.

            :rtype: bool
            """
            return bool(self.edge_weight_format) or bool(self.edge_weight_type)

        def is_special(self):
            """Return True if the problem requires a special distance function.

            :rtype: bool
            """
            return self.edge_weight_type == 'SPECIAL'

        def is_complete(self):
            """Return True if the problem specifies a complete graph.

            :rtype: bool
            """
            return not bool(self.edge_data_format)

        def is_symmetric(self):
            """Return True if the problem is not asymmetrical.

            Note that even if this method returns False there is no guarantee that
            there are any two nodes with an asymmetrical distance between them.

            :rtype: bool
            """
            return not self.is_full_matrix() and not self.is_special()

        def is_depictable(self):
            """Return True if the problem is designed to be depicted.

            :rtype: bool
            """
            if bool(self.display_data):
                return True

            if self.display_data_type == 'NO_DISPLAY':
                return False

            return bool(self.node_coords)

        def trace_tours(self, solution):
            """Calculate the total weights of the tours in the given solution.

            :param solution: solution with tours to trace
            :type solution: :class:`~Solution`
            :return: one or more tour weights
            :rtype: list
            """
            solutions = []
            for tour in solution.tours:
                weight = sum(self.wfunc(i, j) for i, j in utils.pairwise(tour))
                solutions.append(weight)
            return solutions

        def _create_wfunc(self, special=None):
            # smooth out the differences between explicit and calculated problems
            if self.is_explicit():
                matrix = self._create_explicit_matrix()
                return lambda i, j: matrix[i, j]
            else:
                return self._create_distance_function(special=special)

        def _create_distance_function(self, special=None):
            # wrap a distance function so that it takes node indexes, not coords
            if self.is_special():
                if special is None:
                    raise Exception('missing needed special weight function')
                wfunc = special
            elif self.is_weighted():
                wfunc = distances.TYPES[self.edge_weight_type]
            else:
                return lambda i, j: 1

            def adapter(i, j):
                return wfunc(self.node_coords[i], self.node_coords[j])

            return adapter

        def _create_explicit_matrix(self):
            # instantiate the right matrix class for the problem
            m = min(self.get_nodes())
            Matrix = matrix.TYPES[self.edge_weight_format]
            return Matrix(self.edge_weights, self.dimension, min_index=m)

        def get_nodes(self):
            """Return an iterator over the nodes.

            :return: nodes
            :rtype: iter
            """
            if self.node_coords:
                return iter(self.node_coords)
            elif self.display_data:
                return iter(self.display_data)
            else:
                return iter(range(self.dimension))

        def get_edges(self):
            """Return an iterator over the edges.

            :return: edges
            :rtype: iter
            """
            if self.edge_data_format == 'EDGE_LIST':
                yield from self.edge_data
            elif self.edge_data_format == 'ADJ_LIST':
                for i, adj in self.edge_data.items():
                    yield from ((i, j) for j in adj)
            else:
                yield from itertools.product(self.get_nodes(), self.get_nodes())

        def get_display(self, i):
            """Return the display data for node at index *i*, if available.

            :param int i: node index
            :return: display data for node i
            """
            if self.is_depictable():
                try:
                    return self.display_data[i]
                except TypeError:
                    return self.node_coords[i]
            else:
                return None

        def get_graph(self):
            """Return the corresponding networkx.Graph instance.

            If the graph is not symmetric then a DiGraph is returned. If present,
            the coordinates of each node are set to the ``coord`` key, and each
            edge has an ``is_fixed`` key that is True if the edge is in the list
            of fixed edges.

            :return: graph
            """
            G = networkx.Graph() if self.is_symmetric() else networkx.DiGraph()
            G.graph['name'] = self.name
            G.graph['comment'] = self.comment
            G.graph['type'] = self.type
            G.graph['dimension'] = self.dimension
            G.graph['capacity'] = self.capacity
            G.graph['depots'] = self.depots
            G.graph['demands'] = self.demands
            G.graph['fixed_edges'] = self.fixed_edges

            if not self.is_explicit():
                for i, coord in self.node_coords.items():
                    G.add_node(i, coord=coord)

            for i, j in self.get_edges():
                weight = self.wfunc(i, j)
                is_fixed = (i, j) in self.fixed_edges
                G.add_edge(i, j, weight=weight, is_fixed=is_fixed)

            return G