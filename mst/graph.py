import numpy as np
import heapq
from typing import Union

class Graph:
    def __init__(self, adjacency_mat: Union[np.ndarray, str]):
        """ Unlike project 2, this Graph class takes an adjacency matrix as input. `adjacency_mat` 
        can either be a 2D numpy array of floats or the path to a CSV file containing a 2D numpy array of floats.

        In this project, we will assume `adjacency_mat` corresponds to the adjacency matrix of an undirected graph
        """
        if type(adjacency_mat) == str:
            self.adj_mat = self._load_adjacency_matrix_from_csv(adjacency_mat)
        elif type(adjacency_mat) == np.ndarray:
            self.adj_mat = adjacency_mat
        else: 
            raise TypeError('Input must be a valid path or an adjacency matrix')
        self.mst = None

    def _load_adjacency_matrix_from_csv(self, path: str) -> np.ndarray:
        with open(path) as f:
            return np.loadtxt(f, delimiter=',')

    def construct_mst(self):
        """ Given `self.adj_mat`, the adjacency matrix of a connected undirected graph, implement Prim's 
        algorithm to construct an adjacency matrix encoding the minimum spanning tree of `self.adj_mat`. 
            
        `self.adj_mat` is a 2D numpy array of floats. 
        Note that because we assume our input graph is undirected, `self.adj_mat` is symmetric. 
        Row i and column j represents the edge weight between vertex i and vertex j. An edge weight of zero indicates that no edge exists. 
        
        TODO: 
            This function does not return anything. Instead, store the adjacency matrix 
        representation of the minimum spanning tree of `self.adj_mat` in `self.mst`.
        We highly encourage the use of priority queues in your implementation. See the heapq
        module, particularly the `heapify`, `heappop`, and `heappush` functions.
        """

        # initialize empty mst adj matrix
        mst = np.zeros_like(self.adj_mat)

        # start at node 0, get out edges from starting node
        heap = self.get_edge_from_numpy_arr(0, self.adj_mat[0])
        # heap used as priority queue
        heapq.heapify(heap)

        # initialize set for tracking visitation
        seen_list = set()
        seen_list.add(0)

        while len(heap) > 0:
            # heap is not empty
            curr = heapq.heappop(heap)

            if curr.to not in seen_list:
                # destination node has not been seen
                seen_list.add(curr.to)

                # connect current to dest in mst symmetrically,
                # as it is the minimum weight edge
                mst[[curr.frm, curr.to],
                    [curr.to, curr.frm]] = curr.weight

                # construct out-edges from current node
                for out_edge in self.get_edge_from_numpy_arr(curr.to, self.adj_mat[curr.to]):
                    # add to heap if not seen
                    if out_edge.to not in seen_list:
                        heapq.heappush(heap, out_edge)

        # check that all nodes are connected to another node, if not,
        # leave self.mst as None
        if not np.any(np.sum(mst, axis=0) == 0):
            self.mst = mst

        return

    @staticmethod
    def get_edge_from_numpy_arr(frm: int, out_arr: np.array):
        """
        Static method to construct Edges from a given node/index
        to all other connected via some adjacency matrix.
        This method assumes the indices of the out_arr correspond
        to the node numbers
        Args:
            frm: source node/index (the i-th row or j-th column which is == out_arr)
            out_arr: array of outward connected nodes with zeros if disconnected else weights

        Returns: list of Edge objects

        """
        # get indices/nodes with connected edge (non-zero)
        out_idx = np.argwhere(out_arr)
        # construct Edge object for each connected node and non-zero weight
        edges = [Edge(frm, dest.item(), weight.item())
                 for dest, weight in zip(out_idx, out_arr[out_idx])]

        return edges


class Edge:
    """
    Class to store edge connections and weight for sorting
    """

    def __init__(self, frm: int, to: int, weight: float):
        """

        Args:
            frm: int - source node
            to: int - dest node
            weight: float - edge weight
        """
        self._frm = frm
        self._to = to
        self._weight = weight

    def __eq__(self, other):
        """
        Override equals (might not be necessary as lt should cover queue addition)
        Args:
            other: Edge

        Returns: Bool

        """
        return other.weight == self._weight

    def __lt__(self, other):
        """
        Overrides lt for queue/heap addition
        Args:
            other: Edge

        Returns: Bool

        """
        return self._weight < other.weight

    def __gt__(self, other):
        """
        Overrides gt for queue / heap addition
        Args:
            other: Edge

        Returns: Bool

        """
        return self._weight > other.weight

    def __repr__(self):
        """
        Override repr for debugging
        Returns:

        """
        return f"({self._frm})--{self._weight}--({self._to})"

    @property
    def weight(self):
        """
        Getter method/prop
        Returns: float

        """
        return self._weight

    @property
    def frm(self):
        """
        Getter method/prop
        Returns: int

        """
        return self._frm

    @property
    def to(self):
        """
       Getter method/prop
       Returns: int

       """
        return self._to
