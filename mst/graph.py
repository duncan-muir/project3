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

        print(self.adj_mat)
        self.mst = np.zeros_like(self.adj_mat)

        # start at node 0
        heap = self.get_edge_from_numpy_arr(0, self.adj_mat[0])
        heapq.heapify(heap)

        # intiialize set for tracking visitation
        seen_list = set()
        seen_list.add(0)

        while len(heap) > 0:
            curr = heapq.heappop(heap)
            if curr.to not in seen_list:
                seen_list.add(curr.to)
                self.mst[[curr.frm, curr.to],
                         [curr.to, curr.frm]] = curr.weight
                for out_edge in self.get_edge_from_numpy_arr(curr.to, self.adj_mat[curr.to]):
                    if out_edge.to not in seen_list:
                        heapq.heappush(heap, out_edge)

        print(self.mst)
        print(np.any(np.sum(self.mst, axis=0) == 0))

    @staticmethod
    def get_edge_from_numpy_arr(frm: int, out_arr: np.array):
        out_idx = np.argwhere(out_arr)
        edges = [Edge(frm, dest.item(), weight.item())
                 for dest, weight in zip(out_idx, out_arr[out_idx])]

        return edges


class Edge:
    def __init__(self, frm: int, to: int, weight: float):
        self._frm = frm
        self._to = to
        self._weight = weight

    def __eq__(self, other):
        return other.weight == self._weight

    def __lt__(self, other):
        return self._weight < other.weight

    def __repr__(self):
        return f"({self._frm})--{self._weight}--({self._to})"

    @property
    def weight(self):
        return self._weight

    @property
    def frm(self):
        return self._frm

    @property
    def to(self):
        return self._to
