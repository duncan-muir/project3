from mst import Graph
import numpy as np

def main():
    file_path = './data/small.csv'
    g = Graph(file_path)
    g.construct_mst()


if __name__ == '__main__':
    main()
