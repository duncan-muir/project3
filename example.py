from mst import Graph


def main():
    file_path = './data/small.csv'
    g = Graph(file_path)
    g.construct_mst()


if __name__ == '__main__':
    main()
