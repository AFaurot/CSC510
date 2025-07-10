import networkx as nx
import matplotlib.pyplot as plt


class GraphVisualizer:
    def __init__(self, adjacency_list):
        self.adjacency_list = adjacency_list
        self.graph = nx.Graph()  # Create a new graph object

    def add_edges(self):
        # Add edges to the graph from the adjacency list
        for node, neighbors in self.adjacency_list.items():
            for neighbor_dict in neighbors:
                for neighbor, weight in neighbor_dict.items():
                    self.graph.add_edge(node, neighbor, weight=weight)

    def draw_graph(self):
        # Layout for better visualization
        pos = nx.spring_layout(self.graph, seed=506)  # Spring layout seed 506 because CSC506!!

        # Draw nodes and edges
        nx.draw(self.graph, pos, with_labels=True, node_size=700, node_color="lightblue", font_size=10,
                font_weight="bold")

        # Draw edge labels (weights)
        edge_labels = nx.get_edge_attributes(self.graph, 'weight')
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels)

        # Show the plot
        plt.title("Graph Visualization with Edge Weights (close to continue program)")
        plt.show()
