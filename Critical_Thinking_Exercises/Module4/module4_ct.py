import itertools
from heapq import heappush, heappop


# Priority queue custom class implementation
class PriorityQueue:
    def __init__(self):
        self.pq = []
        self.entry_finder = {}
        self.REMOVED = '<removed-task>'
        self.counter = itertools.count()

    def add_task(self, task, priority=0):
        if task in self.entry_finder:
            self.remove_task(task)
        count = next(self.counter)
        entry = [priority, count, task]
        self.entry_finder[task] = entry
        heappush(self.pq, entry)

    def remove_task(self, task):
        entry = self.entry_finder.pop(task)
        entry[-1] = self.REMOVED

    def pop_task(self):
        while self.pq:
            priority, count, task = heappop(self.pq)
            if task is not self.REMOVED:
                self.entry_finder.pop(task, None)  # use .pop to avoid KeyError if already gone
                return priority, task
        return None, None  # <-- NEW: safely return (None, None) instead of raising


# Requires graph_visualization.py to visualize the graph
def visualize(adjacency_list):
    from graph_visualization import GraphVisualizer
    graph = GraphVisualizer(adjacency_list)
    graph.add_edges()
    graph.draw_graph()


# Shortest path implementation inspired from Glassbyte YouTube video on this topic
# https://youtu.be/_B5cx-WD5EA?si=8ijBEbvvM_G0TNNR
def dijkstras_shortest_path(adjacency_list, start, end):
    visited = {v: False for v in adjacency_list.keys()}
    previous = {v: None for v in adjacency_list.keys()}
    distance = {v: float("inf") for v in adjacency_list.keys()}
    distance[start] = 0
    queue = PriorityQueue()
    queue.add_task(start)
    path = []
    nodes_visited = 0

    while queue:
        r_distance, removed = queue.pop_task()
        nodes_visited += 1
        visited[removed] = True

        if removed == end:
            while previous[removed]:
                path.append(removed)
                removed = previous[removed]
            path.append(start)
            print("\nDijkstra Path:", path[::-1])
            print("Dijkstra Total distance:", distance[end])
            print("Dijkstra Nodes visited:", nodes_visited)
            return

        for neighbor in adjacency_list[removed]:
            for vertex, edge_distance in neighbor.items():
                if visited[vertex]:
                    continue
                new_distance = r_distance + edge_distance
                if new_distance < distance[vertex]:
                    distance[vertex] = new_distance
                    previous[vertex] = removed
                    queue.add_task(vertex, new_distance)
    return


# Reverse Dijkstra's algorithm to find shortest paths from goal to all nodes to create a heuristic
def reverse_dijkstra(adjacency_list, goal):
    distance = {v: float("inf") for v in adjacency_list}
    distance[goal] = 0
    visited = {v: False for v in adjacency_list}
    queue = PriorityQueue()
    queue.add_task(goal, 0)

    while True:
        dist, node = queue.pop_task()
        if node is None:
            break
        if visited[node]:
            continue
        visited[node] = True
        for neighbor_dict in adjacency_list[node]:
            for neighbor, weight in neighbor_dict.items():
                if visited[neighbor]:
                    continue
                new_dist = dist + weight
                if new_dist < distance[neighbor]:
                    distance[neighbor] = new_dist
                    queue.add_task(neighbor, new_dist)
    print("\nGenerated Heuristic from Reverse Dijkstra (to goal '{}'):".format(goal))
    for node, h in distance.items():
        print(f"  {node}: {h}")
    return distance


# A * algorithm to find the shortest path using a heuristic
def astar_shortest_path(adjacency_list, start, end, heuristic):
    visited = {v: False for v in adjacency_list.keys()}
    previous = {v: None for v in adjacency_list.keys()}
    g_score = {v: float("inf") for v in adjacency_list.keys()}
    g_score[start] = 0
    queue = PriorityQueue()
    queue.add_task(start, heuristic[start])
    path = []
    nodes_visited = 0

    while queue:
        _, current = queue.pop_task()
        nodes_visited += 1
        visited[current] = True

        if current == end:
            while previous[current]:
                path.append(current)
                current = previous[current]
            path.append(start)
            print("\nA* Path:", path[::-1])
            print("A* Total distance:", g_score[end])
            print("A* Nodes visited:", nodes_visited)
            return

        for neighbor_dict in adjacency_list[current]:
            for neighbor, edge_distance in neighbor_dict.items():
                if visited[neighbor]:
                    continue
                tentative_g = g_score[current] + edge_distance
                if tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    previous[neighbor] = current
                    # Calculate f_score using the heuristic to prioritize nodes
                    f_score = tentative_g + heuristic.get(neighbor, float("inf"))
                    queue.add_task(neighbor, f_score)
    return


# Call and compare both algorithms
def call_algorithms(adjacency_list, heuristic):

    # Checks if input in valid range
    while True:
        start = input("Enter starting point for graph [Values are A-J]: ")
        end = input("Enter end point for graph [Values are A-J]: ")
        start = start.upper()
        end = end.upper()
        if start in adjacency_list and end in adjacency_list:
            break
        else:
            print("Invalid input. Please enter values between A and J.")
    if heuristic == 'toggle_reverse_dijkstra':
        heuristic = reverse_dijkstra(adjacency_list, end)
        print("\nUsing Reverse Dijkstra's heuristic for A* algorithm.")
    dijkstras_shortest_path(adjacency_list, start, end)
    astar_shortest_path(adjacency_list, start, end, heuristic)


# Main function to run the program
# Includes visualization option and heuristic selection
def main():

    # Adjacency list data structure representing a graph, number are edge weights
    adjacency_list = {
        'A': [{'B': 6}, {'C': 3}],
        'B': [{'A': 6}, {'D': 5}, {'E': 2}],
        'C': [{'A': 3}, {'D': 2}, {'G': 4}],
        'D': [{'B': 5}, {'C': 2}, {'F': 3}, {'G': 3}],
        'E': [{'B': 2}, {'F': 2}, {'H': 4}],
        'F': [{'D': 3}, {'E': 2}, {'G': 4}, {'H': 1}],
        'G': [{'C': 4}, {'D': 3}, {'F': 4}, {'J': 6}],
        'H': [{'E': 4}, {'F': 1}, {'I': 2}],
        'I': [{'H': 2}, {'J': 1}],
        'J': [{'I': 1}, {'G': 6}],
    }

    # Average heuristic values computed from average of all edge weights at each node
    heuristic_avg = {
        'A': 4.5, 'B': 4.3, 'C': 3, 'D': 3.25,
        'E': 2.7, 'F': 2.5, 'G': 4.25, 'H': 2.3,
        'I': 1.5, 'J': 3.5
    }
    # Lowest heuristic values based on the lowest edge weights
    heuristic_lowest = {
        'A': 3, 'B': 2, 'C': 2, 'D': 2,
        'E': 2, 'F': 1, 'G': 3, 'H': 1,
        'I': 1, 'J': 1
    }

    print("\nThis program allows you to optionally print a visual representation of the graph.")
    print("A matplotlib window will pop up and allow you to view and/or save an image of the graph.\n")
    print("****IMPORTANT NOTE****\nBecause of Python’s GIL, "
          "the matplotlib window needs to be closed when you are finished for execution to continue."
          "\n********************\n")
    print("For this to work matplotlib and networkx need to be installed with below commands:\n")
    print("pip install networkx")
    print("pip install matplotlib\n")
    choice = input("Continue with visualization? (Y/N): ")
    if choice.lower() == 'y':
        print("Visualizing the graph...")
        print("You can save the graph image by clicking the save button in the matplotlib window.")
        print("Please close the matplotlib window when you are finished.")
        print("If you do not close the window the program will not continue.")
        visualize(adjacency_list)
    else:
        print("Continuing without visualization")

    print("\nChoose a heuristic for A* algorithm:")
    print("1. Average heuristic")
    print("2. Lowest heuristic")
    print("3. Reverse Dijkstra's heuristic")
    heuristic_choice = input("Enter 1, 2, or 3 (default is 1): ")
    if heuristic_choice == '1':
        heuristic = heuristic_avg
    elif heuristic_choice == '2':
        heuristic = heuristic_lowest
    elif heuristic_choice == '3':
        heuristic = 'toggle_reverse_dijkstra'
    else:
        print("Invalid choice, using average heuristic by default.")
        heuristic = heuristic_avg
    # Call the algorithms with the adjacency list and heuristic
    call_algorithms(adjacency_list, heuristic)

    while True:
        choice = input("\nWould you like to test another path (Y/N): ")
        if choice.lower() == 'y':
            print("\nChoose a heuristic for A* algorithm:")
            print("1. Average heuristic")
            print("2. Lowest heuristic")
            print("3. Reverse Dijkstra's heuristic")
            heuristic_choice = input("Enter 1, 2, or 3 (default is 1): ")
            if heuristic_choice == '1':
                heuristic = heuristic_avg
            elif heuristic_choice == '2':
                heuristic = heuristic_lowest
            elif heuristic_choice == '3':
                heuristic = 'toggle_reverse_dijkstra'
            else:
                print("Invalid choice, using average heuristic by default.")
                heuristic = heuristic_avg
            call_algorithms(adjacency_list, heuristic)
        elif choice.lower() == 'n':
            print("Exiting program.")
            break
        else:
            print("Invalid input, please enter Y or N.")


if __name__ == '__main__':
    main()