import itertools
from heapq import heappush, heappop


# Priority queue implementation is from https://docs.python.org/3/library/heapq.html
class PriorityQueue:
    def __init__(self):
        self.pq = []                         # list of entries arranged in a heap
        self.entry_finder = {}               # mapping of tasks to entries
        self.REMOVED = '<removed-task>'      # placeholder for a removed task
        self.counter = itertools.count()     # unique sequence count

    def add_task(self, task, priority=0):
        """Add a new task or update the priority of an existing task"""
        if task in self.entry_finder:
            self.remove_task(task)
        count = next(self.counter)
        entry = [priority, count, task]
        self.entry_finder[task] = entry
        heappush(self.pq, entry)

    def remove_task(self, task):
        """Mark an existing task as REMOVED. Raise KeyError if not found."""
        entry = self.entry_finder.pop(task)
        entry[-1] = self.REMOVED

    def pop_task(self):
        """Remove and return the lowest priority task. Raise KeyError if empty."""
        while self.pq:
            priority, count, task = heappop(self.pq)
            if task is not self.REMOVED:
                del self.entry_finder[task]
                return priority, task
        raise KeyError('pop from an empty priority queue')


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
    while queue:
        r_distance, removed = queue.pop_task()
        visited[removed] = True

        # Code to keep track of pathing once end point is found
        if removed == end:
            while previous[removed]:
                path.append(removed)
                removed = previous[removed]
            path.append(start)
            print("Shortest path is :", path[::-1])
            print("Total distance is :", distance[end])
            return

        # Nested for loop to iterate through list of dictionaries
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


def main():

    # Graph representation as adjacency list
    # Format is dictionary [{k,v},{k,v}]
    # Dictionary chosen as value inside list over tuples for mutability
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

    print("\nThis program allows you to optionally print a visual representation of the graph.")
    print("A matplotlib window will pop up and allow you to view and/or save an image of the graph.\n")
    print("****IMPORTANT NOTE****\nBecause of Pythons GIL "
          "the matplotlib window needs to be closed when you are finished for execution to continue."
          "\n********************\n")
    print("For this to work matplotlib and networkx need to be installed with below commands:\n")
    print("pip install networkx")
    print("pip install matplotlib\n")
    choice = input("Continue with visualization? (Y/N) : ")
    if choice == 'y' or choice == 'Y':
        print("Visualizing the graph...")
        print("You can save the graph image by clicking the save button in the matplotlib window.")
        print("Please close the matplotlib window when you are finished.")
        print("If you do not close the window the program will not continue.")
        visualize(adjacency_list)
    else:
        print("Continuing without visualization")

    start = input("Enter starting point for graph [Values are A-J] : ")
    end = input("Enter end point for graph [Values are A-J] : ")

    dijkstras_shortest_path(adjacency_list, start.upper(), end.upper())

    while True:
        choice = input("Would you like to test another path (Y/N): ")
        if choice.lower() == 'y':
            main()
            break
        elif choice.lower() == 'n':
            print("Exiting program.")
            break
        else:
            print("Invalid input, please enter Y or N.")

if __name__ == '__main__':
    main()