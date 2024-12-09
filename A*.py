import heapq

# A* algorithm implementation
def a_star(graph, start, goal):
    # Priority queue to store (cost, node)
    open_list = []
    heapq.heappush(open_list, (0, start))
    
    # Dictionaries to store the cost of reaching each node and the path
    g_costs = {start: 0}
    f_costs = {start: heuristic(start, goal)}
    
    # Parent map to reconstruct the path
    came_from = {}
    
    while open_list:
        # Get the node with the lowest f_cost
        current = heapq.heappop(open_list)[1]
        
        if current == goal:
            return reconstruct_path(came_from, current)  # Return the path
        
        # Traverse the neighbors
        for neighbor, cost in graph[current].items():
            tentative_g_cost = g_costs[current] + cost
            
            # If this path is better, update the cost and path
            if neighbor not in g_costs or tentative_g_cost < g_costs[neighbor]:
                g_costs[neighbor] = tentative_g_cost
                f_costs[neighbor] = tentative_g_cost + heuristic(neighbor, goal)
                came_from[neighbor] = current
                heapq.heappush(open_list, (f_costs[neighbor], neighbor))
    
    return None  # No path found

# Heuristic function (Manhattan distance in this case)
def heuristic(node, goal):
    x1, y1 = node
    x2, y2 = goal
    return abs(x1 - x2) + abs(y1 - y2)

# Function to reconstruct the path from start to goal
def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path

# Example usage
graph = {
    (0, 0): {(1, 0): 1, (0, 1): 1},
    (1, 0): {(1, 1): 1, (2, 0): 2},
    (0, 1): {(1, 1): 1},
    (1, 1): {(2, 1): 1},
    (2, 0): {(2, 1): 1},
    (2, 1): {}
}

start = (0, 0)
goal = (2, 1)

path = a_star(graph, start, goal)
print("Shortest path:", path)