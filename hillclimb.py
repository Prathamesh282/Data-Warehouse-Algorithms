import random

# Hill Climbing Algorithm
def hill_climbing(function, neighbors, start):
    current = start
    current_value = function(current)

    while True:
        # Generate neighbors and evaluate their values
        neighbor_values = [(neighbor, function(neighbor)) for neighbor in neighbors(current)]
        
        # Find the best neighbor (maximization)
        best_neighbor, best_value = max(neighbor_values, key=lambda x: x[1])

        # If no better neighbor, return the current best state
        if best_value <= current_value:
            return current, current_value

        # Move to the best neighbor
        current, current_value = best_neighbor, best_value

# Example function to maximize (simple quadratic function)
def objective_function(x):
    return -(x ** 2) + 5 * x + 10  # A parabola

# Example neighbors (small step around the current value)
def generate_neighbors(x):
    step_size = 1
    return [x - step_size, x + step_size]

# Starting point
start_point = random.randint(-10, 10)

# Run the hill climbing algorithm
best_solution, best_value = hill_climbing(objective_function, generate_neighbors, start_point)

print("Best Solution:", best_solution)
print("Best Value:", best_value)