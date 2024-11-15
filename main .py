from flask import Flask, render_template, request, jsonify
import heapq
from collections import deque
import time
import psutil
import tracemalloc
import numpy as np
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import os

app = Flask(__name__)

class Node:
    def __init__(self, x, y, cost=float('inf')):
        self.x = x
        self.y = y
        self.cost = cost
        self.parent = None
        
    def __lt__(self, other):
        return self.cost < other.cost

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

def manhattan_distance(x1, y1, x2, y2):
    return abs(x1 - x2) + abs(y1 - y2)

def get_neighbors(node, grid):
    neighbors = []
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)] 
    for dx, dy in directions:
        new_x, new_y = node.x + dx, node.y + dy
        if (0 <= new_x < len(grid) and 
            0 <= new_y < len(grid[0]) and 
            grid[new_x][new_y] != 1):  
            neighbors.append(Node(new_x, new_y))
    return neighbors

def reconstruct_path(node):
    path = []
    current = node
    while current:
        path.append((current.x, current.y))
        current = current.parent
    return path[::-1]

def a_star(grid, start, end):
    start_node = Node(start[0], start[1], 0)
    end_node = Node(end[0], end[1])
    
    open_set = [start_node]
    closed_set = set()
    
    while open_set:
        current = heapq.heappop(open_set)
        
        if (current.x, current.y) == (end_node.x, end_node.y):
            return reconstruct_path(current)
            
        closed_set.add((current.x, current.y))
        
        for neighbor in get_neighbors(current, grid):
            if (neighbor.x, neighbor.y) in closed_set:
                continue
                
            g_score = current.cost + 1
            h_score = manhattan_distance(neighbor.x, neighbor.y, end[0], end[1])
            f_score = g_score + h_score
            
            neighbor.cost = f_score
            neighbor.parent = current
            
            if neighbor not in open_set:
                heapq.heappush(open_set, neighbor)
    
    return []

def bfs(grid, start, end):
    start_node = Node(start[0], start[1])
    queue = deque([start_node])
    visited = set([(start[0], start[1])])
    
    while queue:
        current = queue.popleft()
        
        if (current.x, current.y) == end:
            return reconstruct_path(current)
            
        for neighbor in get_neighbors(current, grid):
            if (neighbor.x, neighbor.y) not in visited:
                visited.add((neighbor.x, neighbor.y))
                neighbor.parent = current
                queue.append(neighbor)
    
    return []

def dfs(grid, start, end):
    start_node = Node(start[0], start[1])
    stack = [start_node]
    visited = set([(start[0], start[1])])
    
    while stack:
        current = stack.pop()
        
        if (current.x, current.y) == end:
            return reconstruct_path(current)
            
        for neighbor in get_neighbors(current, grid):
            if (neighbor.x, neighbor.y) not in visited:
                visited.add((neighbor.x, neighbor.y))
                neighbor.parent = current
                stack.append(neighbor)
    
    return []

def dijkstra(grid, start, end):
    start_node = Node(start[0], start[1], 0)
    priority_queue = [(0, start_node)]
    visited = set()
    
    while priority_queue:
        current_cost, current = heapq.heappop(priority_queue)
        
        if (current.x, current.y) in visited:
            continue
            
        visited.add((current.x, current.y))
        
        if (current.x, current.y) == end:
            return reconstruct_path(current)
            
        for neighbor in get_neighbors(current, grid):
            if (neighbor.x, neighbor.y) not in visited:
                new_cost = current_cost + 1
                neighbor.cost = new_cost
                neighbor.parent = current
                heapq.heappush(priority_queue, (new_cost, neighbor))
    
    return []

@app.route('/analyze_performance', methods=['GET', 'POST'])
def analyze_performance():
    
    default_sizes = [10, 15, 20, 25, 30]
    
    if request.method == 'POST':
        data = request.get_json()
        grid_sizes = data.get('grid_sizes', default_sizes)
    else:
        grid_sizes = default_sizes
    
    results = {
        'time': {'a_star': [], 'bfs': [], 'dfs': [], 'dijkstra': []},
        'memory': {'a_star': [], 'bfs': [], 'dfs': [], 'dijkstra': []},
        'cpu': {'a_star': [], 'bfs': [], 'dfs': [], 'dijkstra': []}
    }
    
    algorithms = {
        'a_star': a_star,
        'bfs': bfs,
        'dfs': dfs,
        'dijkstra': dijkstra
    }
    
    for size in grid_sizes:
       
        grid = [[0 for _ in range(size)] for _ in range(size)]

        num_walls = int(0.2 * size * size)
        wall_positions = np.random.choice(size * size, num_walls, replace=False)
        for pos in wall_positions:
            x, y = pos // size, pos % size
            grid[x][y] = 1
        
        start = (0, 0)
        end = (size-1, size-1)
        
       
        for alg_name, algorithm in algorithms.items():
            
            process = psutil.Process()
            initial_cpu = process.cpu_percent()
            
           
            tracemalloc.start()
            
           
            start_time = time.time()
            algorithm(grid, start, end)
            execution_time = time.time() - start_time
            
            
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
           
            cpu_usage = process.cpu_percent() - initial_cpu
            
            
            results['time'][alg_name].append(execution_time)
            results['memory'][alg_name].append(peak / 1024 / 1024)  # Convert to MB
            results['cpu'][alg_name].append(cpu_usage)
    
    
    plots = generate_performance_plots(results, grid_sizes)
    
    if request.method == 'GET':
      
        return render_template('performance.html', 
                             results=results, 
                             plots=plots,
                             grid_sizes=grid_sizes)
    else:
       
        return jsonify({
            'results': results,
            'plots': plots
        })
def generate_performance_plots(results, grid_sizes):
    plots = {}
    metrics = {
        'time': 'Execution Time (seconds)',
        'memory': 'Memory Usage (MB)',
        'cpu': 'CPU Usage (%)'
    }
    
    for metric, ylabel in metrics.items():
        plt.figure(figsize=(10, 6))
        for alg_name in results[metric].keys():
            plt.plot(grid_sizes, results[metric][alg_name], 
                    marker='o', label=alg_name.upper())
        
        plt.xlabel('Grid Size')
        plt.ylabel(ylabel)
        plt.title(f'{metric.capitalize()} Comparison')
        plt.grid(True)
        plt.legend()
        
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        plt.close()
        
        plots[metric] = base64.b64encode(image_png).decode()
    
    return plots


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/find_paths', methods=['POST'])
def find_paths():
    data = request.get_json()
    grid = data['grid']
    start = tuple(data['start'])
    end = tuple(data['end'])
    
    results = {
        'a_star': {'path': a_star(grid, start, end)},
        'bfs': {'path': bfs(grid, start, end)},
        'dfs': {'path': dfs(grid, start, end)},
        'dijkstra': {'path': dijkstra(grid, start, end)}
    }
    
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)