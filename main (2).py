# app.py
from flask import Flask, render_template, request, jsonify
import heapq
from collections import deque
import time

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
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up
    
    for dx, dy in directions:
        new_x, new_y = node.x + dx, node.y + dy
        if (0 <= new_x < len(grid) and 
            0 <= new_y < len(grid[0]) and 
            grid[new_x][new_y] != 1):  # 1 represents obstacles
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