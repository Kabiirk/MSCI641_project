graph = [
    [1, 3, 2], # 0
    [0, 3], # 1
    [0, 3], # 2
    [0, 1, 2, 4], # 3
    [3] # 4
]

array = [64, 34, 25, 12, 22, 11, 90, 5]


def bubble_sort(array):
    n = len(array)
    for i in range(n-1):
        swapped = False
        for j in range(n-i-1):
            if(array[j] > array[j+1]):
                array[j], array[j+1] = array[j+1], array[j]
                swapped = True
                print(f"Swapping {array[j]},{array[j+1]} <- {array[j+1]},{array[j]}. Resulting array: {array}")
        if not swapped:
            break

    print("Sorted Array: ", array)

def selection_sort(array):
    n = len(array)
    for i in range(n-1):
        min_index = i
        for j in range(i+1, n):
            if array[j] < array[min_index]:
                min_index = j
        min_val = array.pop(min_index)
        array.insert(i, min_val)

    print("Sorted Array: ", array)

def bfs(graph, start, end):
    visited = []
    queue = [start]
    while len(queue) > 0:
        current_node = queue.pop(0)
        if current_node not in visited:
            print("Current Node: ", current_node)
            visited.append(current_node)
            if current_node == end or end in graph[current_node]:
                print("Reached End ! Destination: ", end)
                visited.append(end)
                print("===============================")
                break
            for w in graph[current_node]:
                if w not in visited:
                    queue.append(w)
            print("Queue: ", queue)
            print("Visited: ", visited)
            print("===============================")
    print("Path: ", visited)

# SORTING ALGOS
selection_sort(array)

# GRAPH ALGOS
# bfs(graph, 0, 4)