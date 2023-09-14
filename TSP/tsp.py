from math import sqrt
from typing import List, Tuple
import numpy as np
import math
import heapq
import time
import matplotlib.pyplot as plt
import networkx as nx
import random

def MST_heuristic(cities : List[Tuple[float, float]]) -> float:

    # Generate distances between all cities
    all_cities_distance_matrix = []
    for i in range(len(cities)):
        distance_btw_cities = []
        for j in range(len(cities)):
            distance_btw_cities.append(math.dist(cities[i], cities[j]))
        all_cities_distance_matrix.append(distance_btw_cities)

    # Maintain list of all visited cities
    visited = [True] + [False for i in range(len(cities) - 1)]
    cost = 0

    #Prims Algorithm
    a = 0
    while a < len(cities) - 1 :
        nearest_next_node_distance = float('inf')
        nearest_next_node = 0
        for i in range(len(cities)):
            if visited[i] == True:
                for j in range(len(cities)):
                    if visited[j] == False and all_cities_distance_matrix[i][j] < nearest_next_node_distance:
                        nearest_next_node = j
                        nearest_next_node_distance = all_cities_distance_matrix[i][j]
        cost += nearest_next_node_distance
        visited[nearest_next_node] = True
        a += 1
    return cost

# Function to solve TSP problem using A*search.
# g(n) -> Distance to reach next node from current node
# h(n) -> (MST algorithm to calculate minimum path
# cost to traverse all unvisited nodes) + (minimum cost
# to return to start node from any of the unvisited nodes)
def tsp_search(cities):
    start_time = time.time()

    #Generate distances between cities
    all_cities_distance_matrix = []
    for i in range(len(cities)):
        distance_btw_cities = []
        for j in range(len(cities)):
            distance_btw_cities.append(math.dist(cities[i], cities[j]))
        all_cities_distance_matrix.append(distance_btw_cities)

    # Using Integers to check whether a City has been visited or not.
    # 0 - Unvisited, 1 - Newly Visited (Once), 2 -Visited Twice
    visited = [1] + [0 for i in range(len(cities) - 1)]
    out_list = [cities[0]]
    cost = 0
    abs_cost = 0
    a = 0
    last_visit_index = 0
    nodes_expanded = 0

    while a < len(cities) - 1 :
        nearest_next_node_distance = 5000
        for i in range(len(cities)):
            # We avoid running the loop again for already visited cities as they will have visited = 2
            if visited[i] == 1:
                nodes_expanded += 1
                for j in range(len(cities)):
                    if visited[j] == 0:
                        nodes_expanded += 1
                        mst_city_list = []
                        visited_cities = []
                        temp_id = j
                        for k in range(len(visited)):
                            if visited[k] >= 1:
                                visited_cities.append(cities[k])
                            if visited[k] == 0:
                                mst_city_list.append(cities[k])
                        return_cost = min([math.dist(l1, cities[0]) for l1 in mst_city_list])
                        #Checking g(n) + h(n) here.
                        if (all_cities_distance_matrix[i][j] + MST_heuristic(mst_city_list) + return_cost) < nearest_next_node_distance:
                            nearest_next_node = j
                            nearest_next_node_distance = all_cities_distance_matrix[i][j] + MST_heuristic(mst_city_list) + return_cost
                            abs_cost = all_cities_distance_matrix[i][j]
                #This makes sure we go to new city
                visited[i] += 1
        cost += abs_cost
        visited[nearest_next_node] = 1
        out_list.append(cities[nearest_next_node])
        last_visit_index = nearest_next_node
        a += 1

#    print("Final Cost "+str(cost))
    out_list.append(cities[0])
    tsp_time = time.time() - start_time
    return cost + all_cities_distance_matrix[0][last_visit_index] , out_list, tsp_time, nodes_expanded

# Print out 'k' random real points - (x, y) in the range of (0, 1)
# Minimum separation of epsilon=0.1 exists between co-ordinates
# of every point. Duplicate points are discarded.
def city_generator(k):
    cities_list = []
    while len(cities_list) != k:
        x = round(random.uniform(0, 1), 1)
        y = round(random.uniform(0, 1), 1)
        if (x, y) in cities_list:
            continue
        cities_list.append((x, y))
    return cities_list

def solve_problem(k):
    city_list = city_generator(k)
    print("List of Generated Cities = "+str(city_list))
    c, ols, tsp_time, nodes_expanded = tsp_search(city_list)
    #Uncomment next two line for 'All_paths' graph
    #edges, weights = create_graph_edges(ols)
    #create_graph(edges, weights)
    print("Order of graph traversal for TSP = "+str(ols))
    print("Total time to solve TSP = "+str(tsp_time))
    print("Total path cost = "+str(c))
    #Uncomment for TSP graph
    #tsp_edges = create_TSP_edges(ols)
    #tsp_graph(tsp_edges)
    return tsp_time, nodes_expanded

# Create graph edges from list sorted
# in the order of traversal
def create_graph_edges(lst):
    graph_edges = []
    costs = []
    for city1 in lst:
        for city2 in lst:
            if city1 == city2:
                continue
            graph_edges.append((city1, city2))
            costs.append(((city1, city2), math.dist(city1, city2)))
    return graph_edges, costs

def create_TSP_edges(city_ss):
    supr = []
    for i in range(len(city_ss)-1):
        supr.append(((city_ss[i], city_ss[i+1]), math.dist(city_ss[i], city_ss[i+1])))
    return supr

#Create graph
def create_graph(graph_edges, costs):
    G = nx.Graph()
    G.add_edges_from(graph_edges)
    pos = nx.spring_layout(G)
    plt.figure()
    nx.draw(
        G, pos, edge_color='black', width=1, linewidths=1,
        node_size=500, node_color='pink', alpha=0.9,
        labels={node: node for node in G.nodes()}
    )
    nx.draw_networkx_edge_labels(
        G, pos,

        edge_labels={cost[0]: cost[1] for cost in costs},
        font_color='black'
    )
    plt.axis('off')
    plt.title('All possible paths')
    plt.show()

#Create the TSP graph
def tsp_graph(tsp_edges):
    graph_edges = []
    costs = []
    for a in tsp_edges:
        graph_edges.append(a[0])
        costs.append(a[1])
    G = nx.DiGraph()
    G.add_edges_from(graph_edges)
    pos = nx.spring_layout(G)
    plt.figure()
    nx.draw(
        G, pos, edge_color='black', width=1, linewidths=1,
        node_size=500, node_color='pink', alpha=0.9,
        labels={node: node for node in G.nodes()}
    )
    plt.axis('off')
    plt.title('TSP Solution')
    plt.show()


def check_mst_heuristic_for_case(cities : List[Tuple[float, float]], true_cost : float):
    cost = MST_heuristic(cities)
    print( "Current case: Returned cost for MST "+str(cost)+"; true cost "+str(true_cost))
    if abs(true_cost - cost) > 1e-9:
        raise Exception("Error: True cost doesn't match cost returned by MST implementation")

def check_mst_heuristic():
    cities1 = [(1,1), (2,2)]
    cities2 = [(1,1), (2,2), (3,3)]
    cities3 = [(1,1), (0,1), (0, 0), (1, 0)]
    cities4 = [(1,1), (2,1), (0, 0), (3, 0)]
    sqrt2 = sqrt(2.0)
    check_mst_heuristic_for_case(cities1, sqrt2)
    check_mst_heuristic_for_case(cities2, 2*sqrt2)
    check_mst_heuristic_for_case(cities3, 3)
    check_mst_heuristic_for_case(cities4, 1 + 2*sqrt2)


def check_tsp_for_case(cities : List[Tuple[float, float]], true_cost : float):
    cost, out_list = tsp_search(cities)[0], tsp_search(cities)[1]
    print("Current case: Returned cost for TSP "+str(cost)+"; true cost "+str(true_cost))
    print("Path taken ="+str(out_list))
    if abs((true_cost) - cost) > 1e-9 :
        raise Exception("Error: True cost doesn't match cost returned by TSP implementation")

def check_tsp():
    cities1 = [(1,1), (2,2)]
    cities2 = [(1,1), (2,2), (3,3)]
    cities3 = [(1,1), (0,1), (0, 0), (1, 0)]
    cities4 = [(1,1), (2,1), (0, 0), (3, 0)]
    sqrt2 = sqrt(2.0)
    check_tsp_for_case(cities1, 2*sqrt2)
    check_tsp_for_case(cities2, 4*sqrt2)
    check_tsp_for_case(cities3, 4.0)
    check_tsp_for_case(cities4, 4 + 2*sqrt2)

def plot_final_graph_nodes(comp_time):
    xpoints = []
    ypoints = []
    for c in comp_time:
        xpoints.append(c[0])
        ypoints.append(c[1])
    plt.plot(xpoints, ypoints)
    plt.xlabel('No of iterations')
    plt.ylabel('Computation Time')
    plt.title('No of iterations vs Computation Time')
    plt.show()

def run_tests():
    try:
        check_mst_heuristic()
        check_tsp()
        print("All tests passed!")
    except Exception as e:
        print(str(e))

if __name__ == '__main__':
    #run_tests()
    comp_time = []
    node_e = []
    for i in range(2,50,1):
        a, c = solve_problem(i)
        comp_time.append((i, a))
        node_e.append((i, c))
    plot_final_graph_nodes(comp_time)
