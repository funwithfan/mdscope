"""This module contains functions for analyzing topogoly in a graph."""

import numba as nb
import numpy as np

@nb.njit
def bfs_connected_components(adj_matrix):
    num_nodes = adj_matrix.shape[0]
    visited = np.zeros(num_nodes, dtype=np.int32)  # Track whether a node has been visited
    cluster_sizes = []
    cluster_nodes = []  # Store nodes of each cluster

    def bfs(start_node):
        queue = [start_node]
        visited[start_node] = 1  # Mark the start node as visited
        cluster = [start_node]  # Nodes in the current cluster

        while queue:
            node = queue.pop(0)  # Dequeue the first node
            for neighbor in range(num_nodes):
                # If there's a connection and the neighbor hasn't been visited
                if adj_matrix[node, neighbor] > 0 and not visited[neighbor]:
                    visited[neighbor] = 1  # Mark neighbor as visited
                    queue.append(neighbor)  # Add neighbor to the queue
                    cluster.append(neighbor)  # Add neighbor to the current cluster
        return cluster

    # Iterate through all nodes to find clusters
    for i in range(num_nodes):
        if not visited[i]:  # If node hasn't been visited, start a new BFS
            cluster = bfs(i)
            cluster_sizes.append(len(cluster))  # Record the size of the cluster
            cluster_nodes.append(cluster)  # Record the nodes in the cluster

    return cluster_sizes, cluster_nodes  # Return cluster sizes and nodes in each cluster


@nb.njit
def dfs_connected_components(adj_matrix):
    num_nodes = adj_matrix.shape[0]
    visited = np.zeros(num_nodes, dtype=np.int32)  # Track whether a node has been visited
    cluster_sizes = []
    cluster_nodes = []  # Store nodes of each cluster

    def dfs(start_node):
        stack = [start_node]
        cluster = []  # Nodes in the current cluster

        while stack:
            node = stack.pop()  # Pop the top node from the stack
            if not visited[node]:
                visited[node] = 1  # Mark the node as visited
                cluster.append(node)  # Add node to the current cluster
                # Add all unvisited neighboring nodes to the stack
                # (No reversed order is used to avoid extra memory overhead)
                for neighbor in range(num_nodes):
                    if adj_matrix[node, neighbor] > 0 and not visited[neighbor]:
                        stack.append(neighbor)
        return cluster

    # Iterate through all nodes to find clusters
    for i in range(num_nodes):
        if not visited[i]:  # If node hasn't been visited, start a new DFS
            cluster = dfs(i)
            cluster_sizes.append(len(cluster))  # Record the size of the cluster
            cluster_nodes.append(cluster)  # Record the nodes in the cluster

    return cluster_sizes, cluster_nodes  # Return cluster sizes and nodes in each cluster