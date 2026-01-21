# ==========================================
# Homework 1: Clustering and Regression
# Part: Clustering (T5 - T7)
# Name: Thananop Kullapan
# Student ID: 6530182121
# ==========================================

import numpy as np

# Data Points from the problem
X = np.array([
    [1, 2], [3, 3], [2, 2],
    [8, 8], [6, 6], [7, 7],
    [-3, -3], [-2, -4], [-7, -7]
])

def run_kmeans_step_by_step(data, initial_centroids, label):
    print(f"\n{'='*20}\nRunning {label}\n{'='*20}")
    centroids = initial_centroids.copy()
    iteration = 0
    
    while True:
        iteration += 1
        print(f"\n--- Iteration {iteration} ---")
        print(f"Current Centroids:\n{centroids}")
        
        # 1. Assignment Step
        clusters = [[] for _ in range(len(centroids))]
        assignments = []
        
        for i, point in enumerate(data):
            # Calculate distance to each centroid
            distances = [np.linalg.norm(point - c) for c in centroids]
            # Find closest centroid
            cluster_idx = np.argmin(distances)
            clusters[cluster_idx].append(point)
            assignments.append(cluster_idx)
        
        # Display assignments
        # print(f"Assignments indices: {assignments}")
        
        # 2. Update Step
        new_centroids = []
        max_shift = 0
        
        for k in range(len(centroids)):
            cluster_points = np.array(clusters[k])
            if len(cluster_points) == 0:
                # Handle empty cluster (keep old centroid or re-initialize)
                new_c = centroids[k]
                print(f"  Cluster {k}: Empty (Keeping old centroid)")
            else:
                new_c = np.mean(cluster_points, axis=0)
                print(f"  Cluster {k}: Assigned {len(cluster_points)} points -> New Mean: {new_c}")
            
            # Check convergence shift
            shift = np.linalg.norm(new_c - centroids[k])
            max_shift = max(max_shift, shift)
            new_centroids.append(new_c)
            
        if max_shift < 1e-4:
            print("\n>>> Converged! Centroids have stopped changing.")
            break
            
        centroids = np.array(new_centroids)
    
    return centroids

# --- T5 Execution ---
# Starting points: (3,3), (2,2), (-3,-3)
start_t5 = np.array([[3.0, 3.0], [2.0, 2.0], [-3.0, -3.0]])
final_centroids_t5 = run_kmeans_step_by_step(X, start_t5, "T5: Start (3,3), (2,2), (-3,-3)")

# --- T6 Execution ---
# Starting points: (-3,-3), (2,2), (-7,-7)
start_t6 = np.array([[-3.0, -3.0], [2.0, 2.0], [-7.0, -7.0]])
final_centroids_t6 = run_kmeans_step_by_step(X, start_t6, "T6: Start (-3,-3), (2,2), (-7,-7)")

# --- T7 & OT2 Analysis (Printed) ---
print("\n" + "="*30)
print("Analysis Results")
print("="*30)
print("T7: Comparison - The starting points in T5 are better distributed.")
print("OT2: Best K - Based on visual inspection of the 3 distinct groups, K=3 is optimal.")
