def k_means_centroid_update(points, assignments, k):
    D = len(points[0])  # dimensionality
    
    # Initialize sums and counts
    sums = [[0.0] * D for _ in range(k)]
    counts = [0] * k
    
    # Accumulate sums
    for p, cluster_idx in zip(points, assignments):
        for d in range(D):
            sums[cluster_idx][d] += p[d]
        counts[cluster_idx] += 1
    
    # Compute means
    centroids = []
    for i in range(k):
        if counts[i] == 0:
            # If cluster is empty, keep zeros (or could keep old centroid)
            centroids.append([0.0] * D)
        else:
            centroids.append([
                sums[i][d] / counts[i] for d in range(D)
            ])
    
    return centroids