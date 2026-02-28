def k_means_assignment(points, centroids):

    assignments = []
    
    for p in points:
        best_dist = float('inf')
        best_idx = 0
        
        for i, c in enumerate(centroids):
            # Compute squared Euclidean distance
            dist = 0
            for d in range(len(p)):
                dist += (p[d] - c[d]) ** 2
            
            # Strict < to break ties by smaller index
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        
        assignments.append(best_idx)
    
    return assignments 