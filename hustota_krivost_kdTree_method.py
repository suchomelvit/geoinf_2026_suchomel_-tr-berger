# Prohlášení o používaní AI:
# Nástorje AI (Google Gemini) byly použity pouze pro návrh postupu
# a případné debugování kódu. Části, kde byla AI využita jsou v kódu popsány 



from math import *
from time import *
from matplotlib.pyplot import *
from numpy import * 

def loadPoints (file):
    # Load points from .txt as list
    X, Y, Z = [], [], []
    
    with open(file) as f:
        for line in f:
            x, y, z = line.split('\t')
            
            X.append(float(x))
            Y.append(float(y))
            Z.append(float(z))
    
    return X, Y, Z

#Load points 
X, Y, Z = loadPoints("tree_18.txt")

# Creating class for KDNode and KDTree
class KDNode:

    def __init__(self, point=None, axis=None, left=None, right=None, point_index=None):
    
        self.point = point
        self.axis = axis
        self.left = left
        self.right = right
        self.point_index = point_index
        
        
class KDTree:
    def __init__(self, X, Y, Z):
        self.data = array([X, Y, Z]).T # Had issues with data type, ai suggested using .T to transpose the array so that each point is a row instead of a column
        
        idx = arange(len(X))
        
        self.root = self._build_tree(idx, depth=0)


    def _build_tree(self, idxs, depth):
        if len(idxs) == 0:
            return None
        # determining the depth of the tree 

        axis = depth % 3
        
        # Sorting the indices based on the axis values 

        idxs = sorted(idxs, key=lambda i: self.data[i][axis]) 

        # Computing the median index for spliting the data
        middle_val = len(idxs) // 2
        median_idx = idxs[middle_val]

        median_point = self.data[median_idx]
        left_branch = self._build_tree(idxs[:middle_val], depth + 1)
        right_branch = self._build_tree(idxs[middle_val + 1:], depth + 1) # Syntax reparied by ai

        return KDNode (point=median_point, axis=axis, left=left_branch, right=right_branch, point_index=median_idx)

    def get_NN(self, query_point, knn):
       best = []
       self.find_knn(self.root, query_point, knn + 1, best)
       # Exclude the query point itself from the results
       return best[1:]  

    def find_knn(self, node, query_point, knn_plus_1, best):
        if node is None:
            return
        
        # Calculate the distance between the query point and the current node's point
        dx = node.point[0] - query_point[0]
        dy = node.point[1] - query_point[1]
        dz = node.point[2] - query_point[2]

        distance = (dx ** 2 + dy ** 2 + dz ** 2) ** 0.5
        
        # Adding distance and p_index to the best list if its not full

        if len(best) < knn_plus_1:
            best.append((distance, node.point_index))
            best.sort()
        else:
            # Checking if the final searched distance is less than the maximum distance in list
            if distance < best[-1][0]:
                best[-1] = (distance, node.point_index)
                best.sort()

        # Determining which branch of the tree to search
        # if the result of this calculation is negative, the point should be in the left branch
        
        diff = query_point[node.axis] - node.point[node.axis]
        if diff <= 0:
            near = node.left
            far = node.right
        else:
            near = node.right
            far = node.left
        
        self.find_knn(near, query_point, knn_plus_1, best)
        
        if len(best) < knn_plus_1 or abs(diff) < best[-1][0] :
            self.find_knn(far, query_point, knn_plus_1, best)

        return (best)

# Computing density
# Starting timer

t_start_1 = process_time()

# Bulding the KDtree

kdtree = KDTree(X, Y, Z)

# Find distance for each point to compute point cloud density
sum_distance = 0

for i in range(len(X)):
    neighbours = kdtree.get_NN(kdtree.data[i], 1)
    d_min = neighbours[0][0]
    sum_distance += d_min

d_aver = sum_distance / len(X)
rho = 1 / (d_aver**3)

# Stop timer

t_end_1 = process_time()

# Computing Curvature
# Start timer

t_start_2 = process_time()

#

curvatures = []

for i in range(len(X)):
    neighbours = kdtree.get_NN(kdtree.data[i], 30)
    indices = [n[1] for n in neighbours]
    knn_coords = kdtree.data[indices]

    cov_matrix = cov(knn_coords, rowvar = False)

    eigvals = linalg.eigvalsh(cov_matrix)
    l1 = eigvals[0]
    l2 = eigvals[1]
    l3 = eigvals[2]

    if (l1 + l2 + l3) > 0:
        k = l1 / (l1 + l2 + l3)
    else:
        k = 0  
            
    curvatures.append(k)

# Stop timer

t_end_2 = process_time()

# Characteristics for density and curvature 

print("Average distance:", d_aver)
print("Density (rho):", rho)
print("Processing time for density:", t_end_1 - t_start_1, "s")
print("Processing time for curvature:", t_end_2 - t_start_2, "s")

# Visualization of the curvature using a scatter plot with color mapping based on curvature values
print ("Visualization of curvature using a scatter plot with color mapping based on curvature values")

fig = figure()
ax = axes(projection = "3d")

vis = ax.scatter(X, Y, Z, c = curvatures, cmap = "plasma", s = 1)
fig.colorbar (vis, label = "Approximated Curvature")
show()



# ________________PART 3: Plot of calculation time on number of points______________________
print("Generating the plot of calculation time on number of points.")

# List of test number of points (50 to all points)
test = [50, 100, 250, 500, 1000, 2500, 5000, 7500, 10000, 15000, len(X)]
test_time_sample = []

for sample in test:
    # Select a subset of points
    X_sample = X[:sample]
    Y_sample = Y[:sample]
    Z_sample = Z[:sample]

    # Start timer for the current sample
    t_start_sample = process_time()

    # Build KD-Tree for the current subset of points
    kdtree_sample = KDTree(X_sample, Y_sample, Z_sample)

    # Search for 1-NN for each point in the subset (same logic as density computation)
    for i in range(sample):
        neighbours = kdtree_sample.get_NN(kdtree_sample.data[i], 1)
        d_min = neighbours[0][0]

    # Stop timer and save elapsed time
    t_end_sample = process_time()
    test_time_sample.append(t_end_sample - t_start_sample)

# Plot computing time as a function of the number of points
fig = figure()
plot(test, test_time_sample, '-', label='KD-Tree')
xlabel("Number of points")
ylabel("Time [s]")
title("Plot of computing time on the number of points")
grid(True)
legend()
show()