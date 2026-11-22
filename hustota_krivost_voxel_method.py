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
        
# Step 1: Devide points into voxels

# Create function that divides the grid into voxels
def voxelize(X, Y, Z, numVox):
    
    # Set max and min values for X Y Z
    min_X = min(X)
    max_X = max(X)
    min_Y = min(Y)
    max_Y = max(Y)
    min_Z = min(Z)
    max_Z = max(Z)
    
    # Compute size of maximum bounding box based on X Y Z
    box_dx = max_X - min_X
    box_dy = max_Y - min_Y
    box_dz = max_Z - min_Z
    
    
    # Compute size of each voxel
    vx = (box_dx / numVox)
    vy = (box_dy / numVox)
    vz = (box_dz / numVox)

    # Compute reduced coordinates
    xr = ((array(X) - min_X) / (max_X - min_X)) # we had a problem with computing 
    yr = ((array(Y) - min_Y) / (max_Y - min_Y)) # xr, yr, zr. AI suggested using array
    zr = ((array(Z) - min_Z) / (max_Z - min_Z)) # array converts list to vector we can compute with

    return (vx, vy, vz, xr, yr, zr)

# Step 2: Compute 3D voxel grid coordinates (jx, jy, jz) for a single point [x, y, z]
def CPIndex(numVox, xr, yr, zr):

    # Rouding constant to avoid index out of bounds
    c = 0.9999999999999
    
    jx = int(c * numVox * xr) 
    jy = int(c * numVox * yr)
    jz = int(c * numVox * zr)

    return (jx, jy, jz)

# Convert 3D voxel coordinates (jx, jy, jz) into a unique 1D voxel ID (hash key)
def hash1D(jx, jy, jz, numVox):
    h1D = jx + jy * numVox + jz * (numVox**2)
    return (h1D)

# Assign all points from the point cloud into voxels (build spatial index H)
def assign3DIndex(numVox, X, Y, Z ):
    # Create dictionary for 3D index
    H = {}

    # Assign every point in len(X)
    vx, vy, vz, xr, yr, zr = voxelize(X, Y, Z, numVox)

    for i in range (len(X)):
        jx, jy, jz = CPIndex(numVox, xr[i], yr[i], zr[i])
        
        # Convert 3D coordinates to unique 1D hash key
        h1D = hash1D(jx, jy, jz, numVox)
        
        # Create list in dictionary if this voxel is empty so far
        if h1D not in H:
            H[h1D] = []
            
        H[h1D].append(i)
        
    return H, xr, yr, zr

# Step 3: Initialization 
#Load points
X, Y, Z = loadPoints('tree_18.txt')

# Show number of points in the Tree
print("Number of points:" , len(X))

# ___________________PART 1: Computing density using Voxels______________________
print("Computing density of points using voxels")

# Compute n of voxels in maximum bounding box
numVox = int(len(X)**(1/3))

# Time commitment
t_start_1 = process_time()

# Build spatial index H and retrieve reduced coordinates
H, xr, yr, zr = assign3DIndex(numVox, X, Y, Z)

# Find 1-NN (nearest neighbor) for each point to compute point cloud density
sum_distance = 0

for i in range(len(X)):
    # Get 3D voxel grid coordinates for the current point i
    jx, jy, jz = CPIndex(numVox, xr[i], yr[i], zr[i])
    
    d_min = inf  # Initialize minimum distance to infinity
    r = 1        # Search radius in voxels (r = 1 means a 3x3x3 neighborhood)
    
    # Expand search radius until at least one neighbor is found
    while d_min == inf:
        # Loop through all neighboring voxels within radius r
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                for dz in range(-r, r + 1):
                    nx = jx + dx
                    ny = jy + dy
                    nz = jz + dz
                    
                    # Check if the neighboring voxel is inside the grid boundaries
                    if 0 <= nx < numVox and 0 <= ny < numVox and 0 <= nz < numVox:
                        # Convert 3D voxel coordinates to 1D hash key
                        h1D = hash1D(nx, ny, nz, numVox)
                        
                        # Check if the voxel contains any points
                        if h1D in H:
                            # Compute distance to every point in this voxel
                            for j in H[h1D]:
                                if j == i:
                                    continue  # Skip the point itself (distance would be 0)
                                
                                # 3D Euclidean distance
                                dist = sqrt((X[i] - X[j])**2 + (Y[i] - Y[j])**2 + (Z[i] - Z[j])**2)
                                
                                # Update minimum distance if a closer point is found
                                if dist < d_min:
                                    d_min = dist
        
        r = r + 1  # Increase search radius if no neighbor was found in this layer
        
    # Add the nearest neighbor distance of point i to the total sum
    sum_distance = sum_distance + d_min

# Compute average nearest neighbor distance and spatial density
d_aver = sum_distance / len(X)
rho = 1 / (d_aver**3)

# Stop timer

t_end_1 = process_time()

# ________________________PART 2: Compute curvature_______________________________ 

curvature = []
knn = 30

# Start timer for curvature computation

t_start_2 = process_time()

# We will compute curvature for each point in the point cloud
for i in range(len(X)):
    # Get 3D voxel grid coordinates for the current point i
    jx, jy, jz = CPIndex(numVox, xr[i], yr[i], zr[i])
    
    candidates = [] # We have to reset the candidates list for each point¨
                    # otherwise it will keep accumulating points from previous iterations

    # Search for neighbors in the surrounding voxels
    r = 1  # Start with a search radius of 1 voxel

    # Search until we find enough neighbors
    while len(candidates) < knn:
        candidates = [] # We have to reset the candidates list for each point¨
                        # otherwise it will keep accumulating points from previous iterations
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                for dz in range(-r, r + 1):
                    nx = jx + dx
                    ny = jy + dy
                    nz = jz + dz
                    
                    # Check if the neighboring voxel is inside the grid boundaries
                    if 0 <= nx < numVox and 0 <= ny < numVox and 0 <= nz < numVox:
                        # Convert 3D voxel coordinates to 1D hash key
                        h1D = hash1D(nx, ny, nz, numVox)
                        
                        # Check if the voxel contains any points
                        if h1D in H:
                            for j in H[h1D]:
                                if j == i:
                                    continue  # Skip the point itself
                                
                                dist = sqrt((X[i] - X[j])**2 + (Y[i] - Y[j])**2 + (Z[i] - Z[j])**2)
                                # Save both distance and point index as a pair (tuple)
                                candidates.append((dist, j))
        
        r += 1  # Increase search radius if not enough neighbors found

        
    # Sort candidates by distance (from smallest to largest)
    candidates.sort()
        
    # Keep only the first 30 nearest neighbors
    nearest = candidates[:knn]

    # Extract X, Y, Z coordinates of the 30 nearest neighbors
    knn_X = []
    knn_Y = []
    knn_Z = []

    # Append the coordinates of the nearest neighbors to the respective lists
    for dist, j in nearest:
            knn_X.append(X[j])
            knn_Y.append(Y[j])
            knn_Z.append(Z[j])

    # Create X Y Z cords for knn points
    knn_coords = array([knn_X, knn_Y, knn_Z])

    # Creating a covariance matrix
    # AI used here for numpy functions explanation
    cov_matrix = cov(knn_coords)

    # Compute eigenvalues (eigvalsh automatically sorts them: lambda1 <= lambda2 <= lambda3)
    eigvals = linalg.eigvalsh(cov_matrix)
    l1 = eigvals[0]
    l2 = eigvals[1]
    l3 = eigvals[2]

    # Compute approximated curvature
    if (l1 + l2 + l3) > 0:
            k = l1 / (l1 + l2 + l3)
    else:
            k = 0  # Safety check against division by zero
            
    curvature.append(k)   

# Stop processing time for curvature computation
t_end_2 = process_time()

# Characteristics for density and curvature 

print("Average distance:", d_aver)
print("Density (rho):", rho)
print("Processing time for density:", t_end_1 - t_start_1, "s")
print("Processing time for curvature:", t_end_2 - t_start_2, "s")

# Visualization of the curvature using a scatter plot with color mapping based on curvature values
# AI used here for matplotlib functions explanation
print ("Visualization of curvature using a scatter plot with color mapping based on curvature values")

fig = figure()
ax = axes(projection = "3d")

vis = ax.scatter(X, Y, Z, c = curvature, cmap = "plasma", s = 1 )

fig.colorbar(vis, label = "Approximated Curvature")
show()

# ________________PART 3: Plot of calculation time on number of points______________________
print("Generating the plot of calculation time on number of points.")

# List of test number of points (50 to all points)
test = [50, 100, 250, 500, 1000, 2500, 5000, 7500, 10000, 15000, len(X)]
test_time_sample = []
numVox_fixed = 15  # Fixed number of voxels per axis

for sample in test:
    X_sample = X[:sample]
    Y_sample = Y[:sample]
    Z_sample = Z[:sample]

    t_start_sample = process_time()

    # Build spatial index
    H_s, xr_s, yr_s, zr_s = assign3DIndex(numVox_fixed, X_sample, Y_sample, Z_sample)

    # Search for 1-NN (same logic as in PART 1)
    for i in range(sample):
        jx, jy, jz = CPIndex(numVox_fixed, xr_s[i], yr_s[i], zr_s[i])
        d_min = inf
        r = 1

        while d_min == inf:
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    for dz in range(-r, r + 1):
                        nx, ny, nz = jx + dx, jy + dy, jz + dz

                        if 0 <= nx < numVox_fixed and 0 <= ny < numVox_fixed and 0 <= nz < numVox_fixed:
                            h1D = hash1D(nx, ny, nz, numVox_fixed)
                            if h1D in H_s:
                                for j in H_s[h1D]:
                                    if j == i:
                                        continue
                                    dist = sqrt((X_sample[i] - X_sample[j])**2 + (Y_sample[i] - Y_sample[j])**2 + (Z_sample[i] - Z_sample[j])**2)
                                    if dist < d_min:
                                        d_min = dist
            r += 1

    t_end_sample = process_time()
    test_time_sample.append(t_end_sample - t_start_sample)

fig = figure()
plot(test, test_time_sample, '-', label='Voxel Grid (numVox = 15)')
xlabel("Number of points")
ylabel("Time [s]")
title("Plot of computing time on the number of points")
grid(True)
legend()
show()


# _______________PART 4: Plot of calculation time on number of voxels___________________
print("Generating the plot of calculation time on number of voxels.")

# List of test grid resolutions (number of voxels per axis)
test_grid = [5, 10, 15, 20, 25, 30]
test_time_grid = []

for nv in test_grid:
    t_start_grid = process_time()

    # Build spatial index for current voxel grid size nv
    H_g, xr_g, yr_g, zr_g = assign3DIndex(nv, X, Y, Z)

    # Search for 1-NN (same logic as in PART 1)
    for i in range(len(X)):
        jx, jy, jz = CPIndex(nv, xr_g[i], yr_g[i], zr_g[i])
        d_min = inf
        r = 1

        while d_min == inf:
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    for dz in range(-r, r + 1):
                        nx, ny, nz = jx + dx, jy + dy, jz + dz

                        if 0 <= nx < nv and 0 <= ny < nv and 0 <= nz < nv:
                            h1D = hash1D(nx, ny, nz, nv)
                            if h1D in H_g:
                                for j in H_g[h1D]:
                                    if j == i:
                                        continue
                                    dist = sqrt((X[i] - X[j])**2 + (Y[i] - Y[j])**2 + (Z[i] - Z[j])**2)
                                    if dist < d_min:
                                        d_min = dist
            r += 1

    t_end_grid = process_time()
    test_time_grid.append(t_end_grid - t_start_grid)

fig = figure()
plot(test_grid, test_time_grid, '-', label='Voxel Grid (all points)')
xlabel("Number of voxels per axis")
ylabel("Time [s]")
title("Plot of computing time on the number of voxels")
grid(True)
legend()
show()