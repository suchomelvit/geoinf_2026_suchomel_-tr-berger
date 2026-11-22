# Prohlášení o používaní AI:
# Nástorje AI (Google Gemini) byly použity pouze pro návrh postupu
# a případné debugování kódu. Části, kde byla AI využita jsou v kódu popsány 



from math import *
from time import *
from matplotlib.pyplot import *
from numpy import *

# Load points from .txt as list

def loadPoints(file):

    X, Y, Z = [], [], []
    
    with open(file) as f:
        
        for line in f:
            x, y, z = line.split('\t')
            
            X.append(float(x))
            Y.append(float(y))
            Z.append(float(z))
    
    return X, Y, Z

# Step 1: Find NN for each point

def findNN(xq, yq, zq, X, Y, Z):
    # Set initial minimum distance to infinity
    dmin = inf
    # Placeholder values for NN point are set 
    xn, yn, zn, = X[0], Y[0], Z[0]

    # Process all points from first instance
    for i in range(len(X)):

        # Compute distance between each point and the query point
        dx, dy, dz = xq - X[i], yq - Y[i], zq - Z[i]
        d = (dx*dx + dy*dy + dz*dz)**0.5 #sqrt(dx^2 + dy^2 + dz^2)
        
        # Update coordinates for NN for each point based on minimum distance 
        if d < dmin and d > 0:
            dmin = d
            xn, yn, zn = X[i], Y[i], Z[i]
    
    # Return the coordinates of the nearest neighbor and the minimum distance        
    return xn, yn, zn, dmin
                    
# Step 2: Visualize the points in 3D space

def drawTree(X, Y, Z, sx, transp = 0.2):
    # Create figure
    fig = figure()
    ax = axes(projection = '3d')
    ax.set_aspect('equal')

    # Compute sphere scale: 1 pix = 25.4 mm
    scale = 1
    if sx > 0:
            scale = int(sx * sx * 40 * 40)
            
    # Plot points
    ax.scatter(X, Y, Z, s=scale, alpha = transp)
    
    show()
    
# Load points
X, Y, Z = loadPoints('tree_18.txt')

# Draw points
drawTree(X, Y, Z, 0, 0.2)

# Initialization
print("Number of points:", len(X))
print("Computing density with Naive Method")

# Step 3: Compute Density using "Naive Search" -> basic method -> takes dmin of all points/number of points

# Measuring processing time
proc_start = process_time()

# Starting from zero distance (then continually adding distances of all NN)
sum_distance = 0

# Finding distance for each point 
for i in range(len(X)):
    # Using function findNN to find nearest neighbour for point i
    # It returns coordinates (xn, yn, zn) and minimal distance (dmin)
    xn, yn, zn, dmin = findNN(X[i], Y[i], Z[i], X, Y, Z)
    
    # Adding the minimum distance to the total sum of distances
    sum_distance += dmin

# Calculate average distance

dmin_aver = sum_distance / len(X)

# Calculate density: rho = 1 / d^3 (can not devide by zero)
if dmin_aver > 0:
    rho = 1 / (dmin_aver**3)
else:
    rho = 0

# Stop measuring of processing time
proc_stop = process_time()

# Characteristics of point cloud
print("Average distance: ", dmin_aver)
print("Density: ", rho)
print("Processing time: ", proc_stop - proc_start, "s")

# Step 4: Visualization of calculation time on number of points

print("Generating the plot of calculation time on number of points.")

points_list = []
# List of test number of points (50 to 10000)
test = [50,100,250,500,1000,2500,5000,7500,10000]
# Blank list of calculated time for each sample
test_time = []


for sample in test:
    X_sample = X[:sample]
    Y_sample = Y[:sample]
    Z_sample = Z[:sample]

    proc_start_sample = process_time()

    sum_dist_test = 0
    
    # Go through all points
    for i in range(sample):
        xn, yn, zn, dmin = findNN(X_sample[i], Y_sample[i], Z_sample[i], X_sample, Y_sample, Z_sample)
    
        sum_dist_test += dmin

    dmin_aver = sum_dist_test / sample

    if dmin_aver > 0:
        rho= 1 / (dmin_aver**3)
    else:
        rho = 0

    proc_stop_sample = process_time()

    time = proc_stop_sample - proc_start_sample
    test_time.append(time)


# Generating the plot

fig = figure()

plot(test, test_time, '-', label='Naive Method')

xlabel("Number of points")
ylabel("Time [s]")
title("Plot of computing time on the number of points")
grid(True)
legend()

show()




