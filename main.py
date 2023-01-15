import time
from mpi4py import MPI
import random

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

numPoints = 10000

# Main process generates points
if rank == 0:
    startTime = time.time()
    points = [(random.random(), random.random()) for _ in range(numPoints)]
    # Divide the points among the processes
    chunks = [points[i::size] for i in range(size)]
else:
    chunks = None

startTimeLocal = time.time()
# Scatter the points to the processes
points = comm.scatter(chunks, root=0)

# Each process counts the number of points inside the unit circle
pointsInCircle = sum(x * x + y * y <= 1 for x, y in points)

localPiEstimate = 4 * pointsInCircle / len(points)
localTime = time.time() - startTimeLocal

# Gather the results from all processes
results = comm.gather((localPiEstimate, localTime), root=0)

globalPiEstimate = sum(result[0] for result in results) / size
globalTime = time.time() - startTime
print("Rank: ", rank, "\nNumber of points: ", numPoints, "\nTotal time: ", globalTime,
      "\nPi value: ", globalPiEstimate)
