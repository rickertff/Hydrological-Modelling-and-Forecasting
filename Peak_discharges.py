import numpy as np
import pandas as pd
import scipy.stats as ss

monte_carlo = pd.read_csv("All_Series.txt", header=None, delimiter=" ")
#monte_carlo.to_numpy()
years = [274, 639, 1004, 1369, 1734, 2099]
peaks = []

for i in range(monte_carlo.shape[0]):
    for j in range(len(years)-1):
        values = monte_carlo.iloc[i,years[j]:years[j+1]]
        maxi = np.max(values)
        peaks.append(maxi)
avg_peaks = np.average(peaks)
print(avg_peaks)

lijst = []
for i in range(monte_carlo.shape[0]):
    for j in range(monte_carlo.shape[1]):
        lijst.append(monte_carlo.iloc[i,j])
lijst.sort(reverse=True)
pos = int(0.1*len(lijst))
print(pos)
q90 = lijst[pos]
print(q90)