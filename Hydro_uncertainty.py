import numpy as np
import pandas as pd
import os
import scipy.stats as ss
import matplotlib.pyplot as plt

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    [low, up] = ss.t.interval(confidence, len(a)-1, loc=np.mean(a), scale=ss.sem(a))
    return low, up

# Observed
path1 = os.path.join(os.path.dirname(__file__), "Observed time series 1968-1982.xlsx")
discharge = pd.read_excel(path1, 2, header=2)
discharge_lesse = discharge["Lesse"]

# Simulated
total_discharge = pd.read_csv("Simulated series.txt", header=None, index_col=None)

# Monte Carlo
monte_carlo = pd.read_csv("All_Series.txt", header=None, delimiter=" ")
monte_carlo.to_numpy()

warmup = 365
length = 6*365
intervals_lower95 = []
intervals_upper95 = []
intervals_lower50 = []
intervals_upper50 = []

# Confidence interval
for i in range(length):
    col = monte_carlo[i]
    values = []
    for j in col:
        values.append(j)
    [lower95, upper95] = mean_confidence_interval(col, 0.95)
    intervals_lower95.append(lower95)
    intervals_upper95.append(upper95)

    #[lower50, upper50] = mean_confidence_interval(col, 0.5)
    #intervals_lower50.append(lower50)
    #intervals_upper50.append(upper50)

x = np.linspace(warmup+1, length+1, length-warmup, False)

plt.plot(discharge_lesse[warmup:length])
plt.plot(total_discharge[:length])
plt.fill_between(x, intervals_lower95[warmup:], intervals_upper95[warmup:], alpha=0.5, color='black')
plt.ylabel("Discharge (m^3/s)")
plt.xlabel("Time (Days)")
plt.legend(["Measured", "Simulated", "95% Confidence Interval"])
plt.title("Measured versus simulated discharge")
plt.xlim([warmup, length])
plt.ylim([0, 200])
plt.show()