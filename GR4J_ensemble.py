#
# https://doi.org/10.1016/S0022-1694(03)00225-7
#

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

#
# Functions
#

def update_timestep(t, P, E, R, S, x_1, x_2, x_3, UH1, UH2):
    """
    This function calculates the new water balance according to the equations.
    """
    if P >= E:
        P_n = P - E
        E_n = 0.00
    else:
        P_n = 0.00
        E_n = E - P
    
    # Equation 3: production store
    P_s = (x_1*(1-(S/x_1)**2)*np.tanh(P_n/x_1))/(1+(S/x_1)*np.tanh(P_n/x_1))

    # Equation 4: storage evaporation
    E_s = (S*(2-(S/x_1))*np.tanh(E_n/x_1))/(1+(1-(S/x_1))*np.tanh(E_n/x_1))

    # Equation 5: water content in the production store
    S = S - E_s + P_s

    # Equation 6: perculation equation:
    Perc = S*(1-(1+((4/9)*(S/x_1)**4))**(-0.25))

    # Equation 7: change in resevoir content
    S = S - Perc

    # Equation 8: routing equation
    P_r = Perc + (P_n - P_s)

    for j in range(len(UH1)):
        Q_1[t, t+j] = 0.1 * P_r * UH1[j]
    for j in range(len(UH2)):
        Q_9[t, t+j] = 0.9 * P_r * UH2[j]

    F = x_2*(R/x_3)**(7/2)

    precip_total_9[t] = sum(Q_9[:,t])
    precip_total_1[t] = sum(Q_1[:,t])

    R = max((R + precip_total_9[t] + F),0)

    Q_r = R*(1-(1+(R/x_3)**4)**(-1/4))
    if Q_r < R:
        R = R - Q_r
    else:
        print("Hehe dit gaat fout")
        fout = True
        return 0, 0, 0

    Q_d = max((precip_total_1[t] + F), 0)

    Q = Q_r + Q_d
    
    return R, S, Q

def fun_UH1(x_4):
    SH1 = []
    UH1 = []
    dt1 = 1
    for t in np.arange(0.0, (x_4+2*dt1), dt1):
        if t <= 0:
            SH1.append([t,0])
        elif t > 0 and t < x_4:
            SH1.append([t, (t/x_4)**(5/2)])  
        else:
            SH1.append([t, 1])
    for j in range(1,len(SH1),1):
       UH1.append(SH1[j][1] - SH1[(j-1)][1]) 
    return UH1

def fun_UH2(x_4):
    SH2 = []
    UH2 = []
    dt1 = 1
    for t in np.arange(0,(2*x_4+2*dt1), dt1):
        if t <= 0:
            SH2.append([t,0]) 
        elif t > 0 and t <= x_4:
            SH2.append([t, 0.5*(t/x_4)**(5/2)])
        elif t > x_4 and t <= 2*x_4:
            SH2.append([t, 1-(0.5*(2-(t/x_4))**(5/2))]) 
        else:
            SH2.append([t,1])
    for j in range(1,len(SH2),1):
       UH2.append(SH2[j][1] - SH2[(j-1)][1]) 
    return UH2

#
# Init
#
warmup = 365
t = 0
dt = 1
x_1 = 166.1
x_2 = -1.57
x_3 = 40.3
x_4 = 2.26
S = 0
R = 0
area = 1311

path1 = os.path.join(os.path.dirname(__file__), "Observed time series 1968-1982.xlsx")
precipitation = pd.read_excel(path1, 0, header=2)
precip_lesse = precipitation["Lesse"]
precip_total_9 = np.zeros(len(precip_lesse)+12)
precip_total_1 = np.zeros(len(precip_lesse)+12)

evapotranspiration = pd.read_excel(path1, 1, header=2)
evap_lesse = evapotranspiration["Lesse"]

#evap_lesse = np.zeros(verif_length)

discharge = pd.read_excel(path1, 2, header=2)
discharge_lesse = discharge["Lesse"]
total_discharge = np.zeros(len(precip_lesse))

#
# Calculate Unit Hydrographs
#

Q_1 = np.zeros([len(precip_lesse), len(precip_lesse)+12])
Q_9 = np.zeros([len(precip_lesse), len(precip_lesse)+12])

#
# Main
#
 
UH1 = fun_UH1(x_4)
UH2 = fun_UH2(x_4)
for t in range(len(precip_lesse)):
    P = precip_lesse.iat[t]
    E = evap_lesse.iat[t]
    [R, S, Q] = update_timestep(t, P, E, R, S, x_1, x_2, x_3, UH1, UH2)
    total_discharge[t] = Q / 86400 * area * 1000

plt.plot(discharge_lesse[warmup:])
plt.plot(total_discharge)
plt.ylabel("Discharge (m^3/s)")
plt.xlabel("Time (Days)")
plt.legend(["Measured", "Simulated"])
plt.title("Measured versus simulated discharge")
plt.xlim([warmup, len(discharge_lesse)])
plt.show()
