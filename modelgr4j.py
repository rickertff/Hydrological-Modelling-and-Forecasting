#
# https://doi.org/10.1016/S0022-1694(03)00225-7
#

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import random as rd
import scipy.stats as ss

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

def KGE(Qmodel, Qreal, warmup):
    r = ss.pearsonr(Qmodel[warmup:], Qreal[warmup:])
    a = np.std(Qmodel[warmup:]) / np.std(Qreal[warmup:])
    b = np.mean(Qmodel[warmup:]) / np.mean(Qreal[warmup:])

    KGE_value = 1 - np.sqrt((r[0]-1)**2 + (a-1)**2 + (b-1)**2)

    return KGE_value

def NSE(Qmodel, Qreal, warmup):
    part1 = 0
    part2 = 0
    for i in range(len(Qmodel)):
        if i > warmup:
            part1 += (Qmodel[i] - Qreal[i])**2
            part2 += (Qreal[i] - np.mean(Qreal))**2
    
    NSE_value = 1 - part1 / part2

    return NSE_value

def RVE(Qmodel, Qreal, warmup):

    RVE_value = ((sum(Qmodel[warmup:]) - sum(Qreal[warmup:])) / sum(Qreal[warmup:])) * 100

    return RVE_value

#
# Init
#

sens_analysis = False
calibration = False
warmup = 365
fout = False

t = 0
dt = 1
x_1 = 350
sensx1 = [100, 183, 266, 350, 633, 916, 1200]
x_2 = 0
sensx2 = [-5, -3.3, -1.6, 0, 1, 2, 3]
x_3 = 90
sensx3 = [20, 43, 67, 90, 160, 230, 300]
x_4 = 1.7
sensx4 = [1.1, 1.3, 1.5, 1.7, 2.1, 2.5, 2.9]
S = 0
R = 0

area = 1311

path = os.path.join(os.path.dirname(__file__), "Observed time series 1968-1982.xlsx")
precipitation = pd.read_excel(path, 0, header=2)
precip_lesse = precipitation["Lesse"]
# Constant precipitation
#precip_lesse = np.zeros(sens_length)
#for i in range(100):
#    precip_lesse[i] = 8
precip_total_9 = np.zeros(len(precip_lesse)+12)
precip_total_1 = np.zeros(len(precip_lesse)+12)

evapotranspiration = pd.read_excel(path, 1, header=2)
evap_lesse = evapotranspiration["Lesse"]
#evap_lesse = np.zeros(sens_length)
#for i in range(100):
#    evap_lesse[i] = 2

discharge = pd.read_excel(path, 2, header=2)
discharge_lesse = discharge["Lesse"]
sens_length = len(precip_lesse)
total_discharge = np.zeros(len(precip_lesse))
total_discharge1 = np.zeros([7, sens_length])
total_discharge2 = np.zeros([7, sens_length])
total_discharge3 = np.zeros([7, sens_length])
total_discharge4 = np.zeros([7, sens_length])
KGE_values = np.zeros([4, 7])

#
# Calculate Unit Hydrographs
#

Q_1 = np.zeros([len(precip_lesse), len(precip_lesse)+12])
Q_9 = np.zeros([len(precip_lesse), len(precip_lesse)+12])

#
# Main
#

if sens_analysis:
    for i in range(len(sensx1)):
        x_1 = sensx1[i]
        x_2 = 0
        x_3 = 90
        x_4 = 1.7
        UH1 = fun_UH1(x_4)
        UH2 = fun_UH2(x_4)
        for t in range(sens_length):
            P = precip_lesse.iat[t]
            E = evap_lesse.iat[t]
            [R, S, Q] = update_timestep(t, P, E, R, S, x_1, x_2, x_3, UH1, UH2)
            total_discharge1[i, t] = Q / 86400 * area * 1000
            KGE_values[0, i] = KGE(total_discharge1[i,:], discharge_lesse, warmup)
            
    for i in range(len(sensx2)):
        x_1 = 350
        x_2 = sensx2[i]
        x_3 = 90
        x_4 = 1.7
        UH1 = fun_UH1(x_4)
        UH2 = fun_UH2(x_4)
        for t in range(sens_length):
            P = precip_lesse.iat[t]
            E = evap_lesse.iat[t]
            [R, S, Q] = update_timestep(t, P, E, R, S, x_1, x_2, x_3, UH1, UH2)
            total_discharge2[i, t] = Q / 86400 * area * 1000
            KGE_values[1, i] = KGE(total_discharge2[i,:], discharge_lesse, warmup)

    for i in range(len(sensx3)):
        x_1 = 350
        x_2 = 0
        x_3 = sensx3[i]
        x_4 = 1.7
        UH1 = fun_UH1(x_4)
        UH2 = fun_UH2(x_4)
        for t in range(sens_length):
            P = precip_lesse.iat[t]
            E = evap_lesse.iat[t]
            [R, S, Q] = update_timestep(t, P, E, R, S, x_1, x_2, x_3, UH1, UH2)
            total_discharge3[i, t] = Q / 86400 * area * 1000
            KGE_values[2, i] = KGE(total_discharge3[i,:], discharge_lesse, warmup)

    for i in range(len(sensx4)):
        x_1 = 350
        x_2 = 0
        x_3 = 90
        x_4 = sensx4[i]
        UH1 = fun_UH1(x_4)
        UH2 = fun_UH2(x_4)
        for t in range(sens_length):
            P = precip_lesse.iat[t]
            E = evap_lesse.iat[t]
            [R, S, Q] = update_timestep(t, P, E, R, S, x_1, x_2, x_3, UH1, UH2)
            total_discharge4[i, t] = Q / 86400 * area * 1000
            KGE_values[3, i] = KGE(total_discharge4[i,:], discharge_lesse, warmup)

elif calibration:
    bestKGE = 0.7
    bestx1 = 248
    bestx2 = -1.21
    bestx3 = 71
    bestx4 = 2.19
    rd.seed(0)

    for k in range(100):
        randpar = rd.randint(1,4)
        if randpar == 1:
            x_1 = rd.normalvariate(bestx1, 50)
            while x_1 < 100:
                x_1 = rd.normalvariate(bestx1, 50)
        elif randpar == 2:
            x_2 = rd.normalvariate(bestx2, 0.5)
        elif randpar == 3:
            x_3 = rd.normalvariate(bestx3, 10)
        else:
            x_4 = rd.normalvariate(bestx4, 0.5)
        second = rd.randint(0,1)
        if second == 1:
            randpar2 = rd.randint(1,4)
            while randpar2 == randpar:
                randpar2 = rd.randint(1,4)
            if randpar2 == 1:
                x_1 = rd.normalvariate(bestx1, 50)
                while x_1 < 100:
                    x_1 = rd.normalvariate(bestx1, 50)
            elif randpar2 == 2:
                x_2 = rd.normalvariate(bestx2, 0.5)
            elif randpar2 == 3:
                x_3 = rd.normalvariate(bestx3, 10)
            else:
                x_4 = rd.normalvariate(bestx4, 0.5)
        
        UH1 = fun_UH1(x_4)
        UH2 = fun_UH2(x_4)
        for t in range(len(precip_lesse)):
            if not fout:
                P = precip_lesse.iat[t]
                E = evap_lesse.iat[t]
                [R, S, Q] = update_timestep(t, P, E, R, S, x_1, x_2, x_3, UH1, UH2)
                total_discharge[t] = Q / 86400 * area * 1000
            else:
                break
        if not fout:
            KGE_value = KGE(total_discharge, discharge_lesse, warmup)
            RVE_value = RVE(total_discharge, discharge_lesse, warmup)
            if KGE_value > bestKGE and abs(RVE_value) < 5:
                bestKGE = KGE_value
                bestRVE = RVE_value
                bestx1 = x_1
                bestx2 = x_2
                bestx3 = x_3
                bestx4 = x_4
                bestdischarge = total_discharge
        else:
            fout = False
    NSE_value = NSE(total_discharge, discharge_lesse, warmup)
    print(bestKGE)
    print(bestRVE)
    print(NSE_value)
    print(bestx1)
    print(bestx2)
    print(bestx3)
    print(bestx4)
    np.savetxt("Calibrated discharge.txt", bestdischarge)

else:
    UH1 = fun_UH1(x_4)
    UH2 = fun_UH2(x_4)
    for t in range(len(precip_lesse)):
        P = precip_lesse.iat[t]
        E = evap_lesse.iat[t]
        [R, S, Q] = update_timestep(t, P, E, R, S, x_1, x_2, x_3, UH1, UH2)
        total_discharge[t] = Q / 86400 * area * 1000
    KGE_value = KGE(total_discharge, discharge_lesse, warmup)
    RVE_value = RVE(total_discharge, discharge_lesse, warmup)
    NSE_value = NSE(total_discharge, discharge_lesse, warmup)
    print(KGE_value)
    print(RVE_value)
    print(NSE_value)

if sens_analysis:
    np.savetxt("KGE_Values.txt", KGE_values)
    for i in range(4):
        plt.plot(KGE_values[i,:])
    plt.ylabel("KGE Value")
    plt.xlabel("Parameter value")
    plt.legend(["X1", "X2", "X3", "X4"])
    plt.title("Sensitivity of model to KGE values")
    #for j in range(len(sensx3)):
        #plt.plot(total_discharge3[j,:])
        #plt.ylabel("Discharge (m^3/s)")
        #plt.xlabel("Time (Days)")
elif calibration:
    plt.plot(discharge_lesse)
    plt.plot(bestdischarge)
    plt.ylabel("Discharge (m^3/s)")
    plt.xlabel("Time (Days)")
    plt.legend(["Measured", "Simulated"])
else:
    plt.plot(discharge_lesse)
    plt.plot(total_discharge)
    plt.ylabel("Discharge (m^3/s)")
    plt.xlabel("Time (Days)")
    plt.legend(["Measured", "Simulated"])
plt.show()