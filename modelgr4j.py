#
# https://doi.org/10.1016/S0022-1694(03)00225-7
#

import numpy as np
import pandas as pd

t = 0
dt = 1

P = 2.8
E = 0.4
P_n = 0.00
E_n = 0.00
x_1 = 0.7
x_2 = 4.0
x_3 = 5.0
x_4 = 5
S = 0
R = 0 


precipitation = pd.read_excel("C:\\Users\\pc-ri\\Documents\\Uni\\18 - Q6 TEM\\Hydrological Modelling and Forecasting\\Assignments\\Hydrological-Modelling-and-Forecasting\\Observed time series 1968-1982.xlsx", 0)
evapotranspiration = pd.read_excel("C:\\Users\\pc-ri\\Documents\\Uni\\18 - Q6 TEM\\Hydrological Modelling and Forecasting\\Assignments\\Hydrological-Modelling-and-Forecasting\\Observed time series 1968-1982.xlsx", 1)
discharge = pd.read_excel("C:\\Users\\pc-ri\\Documents\\Uni\\18 - Q6 TEM\\Hydrological Modelling and Forecasting\\Assignments\\Hydrological-Modelling-and-Forecasting\\Observed time series 1968-1982.xlsx", 2)

def update_timestep(t, P, E, P_n, E_n, S, x_1, x_2, x_3, x_4):
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

def fun_UH1(x_4):
    SH1 = []
    UH1 = [0.0]
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
    UH2 = [0.0]
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
    
UH1 = fun_UH1(x_4)
UH2 = fun_UH2(x_4)


for t in range(len(precipitation)-2):
    P = precipitation.iat[t+2, 5]
    E = evapotranspiration.iat[t+2, 5]
    update_timestep(t, P, E, P_n, E_n, S, x_1, x_2, x_3, x_4)


