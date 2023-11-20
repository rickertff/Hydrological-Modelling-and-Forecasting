#
# https://doi.org/10.1016/S0022-1694(03)00225-7
#

import numpy as np
import pandas as pd

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

    # Equation 9 - 15:

    
    if t <= 0:
        SH1 = 0
    elif t > 0 and t < x_4:
        SH1 = (t/x_4)**(5/2)
    else:
        SH1 = 1

    if t <= 0:
        SH2 = 0
    elif t > 0 and t <= x_4:
        SH2 = 0.5*(t/x_4)**(5/2)
    elif t > x_4 and t <= 2*x_4:
        SH2 = 1-(0.5*(2-(t/x_4))**(5/2))
    else:
        SH2 = 1
    """
    UH1(j) = SH1(j) - SH1(j-1)
    UH2(j) = SH2(j) - SH2(j-1)
    """
    return

P = 0
E = 0
P_n = 0.00
E_n = 0.00
S = 0

x_1 = 350
x_2 = 0
x_3 = 90
x_4 = 1.7

for t in range(len(precipitation)-2):
    P = precipitation.iat[t+2, 5]
    E = evapotranspiration.iat[t+2, 5]
    update_timestep(t, P, E, P_n, E_n, S, x_1, x_2, x_3, x_4)


