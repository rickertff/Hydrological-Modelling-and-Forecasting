#
# https://doi.org/10.1016/S0022-1694(03)00225-7
#

import numpy as np
import pandas as pd

P = 0
E = 0
P_n = 0.00
E_n = 0.00
S = 0

x_1 = 0
x_2 = 0
x_3 = 0
x_4 = 0



if P >= E:
    P_n = P - E
    E_n = 0.00
else:
    P_n = 0.00
    E_n = E - P

# Equation 3: production store
P_s = (x_1*(1-(S/x_1)**2)*np.tanh(P_n/x_1))/(1+ (S/x_1)*np.tanh(P_n/x_1))

# Equation 4: storage evaporation
E_s = (S*(2-(S/x_1))*np.tanh(E_n/x_1))/(1+(1-(S/x_1))*np.tanh(E_n/x_1))

# Equation 5: water content in the production store
S = S - E_s + P_s

# Equation 6: perculation equation:
Perc = S(1-(1+((4/9)*(S/x_1)**4))**(-0.25))

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

UH1(j) = SH1(j) - SH1(j-1)
UH2(j) = SH2(j) - SH2(j-1)





