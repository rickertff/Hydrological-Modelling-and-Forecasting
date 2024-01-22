import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import datetime

#
# Functions
#

def dataimport(path, interval, skiprows):                        
    # Read out the CSV, define index and datacolumn
    df = pd.read_csv(path, header=0, index_col=None, skiprows=skiprows, delimiter=",", parse_dates=[0], skipinitialspace=True, dayfirst=True)          
    # Convert indexcolumn to datetime format                                                                            
    df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y %H:%M',errors='coerce', utc=True, dayfirst=True)      
    # Drop NAN's                                   
    #df.dropna(subset=['date'], inplace=True) 
    # Convert DataFrame to floats (otherwise errors occur)                                                                                 
    #df = pd.to_numeric(df, downcast="float")                                                                  
    # Calculate respective output values based on input parameters, by using interval.
    output = df.groupby(pd.PeriodIndex(df['date'], freq=interval))[df.columns[2:]].sum()
    return output

def ensemble(df, deterministic, observed_P, R, S, x_1, x_2, x_3, x_4):
    UH1 = fun_UH1(x_4)
    UH2 = fun_UH2(x_4)
    counter = 0
    for column_name in df.columns:
        P_ensemble = df[column_name]
        S = 0
        R = 0
        for t in range(len(observed_P)):
            P = observed_P.iat[t, 0]
            E = observed_P.iat[t, 1]
            [R, S, Q] = update_timestep(t, P, E, R, S, x_1, x_2, x_3, UH1, UH2)
            total_discharge[t] = Q / 86400 * area * 1000
        for u in range(len(P_ensemble)):
            P = P_ensemble.iat[u]
            E = 3
            [R, S, Q] = update_timestep(u+t+1, P, E, R, S, x_1, x_2, x_3, UH1, UH2)
            total_discharge[u+t+1] = Q / 86400 * area * 1000
        counter += 1
        output[counter] = total_discharge
        print(counter)
    
    S = 0
    R = 0
    for t in range(len(observed_P)):
        P = observed_P.iat[t, 0]
        E = observed_P.iat[t, 1]
        [R, S, Q] = update_timestep(t, P, E, R, S, x_1, x_2, x_3, UH1, UH2)
        total_discharge[t] = Q / 86400 * area * 1000
    for u in range(len(P_ensemble)):
        P = deterministic.iat[u, 0]
        E = 3
        [R, S, Q] = update_timestep(u+t+1, P, E, R, S, x_1, x_2, x_3, UH1, UH2)
        total_discharge[u+t+1] = Q / 86400 * area * 1000
    counter += 1
    deterministic_output[counter] = total_discharge
    print(counter)

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
        #print("Hehe dit gaat fout")
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
t = 0
dt = 1
x_1 = 166.1
x_2 = -1.57
x_3 = 40.3
x_4 = 1.5
S = 0
R = 0
area = 1311
start_date = '2020-12-13'
end_date = '2021-07-9'
warmup = '2021-06-01'
forecast = '2021-07-19'


plotprocess = True

df_7_9  = dataimport(os.path.join(os.path.dirname(__file__), "forecasts/2021070900.csv"), 'D', 2)
df_7_10 = dataimport(os.path.join(os.path.dirname(__file__), "forecasts/2021071000.csv"), 'D', 2)
df_7_11 = dataimport(os.path.join(os.path.dirname(__file__), "forecasts/2021071100.csv"), 'D', 2)
df_7_12 = dataimport(os.path.join(os.path.dirname(__file__), "forecasts/2021071200.csv"), 'D', 2)
df_7_13 = dataimport(os.path.join(os.path.dirname(__file__), "forecasts/2021071300.csv"), 'D', 2)
df_7_14 = dataimport(os.path.join(os.path.dirname(__file__), "forecasts/2021071400.csv"), 'D', 2)

deter_10 = dataimport(os.path.join(os.path.dirname(__file__), "forecasts/deterministic_10.csv"), 'D', 0)
deter_11 = dataimport(os.path.join(os.path.dirname(__file__), "forecasts/deterministic_11.csv"), 'D', 0)
deter_12 = dataimport(os.path.join(os.path.dirname(__file__), "forecasts/deterministic_12.csv"), 'D', 0)
deter_13 = dataimport(os.path.join(os.path.dirname(__file__), "forecasts/deterministic_13.csv"), 'D', 0)
deter_14 = dataimport(os.path.join(os.path.dirname(__file__), "forecasts/deterministic_14.csv"), 'D', 0)
deter_15 = dataimport(os.path.join(os.path.dirname(__file__), "forecasts/deterministic_15.csv"), 'D', 0)
deter_17 = dataimport(os.path.join(os.path.dirname(__file__), "forecasts/deterministic_17.csv"), 'D', 0)
deter_18 = dataimport(os.path.join(os.path.dirname(__file__), "forecasts/deterministic_18.csv"), 'D', 0)
deter_19 = dataimport(os.path.join(os.path.dirname(__file__), "forecasts/deterministic_19.csv"), 'D', 0)
deter_21 = dataimport(os.path.join(os.path.dirname(__file__), "forecasts/deterministic_21.csv"), 'D', 0)

observed_P = pd.read_excel(os.path.join(os.path.dirname(__file__), "forecasts/as5.xlsx", ), 4, header=0, index_col=None)
observed_P['date'] = pd.to_datetime(observed_P['date'], format='%Y-%m-%d', errors='coerce', utc=True) 
observed_P.set_index('date', inplace=True)
#df_7_11.rename(columns={'2': 'P_Observed'}, inplace=True)

observed_Q = pd.read_excel(os.path.join(os.path.dirname(__file__), "forecasts/as5.xlsx"), 6, header=0, index_col=0)

evap_lesse = observed_P["Evaporation"]

precip_total_1 = np.zeros(len(observed_P)+12)
precip_total_9 = np.zeros(len(observed_P)+12)

Q_1 = np.zeros([len(observed_P), len(observed_P)+12])
Q_9 = np.zeros([len(observed_P), len(observed_P)+12])

observed_P = observed_P.loc[start_date:end_date]
total_discharge = np.zeros(len(observed_P)+10)
output = pd.DataFrame(index=range(len(observed_P)+len(df_7_9)))
deterministic_output = pd.DataFrame(index=range(len(observed_P)+len(df_7_9)))
observed_Q = observed_Q.loc[:end_date]

#
# Main
#

ensemble(df_7_9, deter_10, observed_P, R, S, x_1, x_2, x_3, x_4)

# UH1 = fun_UH1(x_4)
# UH2 = fun_UH2(x_4)
# for t in range(len(observed_P)):
#     P = observed_P.iat[t]
#     E = evap_lesse.iat[t]
#     [R, S, Q] = update_timestep(t, P, E, R, S, x_1, x_2, x_3, UH1, UH2)
#     total_discharge[t] = Q / 86400 * area * 1000

if plotprocess == True:
    max_values = output.max(axis=1)
    min_values = output.min(axis=1)
    diff_values = max_values-min_values
    print(diff_values.max(axis=0))
    fig, ax = plt.subplots()
    plt.plot(observed_Q.index, observed_Q["Q"], 'g')
    plt.plot(pd.date_range(start=start_date, end=forecast, freq='D'), deterministic_output, linewidth=0.6, color='r', alpha=1)
    plt.plot(pd.date_range(start=start_date, end=forecast, freq='D'), output, linewidth=0.2, color='b', alpha=0.4)
    plt.fill_between(pd.date_range(start=start_date, end=forecast, freq='D'), min_values, max_values, color='b', alpha=0.2, label='Flume')
    plt.vlines(datetime.date(2021, 7, 9), 0, 510, 'r', ':')
    plt.ylabel("Discharge (m^3/s)")
    plt.xlabel("Time (Days)")
    plt.legend(["Observed discharge", "Deterministic forecast", "Ensemble members"])
    plt.title("Forecast July 9")
    ax.grid(axis="y", which="both", alpha=0.2)
    ax.minorticks_on()
    ax.grid(axis="x", which="major", alpha=0.2)
    ax.set_axisbelow(True)
    plt.xlim([datetime.date(2021, 7, 4), datetime.date(2021, 7, 19)])
    plt.ylim([0, 510])
    plt.show()
else:
    np.savetxt("resultaten\\output.txt", output)
    np.savetxt("resultaten\\det_output.txt", deterministic_output)
    print("done")
