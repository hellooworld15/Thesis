
######################################################################################3
# SID: 470378351
import csv
import numpy as np
import os
import matplotlib.pyplot as plt
import random
import pandas as pd

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 15}

plt.rc('font', **font)

vel_dats = []
filename = ['10ms_6x4_tractor','10ms_noprop_pusher','10ms_6x4_pusher']
df_trac = pd.read_csv("./" + filename[0] + ".csv")
df_noProp = pd.read_csv("./" + filename[1] + ".csv")
df_pusher = pd.read_csv("./" + filename[2] + ".csv")
all_data = [df_trac, df_noProp, df_pusher]
vel_dats.append(all_data)

filename = ['15ms_6x4_tractor','15ms_noprop_pusher','15ms_6x4_pusher']
df_trac = pd.read_csv("./" + filename[0] + ".csv")
df_noProp = pd.read_csv("./" + filename[1] + ".csv")
df_pusher = pd.read_csv("./" + filename[2] + ".csv")
all_data = [df_trac, df_noProp, df_pusher]
vel_dats.append(all_data)

filename = ['20ms_6x4_tractor','20ms_noprop_pusher','20ms_6x4_pusher']
df_trac = pd.read_csv("./" + filename[0] + ".csv")
df_noProp = pd.read_csv("./" + filename[1] + ".csv")
df_pusher = pd.read_csv("./" + filename[2] + ".csv")
all_data = [df_trac, df_noProp, df_pusher]
vel_dats.append(all_data)

all_data = vel_dats[0]

# df_at_3000 = df.iloc[0::3, :]
# df_at_9000 = df.iloc[1::3, :]
# df_at_11000 = df.iloc[2::3, :]

# First Graph - Cl
def cl_calculation(data, rho, V):
    fx_dat = data[" fx"]
    fz_dat = data[" fz"]
    aoa_dat = data[" aoa"]
    L = fz_dat*np.cos(aoa_dat*np.pi/180) + fx_dat*np.sin(aoa_dat*np.pi/180)
    D = fx_dat*np.cos(aoa_dat*np.pi/180) + fz_dat*np.sin(aoa_dat*np.pi/180)
    c_L = L/(0.5*rho*(V**2)*(0.86*0.186))
    c_D = D/(0.5*rho*(V**2)*(0.86*0.186))
    return [c_L, c_D, aoa_dat]

def moment_coeffs(data, rho, V, S, c, b):
    Mx_dat = data[' tx']
    My_dat = data[' ty']
    Mz_dat = data[' tz']
    fx_dat = data[' fx']
    fy_dat = data[' fy']
    fz_dat = data[' fz']
    aoa_dat = data[' aoa']

    # Roll moment
    c_l = (Mx_dat + fy_dat*(0.083/2) + fz_dat * 0)/ (0.5*rho*(V**2)*S*b)
    # Pitch moment
    c_n = (Mz_dat + fy_dat * (0.00925) + fx_dat * 0) / (0.5*rho*(V**2)*S*b)
    # Pitch moment - negative sign for moment direction
    c_m = (My_dat - fx_dat * (0.083/2) - fz_dat * 0.00925)/(0.5*rho*(V**2)*S*c)
    # c_m = Mz_dat
    return [c_l, c_m, c_n, aoa]

def static_margin(data, rho, V, S, c, b):
    Mx_dat = data[' tx']
    My_dat = data[' ty']
    Mz_dat = data[' tz']
    fx_dat = data[' fx']
    fy_dat = data[' fy']
    fz_dat = data[' fz']
    aoa_dat = data[' aoa']
    L = fz_dat*np.cos(aoa_dat*np.pi/180) + fx_dat*np.sin(aoa_dat*np.pi/180)
    c_L = L/(0.5*rho*(V**2)*(0.86*0.186))
    c_m = (My_dat - fx_dat * (0.083/2) - fz_dat * 0.00925)/(0.5*rho*(V**2)*S*c)
    return [c_m, c_L]
    
#########################################################################
# # Figure 1 
RPMS = [0,1,2]
Vels = [ 10, 15, 20]
c = 0.186
b = 0.86
S = c * b
####################################################################################
##################################  Cm ####################################################3
all_data = vel_dats[0]

plt.figure(figsize=(10,8))
markers= ['D', 'o', 's']
colours = ['blue', 'red', 'yellow']
# line_colours = ['m','c', 'g']
for i in all_data:
    df = i
    [c_l, c_m, c_n, aoa] = moment_coeffs(df.iloc[RPMS[0]::3, :], 1.174, 10, S, c ,b)
    print(aoa)
    print(c_m)
    plt.plot(aoa, c_m, linewidth=3)

plt.xlabel("Angle of Attack")
plt.ylabel("Pitching Moment (Cm)")
plt.tick_params(direction="in")
plt.legend(["10m/s Tractor 6000RPM","10m/s No Prop 6000RPM","10m/s Pusher 6000RPM"])
plt.savefig("./tractor_vs_pusher/Cm/10ms_6000RPM_Cm.png",  dpi=600)

######################################################################################
plt.figure(figsize=(10,8))
markers= ['D', 'o', 's']
colours = ['blue', 'red', 'yellow']
# line_colours = ['m','c', 'g']
for i in all_data:
    df = i
    [c_l, c_m, c_n, aoa] = moment_coeffs(df.iloc[RPMS[2]::3, :], 1.174, 10, S, c ,b)
    plt.plot(aoa, c_m, linewidth=3)

plt.xlabel("Angle of Attack")
plt.ylabel("Pitching Moment (Cm)")
plt.tick_params(direction="in")
plt.legend(["10m/s Tractor 11000RPM","10m/s No Prop 11000RPM","10m/s Pusher 11000RPM"])
plt.savefig("./tractor_vs_pusher/Cm/10ms_11000RPM_Cm.png",  dpi=600)
##################################################################################3

plt.figure(figsize=(10,8))
markers= ['D', 'o', 's']
colours = ['blue', 'red', 'yellow']
# line_colours = ['m','c', 'g']
k = 0
for j in Vels:
    all_data = vel_dats[k]
    for i in all_data:
        df = i
        [c_l, c_m, c_n, aoa] = moment_coeffs(df.iloc[RPMS[0]::3, :], 1.174, j, S, c ,b)
        plt.plot(aoa, c_m, linewidth=3)
    k = k + 1

plt.xlabel("Angle of Attack")
plt.ylabel("Pitching Moment (Cm)")
plt.tick_params(direction="in")
plt.legend(["10m/s Tractor","10m/s No Prop","10m/s Pusher","15m/s Tractor", "15m/s No Prop", "15m/s Pusher", "20m/s Tractor","20m/s No Prop", "20m/s Pusher"])
plt.savefig("./tractor_vs_pusher/Cm/6000RPM_Cm.png",  dpi=600)

#####################################################################################

plt.figure(figsize=(10,8))
markers= ['D', 'o', 's']
colours = ['blue', 'red', 'yellow']
# line_colours = ['m','c', 'g']
k = 0
for j in Vels:
    all_data = vel_dats[k]
    for i in all_data:
        df = i
        [c_l, c_m, c_n, aoa] = moment_coeffs(df.iloc[RPMS[2]::3, :], 1.174, j, S, c ,b)
        plt.plot(aoa, c_m, linewidth=3)
    k = k + 1

plt.xlabel("Angle of Attack")
plt.ylabel("Pitching Moment (Cm)")
plt.tick_params(direction="in")
plt.legend(["10m/s Tractor","10m/s No Prop","10m/s Pusher","15m/s Tractor", "15m/s No Prop", "15m/s Pusher", "20m/s Tractor","20m/s No Prop", "20m/s Pusher"])
plt.savefig("./tractor_vs_pusher/Cm/11000RPM_Cm.png",  dpi=600)
#################################################j##################################
all_data = vel_dats[0]

plt.figure(figsize=(10,8))
markers= ['D', 'o', 's']
colours = ['blue', 'red', 'yellow']

for j in RPMS:
    for i in all_data:
        df = i
        [c_l, c_m, c_n, aoa] = moment_coeffs(df.iloc[RPMS[j]::3, :], 1.174, 10, S, c ,b)
        plt.plot(aoa, c_m, linewidth=3)

plt.xlabel("Angle of Attack")
plt.ylabel("Pitching Moment (Cm)")
plt.tick_params(direction="in")
plt.legend(["Tractor 6000RPM ","No Prop 6000RPM","Pusher 6000RPM", "Tractor 9000RPM", "No Prop 9000RPM", "Pusher 9000RPM", "Tractor 11000RPM", "No Prop 11000RPM", " Pusher 11000RPM"])
plt.savefig("./tractor_vs_pusher/Cm/10ms_Cm.png",  dpi=300)

######################################################################################3
all_data = vel_dats[2]

plt.figure(figsize=(10,8))
markers= ['D', 'o', 's']
colours = ['blue', 'red', 'yellow']
# line_colours = ['m','c', 'g']
for j in RPMS:
    for i in all_data:
        df = i
        [c_l, c_m, c_n, aoa] = moment_coeffs(df.iloc[RPMS[j]::3, :], 1.174, 20, S, c ,b)
        plt.plot(aoa, c_m, linewidth=3)

plt.xlabel("Angle of Attack")
plt.ylabel("Pitching Moment (Cm)")
plt.tick_params(direction="in")
plt.legend(["Tractor 6000RPM ","No Prop 6000RPM","Pusher 6000RPM", "Tractor 9000RPM", "No Prop 9000RPM", "Pusher 9000RPM", "Tractor 11000RPM", "No Prop 11000RPM", " Pusher 11000RPM"])
plt.savefig("./tractor_vs_pusher/Cm/20ms_Cm.png",  dpi=300)

######################################################################################3
#####################################   Cn #################################################3plt.figure(figsize=(10,8))
all_data = vel_dats[0]

plt.figure(figsize=(10,8))
markers= ['D', 'o', 's']
colours = ['blue', 'red', 'yellow']
# line_colours = ['m','c', 'g']
for i in all_data:
    df = i
    [c_l, c_m, c_n, aoa] = moment_coeffs(df.iloc[RPMS[0]::3, :], 1.174, 10, S, c ,b)
    plt.plot(aoa, c_n, linewidth=3)

plt.xlabel("Angle of Attack")
plt.ylabel("Yaw Moment (Cn)")
plt.tick_params(direction="in")
plt.legend(["10m/s Tractor 6000RPM","10m/s No Prop 6000RPM","10m/s Pusher 6000RPM"])
plt.savefig("./tractor_vs_pusher/Cn/10ms_6000RPM_Cn.png",  dpi=600)

######################################################################################

plt.figure(figsize=(10,8))
markers= ['D', 'o', 's']
colours = ['blue', 'red', 'yellow']
# line_colours = ['m','c', 'g']
for i in all_data:
    df = i
    [c_l, c_m, c_n, aoa] = moment_coeffs(df.iloc[RPMS[2]::3, :], 1.174, 10, S, c ,b)
    plt.plot(aoa, c_n, linewidth=3)

plt.xlabel("Angle of Attack")
plt.ylabel("Yaw Moment (Cn)")
plt.tick_params(direction="in")
plt.legend(["10m/s Tractor 11000RPM","10m/s No Prop 11000RPM","10m/s Pusher 11000RPM"])
plt.savefig("./tractor_vs_pusher/Cn/10ms_11000RPM_Cn.png",  dpi=600)
##################################################################################3

plt.figure(figsize=(10,8))
markers= ['D', 'o', 's']
colours = ['blue', 'red', 'yellow']
# line_colours = ['m','c', 'g']
k = 0
for j in Vels:
    all_data = vel_dats[k]
    for i in all_data:
        df = i
        [c_l, c_m, c_n, aoa] = moment_coeffs(df.iloc[RPMS[0]::3, :], 1.174, j, S, c ,b)
        plt.plot(aoa, c_n, linewidth=3)
    k = k + 1

plt.xlabel("Angle of Attack")
plt.ylabel("Yaw Moment (Cn)")
plt.tick_params(direction="in")
plt.legend(["10m/s Tractor","10m/s No Prop","10m/s Pusher","15m/s Tractor", "15m/s No Prop", "15m/s Pusher", "20m/s Tractor","20m/s No Prop", "20m/s Pusher"])
plt.savefig("./tractor_vs_pusher/Cn/6000RPM_Cn.png",  dpi=600)

#####################################################################################

plt.figure(figsize=(10,8))
markers= ['D', 'o', 's']
colours = ['blue', 'red', 'yellow']
# line_colours = ['m','c', 'g']
k = 0
for j in Vels:
    all_data = vel_dats[k]
    for i in all_data:
        df = i
        [c_l, c_m, c_n, aoa] = moment_coeffs(df.iloc[RPMS[2]::3, :], 1.174, j, S, c ,b)
        plt.plot(aoa, c_n, linewidth=3)
    k = k + 1

plt.xlabel("Angle of Attack")
plt.ylabel("Yaw Moment (Cn)")
plt.tick_params(direction="in")
plt.legend(["10m/s Tractor","10m/s No Prop","10m/s Pusher","15m/s Tractor", "15m/s No Prop", "15m/s Pusher", "20m/s Tractor","20m/s No Prop", "20m/s Pusher"])
plt.savefig("./tractor_vs_pusher/Cn/11000RPM_Cn.png",  dpi=600)
#################################################j##################################
all_data = vel_dats[0]

plt.figure(figsize=(10,8))
markers= ['D', 'o', 's']
colours = ['blue', 'red', 'yellow']

for j in RPMS:
    for i in all_data:
        df = i
        [c_l, c_m, c_n, aoa] = moment_coeffs(df.iloc[RPMS[j]::3, :], 1.174, 10, S, c ,b)
        plt.plot(aoa, c_n, linewidth=3)

plt.xlabel("Angle of Attack")
plt.ylabel("Yaw Moment (Cn)")
plt.tick_params(direction="in")
plt.legend(["Tractor 6000RPM ","No Prop 6000RPM","Pusher 6000RPM", "Tractor 9000RPM", "No Prop 9000RPM", "Pusher 9000RPM", "Tractor 11000RPM", "No Prop 11000RPM", " Pusher 11000RPM"])
plt.savefig("./tractor_vs_pusher/Cn/10ms_Cn.png",  dpi=300)

######################################################################################3
all_data = vel_dats[2]
plt.figure(figsize=(10,8))
markers= ['D', 'o', 's']
colours = ['blue', 'red', 'yellow']
# line_colours = ['m','c', 'g']
for j in RPMS:
    for i in all_data:
        df = i
        [c_l, c_m, c_n, aoa] = moment_coeffs(df.iloc[RPMS[j]::3, :], 1.174, 20, S, c ,b)
        plt.plot(aoa, c_n, linewidth=3)

plt.xlabel("Angle of Attack")
plt.ylabel("Yaw Moment (Cn)")
plt.tick_params(direction="in")
plt.legend(["Tractor 6000RPM ","No Prop 6000RPM","Pusher 6000RPM", "Tractor 9000RPM", "No Prop 9000RPM", "Pusher 9000RPM", "Tractor 11000RPM", "No Prop 11000RPM", " Pusher 11000RPM"])
plt.savefig("./tractor_vs_pusher/Cn/20ms_Cn.png",  dpi=300)

######################################################################################3
########################################## cl_roll ################################
all_data = vel_dats[0]

plt.figure(figsize=(10,8))
markers= ['D', 'o', 's']
colours = ['blue', 'red', 'yellow']
# line_colours = ['m','c', 'g']
for i in all_data:
    df = i
    [c_l, c_m, c_n, aoa] = moment_coeffs(df.iloc[RPMS[0]::3, :], 1.174, 10, S, c ,b)
    plt.plot(aoa, c_l, linewidth=3)

plt.xlabel("Angle of Attack")
plt.ylabel("Rolling Moment (Cl)")
plt.tick_params(direction="in")
plt.legend(["10m/s Tractor 6000RPM","10m/s No Prop 6000RPM","10m/s Pusher 6000RPM"])
plt.savefig("./tractor_vs_pusher/Cl_roll/10ms_6000RPM_Cl_roll.png",  dpi=600)

######################################################################################
plt.figure(figsize=(10,8))
markers= ['D', 'o', 's']
colours = ['blue', 'red', 'yellow']
# line_colours = ['m','c', 'g']
for i in all_data:
    df = i
    [c_l, c_m, c_n, aoa] = moment_coeffs(df.iloc[RPMS[2]::3, :], 1.174, 10, S, c ,b)
    plt.plot(aoa, c_l, linewidth=3)

plt.xlabel("Angle of Attack")
plt.ylabel("Rolling Moment (Cl)")
plt.tick_params(direction="in")
plt.legend(["10m/s Tractor 11000RPM","10m/s No Prop 11000RPM","10m/s Pusher 11000RPM"])
plt.savefig("./tractor_vs_pusher/Cl_roll/10ms_11000RPM_Cl.png",  dpi=600)
##################################################################################3

plt.figure(figsize=(10,8))
markers= ['D', 'o', 's']
colours = ['blue', 'red', 'yellow']
# line_colours = ['m','c', 'g']
k = 0
for j in Vels:
    all_data = vel_dats[k]
    for i in all_data:
        df = i
        [c_l, c_m, c_n, aoa] = moment_coeffs(df.iloc[RPMS[0]::3, :], 1.174, j, S, c ,b)
        plt.plot(aoa, c_l, linewidth=3)
    k = k + 1

plt.xlabel("Angle of Attack")
plt.ylabel("Rolling Moment (Cl)")
plt.tick_params(direction="in")
plt.legend(["10m/s Tractor","10m/s No Prop","10m/s Pusher","15m/s Tractor", "15m/s No Prop", "15m/s Pusher", "20m/s Tractor","20m/s No Prop", "20m/s Pusher"])
plt.savefig("./tractor_vs_pusher/Cl_roll/6000RPM_Cl.png",  dpi=600)

#####################################################################################

plt.figure(figsize=(10,8))
markers= ['D', 'o', 's']
colours = ['blue', 'red', 'yellow']
# line_colours = ['m','c', 'g']
k = 0
for j in Vels:
    all_data = vel_dats[k]
    for i in all_data:
        df = i
        [c_l, c_m, c_n, aoa] = moment_coeffs(df.iloc[RPMS[2]::3, :], 1.174, j, S, c ,b)
        plt.plot(aoa, c_l, linewidth=3)
    k = k + 1
plt.xlabel("Angle of Attack")
plt.ylabel("Rolling Moment (Cl)")
plt.tick_params(direction="in")
plt.legend(["10m/s Tractor","10m/s No Prop","10m/s Pusher","15m/s Tractor", "15m/s No Prop", "15m/s Pusher", "20m/s Tractor","20m/s No Prop", "20m/s Pusher"])
plt.savefig("./tractor_vs_pusher/Cl_roll/11000RPM_Cl_roll.png",  dpi=600)
#################################################j##################################
all_data = vel_dats[0]
plt.figure(figsize=(10,8))
markers= ['D', 'o', 's']
colours = ['blue', 'red', 'yellow']

for j in RPMS:
    for i in all_data:
        df = i
        [c_l, c_m, c_n, aoa] = moment_coeffs(df.iloc[RPMS[j]::3, :], 1.174, 10, S, c ,b)
        plt.plot(aoa, c_l, linewidth=3)

plt.xlabel("Angle of Attack")
plt.ylabel("Rolling Moment (Cl)")
plt.tick_params(direction="in")
plt.legend(["Tractor 6000RPM ","No Prop 6000RPM","Pusher 6000RPM", "Tractor 9000RPM", "No Prop 9000RPM", "Pusher 9000RPM", "Tractor 11000RPM", "No Prop 11000RPM", " Pusher 11000RPM"])
plt.savefig("./tractor_vs_pusher/Cl_roll/10ms_Cl_roll.png",  dpi=300)

######################################################################################3
all_data = vel_dats[2]

plt.figure(figsize=(10,8))
markers= ['D', 'o', 's']
colours = ['blue', 'red', 'yellow']
# line_colours = ['m','c', 'g']
for j in RPMS:
    for i in all_data:
        df = i
        [c_l, c_m, c_n, aoa] = moment_coeffs(df.iloc[RPMS[j]::3, :], 1.174, 20, S, c ,b)
        plt.plot(aoa, c_l, linewidth=3)

plt.xlabel("Angle of Attack")
plt.ylabel("Rolling Moment (Cl)")
plt.tick_params(direction="in")
plt.legend(["Tractor 6000RPM ","No Prop 6000RPM","Pusher 6000RPM", "Tractor 9000RPM", "No Prop 9000RPM", "Pusher 9000RPM", "Tractor 11000RPM", "No Prop 11000RPM", " Pusher 11000RPM"])
plt.savefig("./tractor_vs_pusher/Cl_roll/20ms_Cl_roll.png",  dpi=300)