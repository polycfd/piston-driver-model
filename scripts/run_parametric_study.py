import numpy as np
from src import parameters as params
from src import mass_spring_model as ms
from scipy.integrate import solve_ivp

# =========================================================================================================
# MASS-SPRING MODEL SIMULATIONS RUN FOR WATER AND LITHIUM, WITH 8 DIFFERENT DRIVING PRESSURES (1, 2, 3, 4, 5, 6, 7.5 and 10MPa) 
# AND 7 DIFFERENT MASS RATIOS (0, 0.1, 0.5, 0.9, 1, 1.5 and 2)
# =========================================================================================================


#Driving pressure: 1MPa#
p_p0 = 10e5
T_10 = 0.08 # Final time (s)

#water#
#M=0.1#
x_fluid_eq_10bar_water_01= ms.equilibrate_fluid_positions(params.n,params.L0,params.k_s_water,params.m_e_water,params.m_piston_01,params.g,params.p_t0,p_p0)
for i in range(len(x_fluid_eq_10bar_water_01)):
    x_fluid_eq_10bar_water_01[i]=x_fluid_eq_10bar_water_01[i]+params.x_air_bottom
x_p0_10bar_water_01=x_fluid_eq_10bar_water_01[0]
offset= x_p0_10bar_water_01 - (params.x_air_bottom+params.th_piston_01)
for i in range(len(x_fluid_eq_10bar_water_01)):
    x_fluid_eq_10bar_water_01[i]=x_fluid_eq_10bar_water_01[i]-offset
x_p0_10bar_water_01=x_fluid_eq_10bar_water_01[0]

dx_eq_10bar_water_01 = np.zeros(params.n)
dx_eq_10bar_water_01[0]=x_fluid_eq_10bar_water_01[1]-x_p0_10bar_water_01
for i in range (1,params.n):
    dx_eq_10bar_water_01[i]=x_fluid_eq_10bar_water_01[i]-x_fluid_eq_10bar_water_01[i-1]

V_p0_10bar_01 = params.A_p * (x_p0_10bar_water_01-params.th_piston_01) #lower gas initial volume
V_t0_10bar_01 = params.A_p * (params.H_01-x_fluid_eq_10bar_water_01[params.n])  #upper gas initial volume

v_p0 = 0 #initial piston velocity
v_fluid0=np.zeros(params.n)
y0 = np.concatenate((x_fluid_eq_10bar_water_01, [v_p0], v_fluid0))

p_0_x_10bar_water_01= np.zeros(params.n)
for i in range (params.n):
    p_0_x_10bar_water_01[i] = params.p_t0 + params.rho_water*params.g*(x_fluid_eq_10bar_water_01[params.n]-x_fluid_eq_10bar_water_01[i])

#Solve#
sol_10bar_water_01 = solve_ivp(ms.system_water, [0, T_10], y0, args=(params.m_piston_01,params.H_01, V_t0_10bar_01, V_p0_10bar_01, params.k_s_water, params.m_e_water, params.n, p_p0), atol = 1e-6, rtol=1e-6, max_step = 1e-6)
sol_10bar_water_01.t=np.array(sol_10bar_water_01.t)
#Solutions#
x_p_10bar_water_01 = sol_10bar_water_01.y[0]      # Piston position
x_fluid_10bar_water_01 = sol_10bar_water_01.y[1:params.n+1] # Liquid elements position
v_fluid_10bar_water_01 = sol_10bar_water_01.y[params.n+1:2*(params.n+1)] #Liquid elements velocity
a0_values_10bar_water_01 = [ms.piston_acceleration_water(y,params.m_piston_01, params.H_01, V_p0_10bar_01, params.n, params.k_s_water, p_p0) for y in sol_10bar_water_01.y.T]
a0_min_10bar_water_01 = np.min(a0_values_10bar_water_01)
a0_max_10bar_water_01 = np.max(a0_values_10bar_water_01)
v_piston_10bar_water_01 = sol_10bar_water_01.y[params.n+1, :]
v_piston_max_10bar_water_01=np.max(v_piston_10bar_water_01)
all_acc_10bar_water_01 = np.array([
    ms.all_accelerations_water(y, params.m_piston_01, params.H_01, V_t0_10bar_01, V_p0_10bar_01, params.k_s_water, params.m_e_water, params.n, p_p0)
    for y in sol_10bar_water_01.y.T
])

a_piston_10bar_water_01 = all_acc_10bar_water_01[:, 0] 
a_fluid_10bar_water_01  = all_acc_10bar_water_01[:, 1:].T 
V_p_10bar_water_01 = params.A_p * (x_p_10bar_water_01 - params.th_piston_01)
V_t_10bar_water_01 = params.A_p * (params.H_01 - x_fluid_10bar_water_01[params.n-1, :])

#Variation of the upper gas volume pressure during the simulation#
p_t_10bar_water_01=[]
for i in range(len(sol_10bar_water_01.t)):
    p_ti=ms.pressure_on_top(sol_10bar_water_01.y[:,i], params.H_01, params.th_piston_01, V_t0_10bar_01, params.n)
    p_t_10bar_water_01.append(p_ti)
p_t_10bar_water_01 = np.array(p_t_10bar_water_01)
p_t_10bar_water_max_01 = np.max(p_t_10bar_water_01)
#Variation of the lower gas volume pressure during the simulation#
p_p_10bar_water_01=[]
for i in range(len(sol_10bar_water_01.t)):
    p_pi=ms.pressure_under_piston(sol_10bar_water_01.y[:,i], params.H_01, params.th_piston_01, V_p0_10bar_01, params.n,p_p0)
    p_p_10bar_water_01.append(p_pi)
p_p_10bar_water_01 = np.array(p_p_10bar_water_01)
#Pressure in the liquid elements#
dx_10bar_water_01 = np.zeros((params.n, len(sol_10bar_water_01.t)))
for j in range(len(sol_10bar_water_01.t)):
    dx_10bar_water_01[0,j]=x_fluid_10bar_water_01[0,j]-x_p_10bar_water_01[j]
    for i in range (1,params.n):
        dx_10bar_water_01[i,j]=x_fluid_10bar_water_01[i,j]-x_fluid_10bar_water_01[i-1,j]
p_10bar_water_01= np.ones((params.n,len(sol_10bar_water_01.t)))*params.p_0
for i in range(params.n):
    for j in range(len(sol_10bar_water_01.t)):
        p_10bar_water_01[i,j]=params.p_0-params.rho_l_water*params.c_water**2*((dx_10bar_water_01[i,j]*params.A_p)-dx_eq_10bar_water_01[i]*params.A_p) /(dx_eq_10bar_water_01[i]*params.A_p)

#M=0.5#
x_fluid_eq_10bar_water_05= ms.equilibrate_fluid_positions(params.n,params.L0,params.k_s_water,params.m_e_water,params.m_piston_05,params.g,params.p_t0,p_p0)
for i in range(len(x_fluid_eq_10bar_water_05)):
    x_fluid_eq_10bar_water_05[i]=x_fluid_eq_10bar_water_05[i]+params.x_air_bottom
x_p0_10bar_water_05=x_fluid_eq_10bar_water_05[0]
offset= x_p0_10bar_water_05 - (params.x_air_bottom+params.th_piston_05)
for i in range(len(x_fluid_eq_10bar_water_05)):
    x_fluid_eq_10bar_water_05[i]=x_fluid_eq_10bar_water_05[i]-offset
x_p0_10bar_water_05=x_fluid_eq_10bar_water_05[0]

dx_eq_10bar_water_05 = np.zeros(params.n)
dx_eq_10bar_water_05[0]=x_fluid_eq_10bar_water_05[1]-x_p0_10bar_water_05
for i in range (1,params.n):
    dx_eq_10bar_water_05[i]=x_fluid_eq_10bar_water_05[i]-x_fluid_eq_10bar_water_05[i-1]

V_p0_10bar_05 = params.A_p * (x_p0_10bar_water_05-params.th_piston_05)
V_t0_10bar_05 = params.A_p * (params.H_05-x_fluid_eq_10bar_water_05[params.n]) 

v_p0 = 0 
v_fluid0=np.zeros(params.n)
y0 = np.concatenate((x_fluid_eq_10bar_water_05, [v_p0], v_fluid0))

p_0_x_10bar_water_05= np.zeros(params.n)
for i in range (params.n):
    p_0_x_10bar_water_05[i] = params.p_t0 + params.rho_water*params.g*(x_fluid_eq_10bar_water_05[params.n]-x_fluid_eq_10bar_water_05[i])

sol_10bar_water_05 = solve_ivp(ms.system_water, [0, T_10], y0, args=(params.m_piston_05,params.H_05, V_t0_10bar_05, V_p0_10bar_05, params.k_s_water, params.m_e_water, params.n, p_p0), atol = 1e-6, rtol=1e-6, max_step = 1e-6)
sol_10bar_water_05.t=np.array(sol_10bar_water_05.t)

x_p_10bar_water_05 = sol_10bar_water_05.y[0]     
x_fluid_10bar_water_05 = sol_10bar_water_05.y[1:params.n+1]
v_fluid_10bar_water_05 = sol_10bar_water_05.y[params.n+1:2*(params.n+1)]
a0_values_10bar_water_05 = [ms.piston_acceleration_water(y,params.m_piston_05, params.H_05, V_p0_10bar_05, params.n, params.k_s_water, p_p0) for y in sol_10bar_water_05.y.T]
a0_min_10bar_water_05 = np.min(a0_values_10bar_water_05)
a0_max_10bar_water_05 = np.max(a0_values_10bar_water_05)
v_piston_10bar_water_05 = sol_10bar_water_05.y[params.n+1, :]
v_piston_max_10bar_water_05=np.max(v_piston_10bar_water_05)

all_acc_10bar_water_05 = np.array([
    ms.all_accelerations_water(y, params.m_piston_05, params.H_05, V_t0_10bar_05, V_p0_10bar_05, params.k_s_water, params.m_e_water, params.n, p_p0)
    for y in sol_10bar_water_05.y.T
])

a_piston_10bar_water_05 = all_acc_10bar_water_05[:, 0]     
a_fluid_10bar_water_05  = all_acc_10bar_water_05[:, 1:].T  
V_p_10bar_water_05 = params.A_p * (x_p_10bar_water_05 - params.th_piston_05)
V_t_10bar_water_05 = params.A_p * (params.H_05 - x_fluid_10bar_water_05[params.n-1, :])

p_t_10bar_water_05=[]
for i in range(len(sol_10bar_water_05.t)):
    p_ti=ms.pressure_on_top(sol_10bar_water_05.y[:,i], params.H_05, params.th_piston_05, V_t0_10bar_05, params.n)
    p_t_10bar_water_05.append(p_ti)
p_t_10bar_water_05 = np.array(p_t_10bar_water_05)
p_t_10bar_water_max_05 = np.max(p_t_10bar_water_05)

p_p_10bar_water_05=[]
for i in range(len(sol_10bar_water_05.t)):
    p_pi=ms.pressure_under_piston(sol_10bar_water_05.y[:,i], params.H_05, params.th_piston_05, V_p0_10bar_05, params.n,p_p0)
    p_p_10bar_water_05.append(p_pi)
p_p_10bar_water_05 = np.array(p_p_10bar_water_05)

dx_10bar_water_05 = np.zeros((params.n, len(sol_10bar_water_05.t)))
for j in range(len(sol_10bar_water_05.t)):
    dx_10bar_water_05[0,j]=x_fluid_10bar_water_05[0,j]-x_p_10bar_water_05[j]
    for i in range (1,params.n):
        dx_10bar_water_05[i,j]=x_fluid_10bar_water_05[i,j]-x_fluid_10bar_water_05[i-1,j]
p_10bar_water_05= np.ones((params.n,len(sol_10bar_water_05.t)))*params.p_0
for i in range(params.n):
    for j in range(len(sol_10bar_water_05.t)):
        p_10bar_water_05[i,j]=params.p_0-params.rho_l_water*params.c_water**2*((dx_10bar_water_05[i,j]*params.A_p)-dx_eq_10bar_water_05[i]*params.A_p) /(dx_eq_10bar_water_05[i]*params.A_p)

#M=0.9#
x_fluid_eq_10bar_water_09= ms.equilibrate_fluid_positions(params.n,params.L0,params.k_s_water,params.m_e_water,params.m_piston_09,params.g,params.p_t0,p_p0)
for i in range(len(x_fluid_eq_10bar_water_09)):
    x_fluid_eq_10bar_water_09[i]=x_fluid_eq_10bar_water_09[i]+params.x_air_bottom
x_p0_10bar_water_09=x_fluid_eq_10bar_water_09[0]
offset= x_p0_10bar_water_09 - (params.x_air_bottom+params.th_piston_09)
for i in range(len(x_fluid_eq_10bar_water_09)):
    x_fluid_eq_10bar_water_09[i]=x_fluid_eq_10bar_water_09[i]-offset
x_p0_10bar_water_09=x_fluid_eq_10bar_water_09[0]

dx_eq_10bar_water_09 = np.zeros(params.n)
dx_eq_10bar_water_09[0]=x_fluid_eq_10bar_water_09[1]-x_p0_10bar_water_09
for i in range (1,params.n):
    dx_eq_10bar_water_09[i]=x_fluid_eq_10bar_water_09[i]-x_fluid_eq_10bar_water_09[i-1]

V_p0_10bar_09 = params.A_p * (x_p0_10bar_water_09-params.th_piston_09) 
V_t0_10bar_09 = params.A_p * (params.H_09-x_fluid_eq_10bar_water_09[params.n])

v_p0 = 0
v_fluid0=np.zeros(params.n)
y0 = np.concatenate((x_fluid_eq_10bar_water_09, [v_p0], v_fluid0))

p_0_x_10bar_water_09= np.zeros(params.n)
for i in range (params.n):
    p_0_x_10bar_water_09[i] = params.p_t0 + params.rho_water*params.g*(x_fluid_eq_10bar_water_09[params.n]-x_fluid_eq_10bar_water_09[i])


sol_10bar_water_09 = solve_ivp(ms.system_water, [0, T_10], y0, args=(params.m_piston_09,params.H_09, V_t0_10bar_09, V_p0_10bar_09, params.k_s_water, params.m_e_water, params.n, p_p0), atol = 1e-6, rtol=1e-6, max_step = 1e-6)
sol_10bar_water_09.t=np.array(sol_10bar_water_09.t)

x_p_10bar_water_09 = sol_10bar_water_09.y[0]      
x_fluid_10bar_water_09 = sol_10bar_water_09.y[1:params.n+1] 
v_fluid_10bar_water_09 = sol_10bar_water_09.y[params.n+1:2*(params.n+1)] 
a0_values_10bar_water_09 = [ms.piston_acceleration_water(y,params.m_piston_09, params.H_09, V_p0_10bar_09, params.n, params.k_s_water, p_p0) for y in sol_10bar_water_09.y.T]
a0_min_10bar_water_09 = np.min(a0_values_10bar_water_09)
a0_max_10bar_water_09 = np.max(a0_values_10bar_water_09)
v_piston_10bar_water_09 = sol_10bar_water_09.y[params.n+1, :]
v_piston_max_10bar_water_09=np.max(v_piston_10bar_water_09)
all_acc_10bar_water_09 = np.array([
    ms.all_accelerations_water(y, params.m_piston_09, params.H_09, V_t0_10bar_09, V_p0_10bar_09, params.k_s_water, params.m_e_water, params.n, p_p0)
    for y in sol_10bar_water_09.y.T
])

a_piston_10bar_water_09 = all_acc_10bar_water_09[:, 0]      
a_fluid_10bar_water_09  = all_acc_10bar_water_09[:, 1:].T   
V_p_10bar_water_09 = params.A_p * (x_p_10bar_water_09 - params.th_piston_09)
V_t_10bar_water_09 = params.A_p * (params.H_09 - x_fluid_10bar_water_09[params.n-1, :])

p_t_10bar_water_09=[]
for i in range(len(sol_10bar_water_09.t)):
    p_ti=ms.pressure_on_top(sol_10bar_water_09.y[:,i], params.H_09, params.th_piston_09, V_t0_10bar_09, params.n)
    p_t_10bar_water_09.append(p_ti)
p_t_10bar_water_09 = np.array(p_t_10bar_water_09)
p_t_10bar_water_max_09 = np.max(p_t_10bar_water_09)

p_p_10bar_water_09=[]
for i in range(len(sol_10bar_water_09.t)):
    p_pi=ms.pressure_under_piston(sol_10bar_water_09.y[:,i], params.H_09, params.th_piston_09, V_p0_10bar_09, params.n,p_p0)
    p_p_10bar_water_09.append(p_pi)
p_p_10bar_water_09 = np.array(p_p_10bar_water_09)

dx_10bar_water_09 = np.zeros((params.n, len(sol_10bar_water_09.t)))
for j in range(len(sol_10bar_water_09.t)):
    dx_10bar_water_09[0,j]=x_fluid_10bar_water_09[0,j]-x_p_10bar_water_09[j]
    for i in range (1,params.n):
        dx_10bar_water_09[i,j]=x_fluid_10bar_water_09[i,j]-x_fluid_10bar_water_09[i-1,j]
p_10bar_water_09= np.ones((params.n,len(sol_10bar_water_09.t)))*params.p_0
for i in range(params.n):
    for j in range(len(sol_10bar_water_09.t)):
        p_10bar_water_09[i,j]=params.p_0-params.rho_l_water*params.c_water**2*((dx_10bar_water_09[i,j]*params.A_p)-dx_eq_10bar_water_09[i]*params.A_p) /(dx_eq_10bar_water_09[i]*params.A_p)

#M=1#
x_fluid_eq_10bar_water_1= ms.equilibrate_fluid_positions(params.n,params.L0,params.k_s_water,params.m_e_water,params.m_piston_1,params.g,params.p_t0,p_p0)
for i in range(len(x_fluid_eq_10bar_water_1)):
    x_fluid_eq_10bar_water_1[i]=x_fluid_eq_10bar_water_1[i]+params.x_air_bottom
x_p0_10bar_water_1=x_fluid_eq_10bar_water_1[0]
offset= x_p0_10bar_water_1 - (params.x_air_bottom+params.th_piston_1)
for i in range(len(x_fluid_eq_10bar_water_1)):
    x_fluid_eq_10bar_water_1[i]=x_fluid_eq_10bar_water_1[i]-offset
x_p0_10bar_water_1=x_fluid_eq_10bar_water_1[0]

dx_eq_10bar_water_1 = np.zeros(params.n)
dx_eq_10bar_water_1[0]=x_fluid_eq_10bar_water_1[1]-x_p0_10bar_water_1
for i in range (1,params.n):
    dx_eq_10bar_water_1[i]=x_fluid_eq_10bar_water_1[i]-x_fluid_eq_10bar_water_1[i-1]

V_p0_10bar_1 = params.A_p * (x_p0_10bar_water_1-params.th_piston_1)
V_t0_10bar_1 = params.A_p * (params.H_1-x_fluid_eq_10bar_water_1[params.n])

v_p0 = 0
v_fluid0=np.zeros(params.n)
y0 = np.concatenate((x_fluid_eq_10bar_water_1, [v_p0], v_fluid0))

p_0_x_10bar_water_1= np.zeros(params.n)
for i in range (params.n):
    p_0_x_10bar_water_1[i] = params.p_t0 + params.rho_water*params.g*(x_fluid_eq_10bar_water_1[params.n]-x_fluid_eq_10bar_water_1[i])


sol_10bar_water_1 = solve_ivp(ms.system_water, [0, T_10], y0, args=(params.m_piston_1,params.H_1, V_t0_10bar_1, V_p0_10bar_1, params.k_s_water, params.m_e_water, params.n, p_p0), atol = 1e-6, rtol=1e-6, max_step = 1e-6)
sol_10bar_water_1.t=np.array(sol_10bar_water_1.t)

x_p_10bar_water_1 = sol_10bar_water_1.y[0]     
x_fluid_10bar_water_1 = sol_10bar_water_1.y[1:params.n+1] 
v_fluid_10bar_water_1 = sol_10bar_water_1.y[params.n+1:2*(params.n+1)]
a0_values_10bar_water_1 = [ms.piston_acceleration_water(y,params.m_piston_1, params.H_1, V_p0_10bar_1, params.n, params.k_s_water, p_p0) for y in sol_10bar_water_1.y.T]
a0_min_10bar_water_1 = np.min(a0_values_10bar_water_1)
a0_max_10bar_water_1 = np.max(a0_values_10bar_water_1)
v_piston_10bar_water_1 = sol_10bar_water_1.y[params.n+1, :]
v_piston_max_10bar_water_1=np.max(v_piston_10bar_water_1)
all_acc_10bar_water_1 = np.array([
    ms.all_accelerations_water(y, params.m_piston_1, params.H_1, V_t0_10bar_1, V_p0_10bar_1, params.k_s_water, params.m_e_water, params.n, p_p0)
    for y in sol_10bar_water_1.y.T
])

a_piston_10bar_water_1 = all_acc_10bar_water_1[:, 0]      
a_fluid_10bar_water_1  = all_acc_10bar_water_1[:, 1:].T 
V_p_10bar_water_1 = params.A_p * (x_p_10bar_water_1 - params.th_piston_1)
V_t_10bar_water_1 = params.A_p * (params.H_1 - x_fluid_10bar_water_1[params.n-1, :])

p_t_10bar_water_1=[]
for i in range(len(sol_10bar_water_1.t)):
    p_ti=ms.pressure_on_top(sol_10bar_water_1.y[:,i], params.H_1, params.th_piston_1, V_t0_10bar_1, params.n)
    p_t_10bar_water_1.append(p_ti)
p_t_10bar_water_1 = np.array(p_t_10bar_water_1)
p_t_10bar_water_max_1 = np.max(p_t_10bar_water_1)

p_p_10bar_water_1=[]
for i in range(len(sol_10bar_water_1.t)):
    p_pi=ms.pressure_under_piston(sol_10bar_water_1.y[:,i], params.H_1, params.th_piston_1, V_p0_10bar_1, params.n,p_p0)
    p_p_10bar_water_1.append(p_pi)
p_p_10bar_water_1 = np.array(p_p_10bar_water_1)

dx_10bar_water_1 = np.zeros((params.n, len(sol_10bar_water_1.t)))
for j in range(len(sol_10bar_water_1.t)):
    dx_10bar_water_1[0,j]=x_fluid_10bar_water_1[0,j]-x_p_10bar_water_1[j]
    for i in range (1,params.n):
        dx_10bar_water_1[i,j]=x_fluid_10bar_water_1[i,j]-x_fluid_10bar_water_1[i-1,j]
p_10bar_water_1= np.ones((params.n,len(sol_10bar_water_1.t)))*params.p_0
for i in range(params.n):
    for j in range(len(sol_10bar_water_1.t)):
        p_10bar_water_1[i,j]=params.p_0-params.rho_l_water*params.c_water**2*((dx_10bar_water_1[i,j]*params.A_p)-dx_eq_10bar_water_1[i]*params.A_p) /(dx_eq_10bar_water_1[i]*params.A_p)

#M=1.5#
x_fluid_eq_10bar_water_15= ms.equilibrate_fluid_positions(params.n,params.L0,params.k_s_water,params.m_e_water,params.m_piston_15,params.g,params.p_t0,p_p0)
for i in range(len(x_fluid_eq_10bar_water_15)):
    x_fluid_eq_10bar_water_15[i]=x_fluid_eq_10bar_water_15[i]+params.x_air_bottom
x_p0_10bar_water_15=x_fluid_eq_10bar_water_15[0]
offset= x_p0_10bar_water_15 - (params.x_air_bottom+params.th_piston_15)
for i in range(len(x_fluid_eq_10bar_water_15)):
    x_fluid_eq_10bar_water_15[i]=x_fluid_eq_10bar_water_15[i]-offset
x_p0_10bar_water_15=x_fluid_eq_10bar_water_15[0]

dx_eq_10bar_water_15 = np.zeros(params.n)
dx_eq_10bar_water_15[0]=x_fluid_eq_10bar_water_15[1]-x_p0_10bar_water_15
for i in range (1,params.n):
    dx_eq_10bar_water_15[i]=x_fluid_eq_10bar_water_15[i]-x_fluid_eq_10bar_water_15[i-1]

V_p0_10bar_15 = params.A_p * (x_p0_10bar_water_15-params.th_piston_15) 
V_t0_10bar_15 = params.A_p * (params.H_15-x_fluid_eq_10bar_water_15[params.n])  

v_p0 = 0 
v_fluid0=np.zeros(params.n)
y0 = np.concatenate((x_fluid_eq_10bar_water_15, [v_p0], v_fluid0))

p_0_x_10bar_water_15= np.zeros(params.n)
for i in range (params.n):
    p_0_x_10bar_water_15[i] = params.p_t0 + params.rho_water*params.g*(x_fluid_eq_10bar_water_15[params.n]-x_fluid_eq_10bar_water_15[i])


sol_10bar_water_15 = solve_ivp(ms.system_water, [0, T_10], y0, args=(params.m_piston_15,params.H_15, V_t0_10bar_15, V_p0_10bar_15, params.k_s_water, params.m_e_water, params.n, p_p0), atol = 1e-6, rtol=1e-6, max_step = 1e-6)
sol_10bar_water_15.t=np.array(sol_10bar_water_15.t)

x_p_10bar_water_15 = sol_10bar_water_15.y[0]      
x_fluid_10bar_water_15 = sol_10bar_water_15.y[1:params.n+1] 
v_fluid_10bar_water_15 = sol_10bar_water_15.y[params.n+1:2*(params.n+1)] 
a0_values_10bar_water_15 = [ms.piston_acceleration_water(y,params.m_piston_15, params.H_15, V_p0_10bar_15, params.n, params.k_s_water, p_p0) for y in sol_10bar_water_15.y.T]
a0_min_10bar_water_15 = np.min(a0_values_10bar_water_15)
a0_max_10bar_water_15 = np.max(a0_values_10bar_water_15)
v_piston_10bar_water_15 = sol_10bar_water_15.y[params.n+1, :]
v_piston_max_10bar_water_15=np.max(v_piston_10bar_water_15)

all_acc_10bar_water_15 = np.array([
    ms.all_accelerations_water(y, params.m_piston_15, params.H_15, V_t0_10bar_15, V_p0_10bar_15, params.k_s_water, params.m_e_water, params.n, p_p0)
    for y in sol_10bar_water_15.y.T
])

a_piston_10bar_water_15 = all_acc_10bar_water_15[:, 0]      
a_fluid_10bar_water_15  = all_acc_10bar_water_15[:, 1:].T   
V_p_10bar_water_15 = params.A_p * (x_p_10bar_water_15 - params.th_piston_15)
V_t_10bar_water_15 = params.A_p * (params.H_15 - x_fluid_10bar_water_15[params.n-1, :])

p_t_10bar_water_15=[]
for i in range(len(sol_10bar_water_15.t)):
    p_ti=ms.pressure_on_top(sol_10bar_water_15.y[:,i], params.H_15, params.th_piston_15, V_t0_10bar_15, params.n)
    p_t_10bar_water_15.append(p_ti)
p_t_10bar_water_15 = np.array(p_t_10bar_water_15)
p_t_10bar_water_max_15 = np.max(p_t_10bar_water_15)

p_p_10bar_water_15=[]
for i in range(len(sol_10bar_water_15.t)):
    p_pi=ms.pressure_under_piston(sol_10bar_water_15.y[:,i], params.H_15, params.th_piston_15, V_p0_10bar_15, params.n,p_p0)
    p_p_10bar_water_15.append(p_pi)
p_p_10bar_water_15 = np.array(p_p_10bar_water_15)

dx_10bar_water_15 = np.zeros((params.n, len(sol_10bar_water_15.t)))
for j in range(len(sol_10bar_water_15.t)):
    dx_10bar_water_15[0,j]=x_fluid_10bar_water_15[0,j]-x_p_10bar_water_15[j]
    for i in range (1,params.n):
        dx_10bar_water_15[i,j]=x_fluid_10bar_water_15[i,j]-x_fluid_10bar_water_15[i-1,j]
p_10bar_water_15= np.ones((params.n,len(sol_10bar_water_15.t)))*params.p_0
for i in range(params.n):
    for j in range(len(sol_10bar_water_15.t)):
        p_10bar_water_15[i,j]=params.p_0-params.rho_l_water*params.c_water**2*((dx_10bar_water_15[i,j]*params.A_p)-dx_eq_10bar_water_15[i]*params.A_p) /(dx_eq_10bar_water_15[i]*params.A_p)

#M=2#
x_fluid_eq_10bar_water_2= ms.equilibrate_fluid_positions(params.n,params.L0,params.k_s_water,params.m_e_water,params.m_piston_2,params.g,params.p_t0,p_p0)
for i in range(len(x_fluid_eq_10bar_water_2)):
    x_fluid_eq_10bar_water_2[i]=x_fluid_eq_10bar_water_2[i]+params.x_air_bottom
x_p0_10bar_water_2=x_fluid_eq_10bar_water_2[0]
offset= x_p0_10bar_water_2 - (params.x_air_bottom+params.th_piston_2)
for i in range(len(x_fluid_eq_10bar_water_2)):
    x_fluid_eq_10bar_water_2[i]=x_fluid_eq_10bar_water_2[i]-offset
x_p0_10bar_water_2=x_fluid_eq_10bar_water_2[0]

dx_eq_10bar_water_2 = np.zeros(params.n)
dx_eq_10bar_water_2[0]=x_fluid_eq_10bar_water_2[1]-x_p0_10bar_water_2
for i in range (1,params.n):
    dx_eq_10bar_water_2[i]=x_fluid_eq_10bar_water_2[i]-x_fluid_eq_10bar_water_2[i-1]

V_p0_10bar_2 = params.A_p * (x_p0_10bar_water_2-params.th_piston_2) 
V_t0_10bar_2 = params.A_p * (params.H_2-x_fluid_eq_10bar_water_2[params.n]) 

v_p0 = 0 
v_fluid0=np.zeros(params.n)
y0 = np.concatenate((x_fluid_eq_10bar_water_2, [v_p0], v_fluid0))

p_0_x_10bar_water_2= np.zeros(params.n)
for i in range (params.n):
    p_0_x_10bar_water_2[i] = params.p_t0 + params.rho_water*params.g*(x_fluid_eq_10bar_water_2[params.n]-x_fluid_eq_10bar_water_2[i])


sol_10bar_water_2 = solve_ivp(ms.system_water, [0, T_10], y0, args=(params.m_piston_2,params.H_2, V_t0_10bar_2, V_p0_10bar_2, params.k_s_water, params.m_e_water, params.n, p_p0), atol = 1e-6, rtol=1e-6, max_step = 1e-6)
sol_10bar_water_2.t=np.array(sol_10bar_water_2.t)

x_p_10bar_water_2 = sol_10bar_water_2.y[0]      
x_fluid_10bar_water_2 = sol_10bar_water_2.y[1:params.n+1] 
v_fluid_10bar_water_2 = sol_10bar_water_2.y[params.n+1:2*(params.n+1)] 
a0_values_10bar_water_2 = [ms.piston_acceleration_water(y,params.m_piston_2, params.H_2, V_p0_10bar_2, params.n, params.k_s_water, p_p0) for y in sol_10bar_water_2.y.T]
a0_min_10bar_water_2 = np.min(a0_values_10bar_water_2)
a0_max_10bar_water_2 = np.max(a0_values_10bar_water_2)
v_piston_10bar_water_2 = sol_10bar_water_2.y[params.n+1, :]
v_piston_max_10bar_water_2=np.max(v_piston_10bar_water_2)

all_acc_10bar_water_2 = np.array([
    ms.all_accelerations_water(y, params.m_piston_2, params.H_2, V_t0_10bar_2, V_p0_10bar_2, params.k_s_water, params.m_e_water, params.n, p_p0)
    for y in sol_10bar_water_2.y.T
])

a_piston_10bar_water_2 = all_acc_10bar_water_2[:, 0]      
a_fluid_10bar_water_2  = all_acc_10bar_water_2[:, 1:].T  
V_p_10bar_water_2 = params.A_p * (x_p_10bar_water_2 - params.th_piston_2)
V_t_10bar_water_2 = params.A_p * (params.H_2 - x_fluid_10bar_water_2[params.n-1, :])

p_t_10bar_water_2=[]
for i in range(len(sol_10bar_water_2.t)):
    p_ti=ms.pressure_on_top(sol_10bar_water_2.y[:,i], params.H_2, params.th_piston_2, V_t0_10bar_2, params.n)
    p_t_10bar_water_2.append(p_ti)
p_t_10bar_water_2 = np.array(p_t_10bar_water_2)
p_t_10bar_water_max_2 = np.max(p_t_10bar_water_2)

p_p_10bar_water_2=[]
for i in range(len(sol_10bar_water_2.t)):
    p_pi=ms.pressure_under_piston(sol_10bar_water_2.y[:,i], params.H_2, params.th_piston_2, V_p0_10bar_2, params.n,p_p0)
    p_p_10bar_water_2.append(p_pi)
p_p_10bar_water_2 = np.array(p_p_10bar_water_2)

dx_10bar_water_2 = np.zeros((params.n, len(sol_10bar_water_2.t)))
for j in range(len(sol_10bar_water_2.t)):
    dx_10bar_water_2[0,j]=x_fluid_10bar_water_2[0,j]-x_p_10bar_water_2[j]
    for i in range (1,params.n):
        dx_10bar_water_2[i,j]=x_fluid_10bar_water_2[i,j]-x_fluid_10bar_water_2[i-1,j]
p_10bar_water_2= np.ones((params.n,len(sol_10bar_water_2.t)))*params.p_0
for i in range(params.n):
    for j in range(len(sol_10bar_water_2.t)):
        p_10bar_water_2[i,j]=params.p_0-params.rho_l_water*params.c_water**2*((dx_10bar_water_2[i,j]*params.A_p)-dx_eq_10bar_water_2[i]*params.A_p) /(dx_eq_10bar_water_2[i]*params.A_p)

#no piston#
x_fluid_eq_10bar_water_no_piston = ms.equilibrate_fluid_positions_no_piston(params.n, params.L0, params.k_s_water, params.m_e_water, params.g, params.p_t0, p_p0)
x_0_10bar_water_no_piston = x_fluid_eq_10bar_water_no_piston[0]

dx_eq_10bar_water_no_piston = np.zeros(params.n)
dx_eq_10bar_water_no_piston[0] = x_fluid_eq_10bar_water_no_piston[1] - x_0_10bar_water_no_piston
for i in range(1, params.n):
    dx_eq_10bar_water_no_piston[i] = x_fluid_eq_10bar_water_no_piston[i] - x_fluid_eq_10bar_water_no_piston[i - 1]

V_p0 = params.A_p * x_0_10bar_water_no_piston  
V_t0 = params.A_p * (params.H_no_piston - x_fluid_eq_10bar_water_no_piston[params.n])  

v_0 = 0  
v_fluid0 = np.zeros(params.n)
y0 = np.concatenate((x_fluid_eq_10bar_water_no_piston, [v_0], v_fluid0))


sol_10bar_water_no_piston = solve_ivp(ms.system_water_no_piston, [0, T_10], y0, args=(V_t0,V_p0, p_p0), atol=1e-6, rtol=1e-6, max_step=1e-6)
sol_10bar_water_no_piston.t=np.array(sol_10bar_water_no_piston.t)

x_0_10bar_water_no_piston = sol_10bar_water_no_piston.y[0]  
x_fluid_10bar_water_no_piston = sol_10bar_water_no_piston.y[1:params.n + 1]  
v_fluid_10bar_water_no_piston = sol_10bar_water_no_piston.y[params.n + 1:2 * (params.n + 1)]  
a0_values_10bar_water_no_piston = [ms.liquid_acceleration_water(y, params.n, params.k_s_water, params.m_e_water, params.L0, V_p0, p_p0) for y in sol_10bar_water_no_piston.y.T]
a0_min_10bar_water_no_piston = np.min(a0_values_10bar_water_no_piston)
a0_max_10bar_water_no_piston = np.max(a0_values_10bar_water_no_piston)
v_piston_10bar_water_no_piston = sol_10bar_water_no_piston.y[params.n + 1, :]
v_piston_max_10bar_water_no_piston = np.max(v_piston_10bar_water_no_piston)

all_acc_10bar_water_no_piston = np.array([
    ms.all_accelerations_water_no_piston(y, V_t0, V_p0, p_p0)
    for y in sol_10bar_water_no_piston.y.T
])  

a_fluid_10bar_water_no_piston = all_acc_10bar_water_no_piston.T  
V_p_10bar_water_no_piston = params.A_p * x_0_10bar_water_no_piston
V_t_10bar_water_no_piston = params.A_p * (params.H_no_piston - x_fluid_10bar_water_no_piston[params.n-1, :])

p_t_10bar_water_no_piston = []
for i in range(len(sol_10bar_water_no_piston.t)):
    p_ti = ms.pressure_on_top_no_piston(sol_10bar_water_no_piston.y[:, i], V_t0)
    p_t_10bar_water_no_piston.append(p_ti)


p_p_10bar_water_no_piston = []
for i in range(len(sol_10bar_water_no_piston.t)):
    p_pi = ms.pressure_gas_bottom(sol_10bar_water_no_piston.y[:, i], V_p0, p_p0)
    p_p_10bar_water_no_piston.append(p_pi)


dx_10bar_water_no_piston = np.zeros((params.n, len(sol_10bar_water_no_piston.t)))
for j in range(len(sol_10bar_water_no_piston.t)):
    dx_10bar_water_no_piston[0, j] = x_fluid_10bar_water_no_piston[0, j] - x_0_10bar_water_no_piston[j]
    for i in range(1, params.n):
        dx_10bar_water_no_piston[i, j] = x_fluid_10bar_water_no_piston[i, j] - x_fluid_10bar_water_no_piston[i - 1, j]

p_10bar_water_no_piston = np.ones((params.n, len(sol_10bar_water_no_piston.t))) * params.p_0
for i in range(params.n):
    for j in range(len(sol_10bar_water_no_piston.t)):
        p_10bar_water_no_piston[i, j] = params.p_0 - params.rho_l_water * params.c_water**2 * ((dx_10bar_water_no_piston[i, j] * params.A_p) - dx_eq_10bar_water_no_piston[i] * params.A_p) / (dx_eq_10bar_water_no_piston[i] * params.A_p)

#lithium#
#M=1.69#
x_fluid_eq_10bar_lithium= ms.equilibrate_fluid_positions(params.n,params.L0,params.k_s_lithium,params.m_e_lithium,params.m_piston_ref,params.g,params.p_t0,p_p0)
for i in range(len(x_fluid_eq_10bar_lithium)):
    x_fluid_eq_10bar_lithium[i]=x_fluid_eq_10bar_lithium[i]+params.x_air_bottom
x_p0_10bar_lithium=x_fluid_eq_10bar_lithium[0]
offset= x_p0_10bar_lithium - 0.25
for i in range(len(x_fluid_eq_10bar_lithium)):
    x_fluid_eq_10bar_lithium[i]=x_fluid_eq_10bar_lithium[i]-offset
x_p0_10bar_lithium=x_fluid_eq_10bar_lithium[0]

dx_eq_10bar_lithium= np.zeros(params.n)
dx_eq_10bar_lithium[0]=x_fluid_eq_10bar_lithium[1]-x_p0_10bar_lithium
for i in range (1,params.n):
    dx_eq_10bar_lithium[i]=x_fluid_eq_10bar_lithium[i]-x_fluid_eq_10bar_lithium[i-1]

V_p0 = params.A_p * (x_p0_10bar_lithium-params.th_piston_ref)
V_t0 = params.A_p * (params.H_ref-x_fluid_eq_10bar_lithium[params.n]) 

v_p0 = 0 
v_fluid0=np.zeros(params.n)
y0 = np.concatenate((x_fluid_eq_10bar_lithium, [v_p0], v_fluid0))


sol_10bar_lithium = solve_ivp(ms.system_lith, [0, T_10], y0, args=(params.m_piston_ref,params.H_ref, V_t0, V_p0, params.n, p_p0), atol = 1e-6, rtol=1e-6, max_step = 1e-6)


x_p_10bar_lithium = sol_10bar_lithium.y[0]     
x_fluid_10bar_lithium = sol_10bar_lithium.y[1:params.n+1] 
v_fluid_10bar_lithium = sol_10bar_lithium.y[params.n+1:2*(params.n+1)] 
a0_values_10bar_lithium = [ms.piston_acceleration_lith(y, params.m_piston_ref, V_p0, p_p0) for y in sol_10bar_lithium.y.T]
a0_min_10bar_lithium = np.min(a0_values_10bar_lithium)
a0_max_10bar_lithium = np.max(a0_values_10bar_lithium)
v_piston_10bar_lithium = sol_10bar_lithium.y[params.n+1, :]
v_piston_max_10bar_lithium=np.max(v_piston_10bar_lithium)


p_t_10bar_lithium=[]
for i in range(len(sol_10bar_lithium.t)):
    p_ti=ms.pressure_on_top(sol_10bar_lithium.y[:,i], params.H_ref, params.th_piston_ref, V_t0, params.n)
    p_t_10bar_lithium.append(p_ti)

p_p_10bar_lithium=[]
for i in range(len(sol_10bar_lithium.t)):
    p_pi=ms.pressure_under_piston(sol_10bar_lithium.y[:,i], params.H_ref, params.th_piston_ref, V_p0, params.n,p_p0)
    p_p_10bar_lithium.append(p_pi)

dx_10bar_lithium = np.zeros((params.n, len(sol_10bar_lithium.t)))
for j in range(len(sol_10bar_lithium.t)):
    dx_10bar_lithium[0,j]=x_fluid_10bar_lithium[0,j]-x_p_10bar_lithium[j]
    for i in range (1,params.n):
        dx_10bar_lithium[i,j]=x_fluid_10bar_lithium[i,j]-x_fluid_10bar_lithium[i-1,j]
p_10bar_lithium= np.ones((params.n,len(sol_10bar_lithium.t)))*params.p_0
for i in range(params.n):
    for j in range(len(sol_10bar_lithium.t)):
        p_10bar_lithium[i,j]=params.p_0-params.rho_l_lithium*params.c_lithium**2*((dx_10bar_lithium[i,j]*params.A_p)-dx_eq_10bar_lithium[i]*params.A_p) /(dx_eq_10bar_lithium[i]*params.A_p)


#Driving pressure: 2MPa#
p_p0 = 20e5
T_20 = 0.04 

#water#
#M=0.1#
x_fluid_eq_20bar_water_01= ms.equilibrate_fluid_positions(params.n,params.L0,params.k_s_water,params.m_e_water,params.m_piston_01,params.g,params.p_t0,p_p0)
for i in range(len(x_fluid_eq_20bar_water_01)):
    x_fluid_eq_20bar_water_01[i]=x_fluid_eq_20bar_water_01[i]+params.x_air_bottom
x_p0_20bar_water_01=x_fluid_eq_20bar_water_01[0]
offset= x_p0_20bar_water_01 - (params.x_air_bottom+params.th_piston_01)
for i in range(len(x_fluid_eq_20bar_water_01)):
    x_fluid_eq_20bar_water_01[i]=x_fluid_eq_20bar_water_01[i]-offset
x_p0_20bar_water_01=x_fluid_eq_20bar_water_01[0]

dx_eq_20bar_water_01 = np.zeros(params.n)
dx_eq_20bar_water_01[0]=x_fluid_eq_20bar_water_01[1]-x_p0_20bar_water_01
for i in range (1,params.n):
    dx_eq_20bar_water_01[i]=x_fluid_eq_20bar_water_01[i]-x_fluid_eq_20bar_water_01[i-1]

V_p0_20bar_01 = params.A_p * (x_p0_20bar_water_01-params.th_piston_01) 
V_t0_20bar_01 = params.A_p * (params.H_01-x_fluid_eq_20bar_water_01[params.n])  

v_p0 = 0 
v_fluid0=np.zeros(params.n)
y0 = np.concatenate((x_fluid_eq_20bar_water_01, [v_p0], v_fluid0))

p_0_x_20bar_water_01= np.zeros(params.n)
for i in range (params.n):
    p_0_x_20bar_water_01[i] = params.p_t0 + params.rho_water*params.g*(x_fluid_eq_20bar_water_01[params.n]-x_fluid_eq_20bar_water_01[i])


sol_20bar_water_01 = solve_ivp(ms.system_water, [0, T_20], y0, args=(params.m_piston_01,params.H_01, V_t0_20bar_01, V_p0_20bar_01, params.k_s_water, params.m_e_water, params.n, p_p0), atol = 1e-6, rtol=1e-6, max_step = 1e-6)
sol_20bar_water_01.t=np.array(sol_20bar_water_01.t)

x_p_20bar_water_01 = sol_20bar_water_01.y[0]      
x_fluid_20bar_water_01 = sol_20bar_water_01.y[1:params.n+1] 
v_fluid_20bar_water_01 = sol_20bar_water_01.y[params.n+1:2*(params.n+1)] 
a0_values_20bar_water_01 = [ms.piston_acceleration_water(y,params.m_piston_01, params.H_01, V_p0_20bar_01, params.n, params.k_s_water, p_p0) for y in sol_20bar_water_01.y.T]
a0_min_20bar_water_01 = np.min(a0_values_20bar_water_01)
a0_max_20bar_water_01 = np.max(a0_values_20bar_water_01)
v_piston_20bar_water_01 = sol_20bar_water_01.y[params.n+1, :]
v_piston_max_20bar_water_01=np.max(v_piston_20bar_water_01)

all_acc_20bar_water_01 = np.array([
    ms.all_accelerations_water(y, params.m_piston_01, params.H_01, V_t0_20bar_01, V_p0_20bar_01, params.k_s_water, params.m_e_water, params.n, p_p0)
    for y in sol_20bar_water_01.y.T
])

a_piston_20bar_water_01 = all_acc_20bar_water_01[:, 0]      
a_fluid_20bar_water_01  = all_acc_20bar_water_01[:, 1:].T   
V_p_20bar_water_01 = params.A_p * (x_p_20bar_water_01 - params.th_piston_01)
V_t_20bar_water_01 = params.A_p * (params.H_01 - x_fluid_20bar_water_01[params.n-1, :])

p_t_20bar_water_01=[]
for i in range(len(sol_20bar_water_01.t)):
    p_ti=ms.pressure_on_top(sol_20bar_water_01.y[:,i], params.H_01, params.th_piston_01, V_t0_20bar_01, params.n)
    p_t_20bar_water_01.append(p_ti)
p_t_20bar_water_01 = np.array(p_t_20bar_water_01)
p_t_20bar_water_max_01 = np.max(p_t_20bar_water_01)

p_p_20bar_water_01=[]
for i in range(len(sol_20bar_water_01.t)):
    p_pi=ms.pressure_under_piston(sol_20bar_water_01.y[:,i], params.H_01, params.th_piston_01, V_p0_20bar_01, params.n,p_p0)
    p_p_20bar_water_01.append(p_pi)
p_p_20bar_water_01 = np.array(p_p_20bar_water_01)

dx_20bar_water_01 = np.zeros((params.n, len(sol_20bar_water_01.t)))
for j in range(len(sol_20bar_water_01.t)):
    dx_20bar_water_01[0,j]=x_fluid_20bar_water_01[0,j]-x_p_20bar_water_01[j]
    for i in range (1,params.n):
        dx_20bar_water_01[i,j]=x_fluid_20bar_water_01[i,j]-x_fluid_20bar_water_01[i-1,j]
p_20bar_water_01= np.ones((params.n,len(sol_20bar_water_01.t)))*params.p_0
for i in range(params.n):
    for j in range(len(sol_20bar_water_01.t)):
        p_20bar_water_01[i,j]=params.p_0-params.rho_l_water*params.c_water**2*((dx_20bar_water_01[i,j]*params.A_p)-dx_eq_20bar_water_01[i]*params.A_p) /(dx_eq_20bar_water_01[i]*params.A_p)

#M=0.5#
x_fluid_eq_20bar_water_05= ms.equilibrate_fluid_positions(params.n,params.L0,params.k_s_water,params.m_e_water,params.m_piston_05,params.g,params.p_t0,p_p0)
for i in range(len(x_fluid_eq_20bar_water_05)):
    x_fluid_eq_20bar_water_05[i]=x_fluid_eq_20bar_water_05[i]+params.x_air_bottom
x_p0_20bar_water_05=x_fluid_eq_20bar_water_05[0]
offset= x_p0_20bar_water_05 - (params.x_air_bottom+params.th_piston_05)
for i in range(len(x_fluid_eq_20bar_water_05)):
    x_fluid_eq_20bar_water_05[i]=x_fluid_eq_20bar_water_05[i]-offset
x_p0_20bar_water_05=x_fluid_eq_20bar_water_05[0]

dx_eq_20bar_water_05 = np.zeros(params.n)
dx_eq_20bar_water_05[0]=x_fluid_eq_20bar_water_05[1]-x_p0_20bar_water_05
for i in range (1,params.n):
    dx_eq_20bar_water_05[i]=x_fluid_eq_20bar_water_05[i]-x_fluid_eq_20bar_water_05[i-1]

V_p0_20bar_05 = params.A_p * (x_p0_20bar_water_05-params.th_piston_05) 
V_t0_20bar_05 = params.A_p * (params.H_05-x_fluid_eq_20bar_water_05[params.n]) 

v_p0 = 0 
v_fluid0=np.zeros(params.n)
y0 = np.concatenate((x_fluid_eq_20bar_water_05, [v_p0], v_fluid0))

p_0_x_20bar_water_05= np.zeros(params.n)
for i in range (params.n):
    p_0_x_20bar_water_05[i] = params.p_t0 + params.rho_water*params.g*(x_fluid_eq_20bar_water_05[params.n]-x_fluid_eq_20bar_water_05[i])

sol_20bar_water_05 = solve_ivp(ms.system_water, [0, T_20], y0, args=(params.m_piston_05,params.H_05, V_t0_20bar_05, V_p0_20bar_05, params.k_s_water, params.m_e_water, params.n, p_p0), atol = 1e-6, rtol=1e-6, max_step = 1e-6)
sol_20bar_water_05.t=np.array(sol_20bar_water_05.t)

x_p_20bar_water_05 = sol_20bar_water_05.y[0]      
x_fluid_20bar_water_05 = sol_20bar_water_05.y[1:params.n+1] 
v_fluid_20bar_water_05 = sol_20bar_water_05.y[params.n+1:2*(params.n+1)] 
a0_values_20bar_water_05 = [ms.piston_acceleration_water(y,params.m_piston_05, params.H_05, V_p0_20bar_05, params.n, params.k_s_water, p_p0) for y in sol_20bar_water_05.y.T]
a0_min_20bar_water_05 = np.min(a0_values_20bar_water_05)
a0_max_20bar_water_05 = np.max(a0_values_20bar_water_05)
v_piston_20bar_water_05 = sol_20bar_water_05.y[params.n+1, :]
v_piston_max_20bar_water_05=np.max(v_piston_20bar_water_05)

all_acc_20bar_water_05 = np.array([
    ms.all_accelerations_water(y, params.m_piston_05, params.H_05, V_t0_20bar_05, V_p0_20bar_05, params.k_s_water, params.m_e_water, params.n, p_p0)
    for y in sol_20bar_water_05.y.T
])

a_piston_20bar_water_05 = all_acc_20bar_water_05[:, 0]      
a_fluid_20bar_water_05  = all_acc_20bar_water_05[:, 1:].T   
V_p_20bar_water_05 = params.A_p * (x_p_20bar_water_05 - params.th_piston_05)
V_t_20bar_water_05 = params.A_p * (params.H_05 - x_fluid_20bar_water_05[params.n-1, :])

p_t_20bar_water_05=[]
for i in range(len(sol_20bar_water_05.t)):
    p_ti=ms.pressure_on_top(sol_20bar_water_05.y[:,i], params.H_05, params.th_piston_05, V_t0_20bar_05, params.n)
    p_t_20bar_water_05.append(p_ti)
p_t_20bar_water_05 = np.array(p_t_20bar_water_05)
p_t_20bar_water_max_05 = np.max(p_t_20bar_water_05)

p_p_20bar_water_05=[]
for i in range(len(sol_20bar_water_05.t)):
    p_pi=ms.pressure_under_piston(sol_20bar_water_05.y[:,i], params.H_05, params.th_piston_05, V_p0_20bar_05, params.n,p_p0)
    p_p_20bar_water_05.append(p_pi)
p_p_20bar_water_05 = np.array(p_p_20bar_water_05)

dx_20bar_water_05 = np.zeros((params.n, len(sol_20bar_water_05.t)))
for j in range(len(sol_20bar_water_05.t)):
    dx_20bar_water_05[0,j]=x_fluid_20bar_water_05[0,j]-x_p_20bar_water_05[j]
    for i in range (1,params.n):
        dx_20bar_water_05[i,j]=x_fluid_20bar_water_05[i,j]-x_fluid_20bar_water_05[i-1,j]
p_20bar_water_05= np.ones((params.n,len(sol_20bar_water_05.t)))*params.p_0
for i in range(params.n):
    for j in range(len(sol_20bar_water_05.t)):
        p_20bar_water_05[i,j]=params.p_0-params.rho_l_water*params.c_water**2*((dx_20bar_water_05[i,j]*params.A_p)-dx_eq_20bar_water_05[i]*params.A_p) /(dx_eq_20bar_water_05[i]*params.A_p)

#M=0.9#
x_fluid_eq_20bar_water_09= ms.equilibrate_fluid_positions(params.n,params.L0,params.k_s_water,params.m_e_water,params.m_piston_09,params.g,params.p_t0,p_p0)
for i in range(len(x_fluid_eq_20bar_water_09)):
    x_fluid_eq_20bar_water_09[i]=x_fluid_eq_20bar_water_09[i]+params.x_air_bottom
x_p0_20bar_water_09=x_fluid_eq_20bar_water_09[0]
offset= x_p0_20bar_water_09 - (params.x_air_bottom+params.th_piston_09)
for i in range(len(x_fluid_eq_20bar_water_09)):
    x_fluid_eq_20bar_water_09[i]=x_fluid_eq_20bar_water_09[i]-offset
x_p0_20bar_water_09=x_fluid_eq_20bar_water_09[0]

dx_eq_20bar_water_09 = np.zeros(params.n)
dx_eq_20bar_water_09[0]=x_fluid_eq_20bar_water_09[1]-x_p0_20bar_water_09
for i in range (1,params.n):
    dx_eq_20bar_water_09[i]=x_fluid_eq_20bar_water_09[i]-x_fluid_eq_20bar_water_09[i-1]

V_p0_20bar_09 = params.A_p * (x_p0_20bar_water_09-params.th_piston_09) 
V_t0_20bar_09 = params.A_p * (params.H_09-x_fluid_eq_20bar_water_09[params.n]) 

v_p0 = 0 
v_fluid0=np.zeros(params.n)
y0 = np.concatenate((x_fluid_eq_20bar_water_09, [v_p0], v_fluid0))

p_0_x_20bar_water_09= np.zeros(params.n)
for i in range (params.n):
    p_0_x_20bar_water_09[i] = params.p_t0 + params.rho_water*params.g*(x_fluid_eq_20bar_water_09[params.n]-x_fluid_eq_20bar_water_09[i])


sol_20bar_water_09 = solve_ivp(ms.system_water, [0, T_20], y0, args=(params.m_piston_09,params.H_09, V_t0_20bar_09, V_p0_20bar_09, params.k_s_water, params.m_e_water, params.n, p_p0), atol = 1e-6, rtol=1e-6, max_step = 1e-6)
sol_20bar_water_09.t=np.array(sol_20bar_water_09.t)

x_p_20bar_water_09 = sol_20bar_water_09.y[0]     
x_fluid_20bar_water_09 = sol_20bar_water_09.y[1:params.n+1] 
v_fluid_20bar_water_09 = sol_20bar_water_09.y[params.n+1:2*(params.n+1)] 
a0_values_20bar_water_09 = [ms.piston_acceleration_water(y,params.m_piston_09, params.H_09, V_p0_20bar_09, params.n, params.k_s_water, p_p0) for y in sol_20bar_water_09.y.T]
a0_min_20bar_water_09 = np.min(a0_values_20bar_water_09)
a0_max_20bar_water_09 = np.max(a0_values_20bar_water_09)
v_piston_20bar_water_09 = sol_20bar_water_09.y[params.n+1, :]
v_piston_max_20bar_water_09=np.max(v_piston_20bar_water_09)

all_acc_20bar_water_09 = np.array([
    ms.all_accelerations_water(y, params.m_piston_09, params.H_09, V_t0_20bar_09, V_p0_20bar_09, params.k_s_water, params.m_e_water, params.n, p_p0)
    for y in sol_20bar_water_09.y.T
])

a_piston_20bar_water_09 = all_acc_20bar_water_09[:, 0]      
a_fluid_20bar_water_09  = all_acc_20bar_water_09[:, 1:].T   
V_p_20bar_water_09 = params.A_p * (x_p_20bar_water_09 - params.th_piston_09)
V_t_20bar_water_09 = params.A_p * (params.H_09 - x_fluid_20bar_water_09[params.n-1, :])

p_t_20bar_water_09=[]
for i in range(len(sol_20bar_water_09.t)):
    p_ti=ms.pressure_on_top(sol_20bar_water_09.y[:,i], params.H_09, params.th_piston_09, V_t0_20bar_09, params.n)
    p_t_20bar_water_09.append(p_ti)
p_t_20bar_water_09 = np.array(p_t_20bar_water_09)
p_t_20bar_water_max_09 = np.max(p_t_20bar_water_09)

p_p_20bar_water_09=[]
for i in range(len(sol_20bar_water_09.t)):
    p_pi=ms.pressure_under_piston(sol_20bar_water_09.y[:,i], params.H_09, params.th_piston_09, V_p0_20bar_09, params.n,p_p0)
    p_p_20bar_water_09.append(p_pi)
p_p_20bar_water_09 = np.array(p_p_20bar_water_09)

dx_20bar_water_09 = np.zeros((params.n, len(sol_20bar_water_09.t)))
for j in range(len(sol_20bar_water_09.t)):
    dx_20bar_water_09[0,j]=x_fluid_20bar_water_09[0,j]-x_p_20bar_water_09[j]
    for i in range (1,params.n):
        dx_20bar_water_09[i,j]=x_fluid_20bar_water_09[i,j]-x_fluid_20bar_water_09[i-1,j]
p_20bar_water_09= np.ones((params.n,len(sol_20bar_water_09.t)))*params.p_0
for i in range(params.n):
    for j in range(len(sol_20bar_water_09.t)):
        p_20bar_water_09[i,j]=params.p_0-params.rho_l_water*params.c_water**2*((dx_20bar_water_09[i,j]*params.A_p)-dx_eq_20bar_water_09[i]*params.A_p) /(dx_eq_20bar_water_09[i]*params.A_p)

#M=1#
x_fluid_eq_20bar_water_1= ms.equilibrate_fluid_positions(params.n,params.L0,params.k_s_water,params.m_e_water,params.m_piston_1,params.g,params.p_t0,p_p0)
for i in range(len(x_fluid_eq_20bar_water_1)):
    x_fluid_eq_20bar_water_1[i]=x_fluid_eq_20bar_water_1[i]+params.x_air_bottom
x_p0_20bar_water_1=x_fluid_eq_20bar_water_1[0]
offset= x_p0_20bar_water_1 - (params.x_air_bottom+params.th_piston_1)
for i in range(len(x_fluid_eq_20bar_water_1)):
    x_fluid_eq_20bar_water_1[i]=x_fluid_eq_20bar_water_1[i]-offset
x_p0_20bar_water_1=x_fluid_eq_20bar_water_1[0]

dx_eq_20bar_water_1 = np.zeros(params.n)
dx_eq_20bar_water_1[0]=x_fluid_eq_20bar_water_1[1]-x_p0_20bar_water_1
for i in range (1,params.n):
    dx_eq_20bar_water_1[i]=x_fluid_eq_20bar_water_1[i]-x_fluid_eq_20bar_water_1[i-1]

V_p0_20bar_1 = params.A_p * (x_p0_20bar_water_1-params.th_piston_1) 
V_t0_20bar_1 = params.A_p * (params.H_1-x_fluid_eq_20bar_water_1[params.n]) 

v_p0 = 0 
v_fluid0=np.zeros(params.n)
y0 = np.concatenate((x_fluid_eq_20bar_water_1, [v_p0], v_fluid0))

p_0_x_20bar_water_1= np.zeros(params.n)
for i in range (params.n):
    p_0_x_20bar_water_1[i] = params.p_t0 + params.rho_water*params.g*(x_fluid_eq_20bar_water_1[params.n]-x_fluid_eq_20bar_water_1[i])


sol_20bar_water_1 = solve_ivp(ms.system_water, [0, T_20], y0, args=(params.m_piston_1,params.H_1, V_t0_20bar_1, V_p0_20bar_1, params.k_s_water, params.m_e_water, params.n, p_p0), atol = 1e-6, rtol=1e-6, max_step = 1e-6)
sol_20bar_water_1.t=np.array(sol_20bar_water_1.t)

x_p_20bar_water_1 = sol_20bar_water_1.y[0]      
x_fluid_20bar_water_1 = sol_20bar_water_1.y[1:params.n+1]
v_fluid_20bar_water_1 = sol_20bar_water_1.y[params.n+1:2*(params.n+1)] 
a0_values_20bar_water_1 = [ms.piston_acceleration_water(y,params.m_piston_1, params.H_1, V_p0_20bar_1, params.n, params.k_s_water, p_p0) for y in sol_20bar_water_1.y.T]
a0_min_20bar_water_1 = np.min(a0_values_20bar_water_1)
a0_max_20bar_water_1 = np.max(a0_values_20bar_water_1)
v_piston_20bar_water_1 = sol_20bar_water_1.y[params.n+1, :]
v_piston_max_20bar_water_1=np.max(v_piston_20bar_water_1)

all_acc_20bar_water_1 = np.array([
    ms.all_accelerations_water(y, params.m_piston_1, params.H_1, V_t0_20bar_1, V_p0_20bar_1, params.k_s_water, params.m_e_water, params.n, p_p0)
    for y in sol_20bar_water_1.y.T
])

a_piston_20bar_water_1 = all_acc_20bar_water_1[:, 0]    
a_fluid_20bar_water_1  = all_acc_20bar_water_1[:, 1:].T   
V_p_20bar_water_1 = params.A_p * (x_p_20bar_water_1 - params.th_piston_1)
V_t_20bar_water_1 = params.A_p * (params.H_1 - x_fluid_20bar_water_1[params.n-1, :])

p_t_20bar_water_1=[]
for i in range(len(sol_20bar_water_1.t)):
    p_ti=ms.pressure_on_top(sol_20bar_water_1.y[:,i], params.H_1, params.th_piston_1, V_t0_20bar_1, params.n)
    p_t_20bar_water_1.append(p_ti)
p_t_20bar_water_1 = np.array(p_t_20bar_water_1)
p_t_20bar_water_max_1 = np.max(p_t_20bar_water_1)

p_p_20bar_water_1=[]
for i in range(len(sol_20bar_water_1.t)):
    p_pi=ms.pressure_under_piston(sol_20bar_water_1.y[:,i], params.H_1, params.th_piston_1, V_p0_20bar_1, params.n,p_p0)
    p_p_20bar_water_1.append(p_pi)
p_p_20bar_water_1 = np.array(p_p_20bar_water_1)

dx_20bar_water_1 = np.zeros((params.n, len(sol_20bar_water_1.t)))
for j in range(len(sol_20bar_water_1.t)):
    dx_20bar_water_1[0,j]=x_fluid_20bar_water_1[0,j]-x_p_20bar_water_1[j]
    for i in range (1,params.n):
        dx_20bar_water_1[i,j]=x_fluid_20bar_water_1[i,j]-x_fluid_20bar_water_1[i-1,j]
p_20bar_water_1= np.ones((params.n,len(sol_20bar_water_1.t)))*params.p_0
for i in range(params.n):
    for j in range(len(sol_20bar_water_1.t)):
        p_20bar_water_1[i,j]=params.p_0-params.rho_l_water*params.c_water**2*((dx_20bar_water_1[i,j]*params.A_p)-dx_eq_20bar_water_1[i]*params.A_p) /(dx_eq_20bar_water_1[i]*params.A_p)

#M=1.5#
x_fluid_eq_20bar_water_15= ms.equilibrate_fluid_positions(params.n,params.L0,params.k_s_water,params.m_e_water,params.m_piston_15,params.g,params.p_t0,p_p0)
for i in range(len(x_fluid_eq_20bar_water_15)):
    x_fluid_eq_20bar_water_15[i]=x_fluid_eq_20bar_water_15[i]+params.x_air_bottom
x_p0_20bar_water_15=x_fluid_eq_20bar_water_15[0]
offset= x_p0_20bar_water_15 - (params.x_air_bottom+params.th_piston_15)
for i in range(len(x_fluid_eq_20bar_water_15)):
    x_fluid_eq_20bar_water_15[i]=x_fluid_eq_20bar_water_15[i]-offset
x_p0_20bar_water_15=x_fluid_eq_20bar_water_15[0]

dx_eq_20bar_water_15 = np.zeros(params.n)
dx_eq_20bar_water_15[0]=x_fluid_eq_20bar_water_15[1]-x_p0_20bar_water_15
for i in range (1,params.n):
    dx_eq_20bar_water_15[i]=x_fluid_eq_20bar_water_15[i]-x_fluid_eq_20bar_water_15[i-1]

V_p0_20bar_15 = params.A_p * (x_p0_20bar_water_15-params.th_piston_15) 
V_t0_20bar_15 = params.A_p * (params.H_15-x_fluid_eq_20bar_water_15[params.n])  

v_p0 = 0 
v_fluid0=np.zeros(params.n)
y0 = np.concatenate((x_fluid_eq_20bar_water_15, [v_p0], v_fluid0))

p_0_x_20bar_water_15= np.zeros(params.n)
for i in range (params.n):
    p_0_x_20bar_water_15[i] = params.p_t0 + params.rho_water*params.g*(x_fluid_eq_20bar_water_15[params.n]-x_fluid_eq_20bar_water_15[i])


sol_20bar_water_15 = solve_ivp(ms.system_water, [0, T_20], y0, args=(params.m_piston_15,params.H_15, V_t0_20bar_15, V_p0_20bar_15, params.k_s_water, params.m_e_water, params.n, p_p0), atol = 1e-6, rtol=1e-6, max_step = 1e-6)
sol_20bar_water_15.t=np.array(sol_20bar_water_15.t)

x_p_20bar_water_15 = sol_20bar_water_15.y[0]      
x_fluid_20bar_water_15 = sol_20bar_water_15.y[1:params.n+1] 
v_fluid_20bar_water_15 = sol_20bar_water_15.y[params.n+1:2*(params.n+1)] 
a0_values_20bar_water_15 = [ms.piston_acceleration_water(y,params.m_piston_15, params.H_15, V_p0_20bar_15, params.n, params.k_s_water, p_p0) for y in sol_20bar_water_15.y.T]
a0_min_20bar_water_15 = np.min(a0_values_20bar_water_15)
a0_max_20bar_water_15 = np.max(a0_values_20bar_water_15)
v_piston_20bar_water_15 = sol_20bar_water_15.y[params.n+1, :]
v_piston_max_20bar_water_15=np.max(v_piston_20bar_water_15)

all_acc_20bar_water_15 = np.array([
    ms.all_accelerations_water(y, params.m_piston_15, params.H_15, V_t0_20bar_15, V_p0_20bar_15, params.k_s_water, params.m_e_water, params.n, p_p0)
    for y in sol_20bar_water_15.y.T
])

a_piston_20bar_water_15 = all_acc_20bar_water_15[:, 0]     
a_fluid_20bar_water_15  = all_acc_20bar_water_15[:, 1:].T   
V_p_20bar_water_15 = params.A_p * (x_p_20bar_water_15 - params.th_piston_15)
V_t_20bar_water_15 = params.A_p * (params.H_15 - x_fluid_20bar_water_15[params.n-1, :])

p_t_20bar_water_15=[]
for i in range(len(sol_20bar_water_15.t)):
    p_ti=ms.pressure_on_top(sol_20bar_water_15.y[:,i], params.H_15, params.th_piston_15, V_t0_20bar_15, params.n)
    p_t_20bar_water_15.append(p_ti)
p_t_20bar_water_15 = np.array(p_t_20bar_water_15)
p_t_20bar_water_max_15 = np.max(p_t_20bar_water_15)

p_p_20bar_water_15=[]
for i in range(len(sol_20bar_water_15.t)):
    p_pi=ms.pressure_under_piston(sol_20bar_water_15.y[:,i], params.H_15, params.th_piston_15, V_p0_20bar_15, params.n,p_p0)
    p_p_20bar_water_15.append(p_pi)
p_p_20bar_water_15 = np.array(p_p_20bar_water_15)

dx_20bar_water_15 = np.zeros((params.n, len(sol_20bar_water_15.t)))
for j in range(len(sol_20bar_water_15.t)):
    dx_20bar_water_15[0,j]=x_fluid_20bar_water_15[0,j]-x_p_20bar_water_15[j]
    for i in range (1,params.n):
        dx_20bar_water_15[i,j]=x_fluid_20bar_water_15[i,j]-x_fluid_20bar_water_15[i-1,j]
p_20bar_water_15= np.ones((params.n,len(sol_20bar_water_15.t)))*params.p_0
for i in range(params.n):
    for j in range(len(sol_20bar_water_15.t)):
        p_20bar_water_15[i,j]=params.p_0-params.rho_l_water*params.c_water**2*((dx_20bar_water_15[i,j]*params.A_p)-dx_eq_20bar_water_15[i]*params.A_p) /(dx_eq_20bar_water_15[i]*params.A_p)

#M=2#
x_fluid_eq_20bar_water_2= ms.equilibrate_fluid_positions(params.n,params.L0,params.k_s_water,params.m_e_water,params.m_piston_2,params.g,params.p_t0,p_p0)
for i in range(len(x_fluid_eq_20bar_water_2)):
    x_fluid_eq_20bar_water_2[i]=x_fluid_eq_20bar_water_2[i]+params.x_air_bottom
x_p0_20bar_water_2=x_fluid_eq_20bar_water_2[0]
offset= x_p0_20bar_water_2 - (params.x_air_bottom+params.th_piston_2)
for i in range(len(x_fluid_eq_20bar_water_2)):
    x_fluid_eq_20bar_water_2[i]=x_fluid_eq_20bar_water_2[i]-offset
x_p0_20bar_water_2=x_fluid_eq_20bar_water_2[0]

dx_eq_20bar_water_2 = np.zeros(params.n)
dx_eq_20bar_water_2[0]=x_fluid_eq_20bar_water_2[1]-x_p0_20bar_water_2
for i in range (1,params.n):
    dx_eq_20bar_water_2[i]=x_fluid_eq_20bar_water_2[i]-x_fluid_eq_20bar_water_2[i-1]

V_p0_20bar_2 = params.A_p * (x_p0_20bar_water_2-params.th_piston_2) 
V_t0_20bar_2 = params.A_p * (params.H_2-x_fluid_eq_20bar_water_2[params.n])  

v_p0 = 0 
v_fluid0=np.zeros(params.n)
y0 = np.concatenate((x_fluid_eq_20bar_water_2, [v_p0], v_fluid0))

p_0_x_20bar_water_2= np.zeros(params.n)
for i in range (params.n):
    p_0_x_20bar_water_2[i] = params.p_t0 + params.rho_water*params.g*(x_fluid_eq_20bar_water_2[params.n]-x_fluid_eq_20bar_water_2[i])


sol_20bar_water_2 = solve_ivp(ms.system_water, [0, T_20], y0, args=(params.m_piston_2,params.H_2, V_t0_20bar_2, V_p0_20bar_2, params.k_s_water, params.m_e_water, params.n, p_p0), atol = 1e-6, rtol=1e-6, max_step = 1e-6)
sol_20bar_water_2.t=np.array(sol_20bar_water_2.t)

x_p_20bar_water_2 = sol_20bar_water_2.y[0]      
x_fluid_20bar_water_2 = sol_20bar_water_2.y[1:params.n+1] 
v_fluid_20bar_water_2 = sol_20bar_water_2.y[params.n+1:2*(params.n+1)] 
a0_values_20bar_water_2 = [ms.piston_acceleration_water(y,params.m_piston_2, params.H_2, V_p0_20bar_2, params.n, params.k_s_water, p_p0) for y in sol_20bar_water_2.y.T]
a0_min_20bar_water_2 = np.min(a0_values_20bar_water_2)
a0_max_20bar_water_2 = np.max(a0_values_20bar_water_2)
v_piston_20bar_water_2 = sol_20bar_water_2.y[params.n+1, :]
v_piston_max_20bar_water_2=np.max(v_piston_20bar_water_2)

all_acc_20bar_water_2 = np.array([
    ms.all_accelerations_water(y, params.m_piston_2, params.H_2, V_t0_20bar_2, V_p0_20bar_2, params.k_s_water, params.m_e_water, params.n, p_p0)
    for y in sol_20bar_water_2.y.T
])

a_piston_20bar_water_2 = all_acc_20bar_water_2[:, 0]     
a_fluid_20bar_water_2  = all_acc_20bar_water_2[:, 1:].T   
V_p_20bar_water_2 = params.A_p * (x_p_20bar_water_2 - params.th_piston_2)
V_t_20bar_water_2 = params.A_p * (params.H_2 - x_fluid_20bar_water_2[params.n-1, :])

p_t_20bar_water_2=[]
for i in range(len(sol_20bar_water_2.t)):
    p_ti=ms.pressure_on_top(sol_20bar_water_2.y[:,i], params.H_2, params.th_piston_2, V_t0_20bar_2, params.n)
    p_t_20bar_water_2.append(p_ti)
p_t_20bar_water_2 = np.array(p_t_20bar_water_2)
p_t_20bar_water_max_2 = np.max(p_t_20bar_water_2)

p_p_20bar_water_2=[]
for i in range(len(sol_20bar_water_2.t)):
    p_pi=ms.pressure_under_piston(sol_20bar_water_2.y[:,i], params.H_2, params.th_piston_2, V_p0_20bar_2, params.n,p_p0)
    p_p_20bar_water_2.append(p_pi)
p_p_20bar_water_2 = np.array(p_p_20bar_water_2)

dx_20bar_water_2 = np.zeros((params.n, len(sol_20bar_water_2.t)))
for j in range(len(sol_20bar_water_2.t)):
    dx_20bar_water_2[0,j]=x_fluid_20bar_water_2[0,j]-x_p_20bar_water_2[j]
    for i in range (1,params.n):
        dx_20bar_water_2[i,j]=x_fluid_20bar_water_2[i,j]-x_fluid_20bar_water_2[i-1,j]
p_20bar_water_2= np.ones((params.n,len(sol_20bar_water_2.t)))*params.p_0
for i in range(params.n):
    for j in range(len(sol_20bar_water_2.t)):
        p_20bar_water_2[i,j]=params.p_0-params.rho_l_water*params.c_water**2*((dx_20bar_water_2[i,j]*params.A_p)-dx_eq_20bar_water_2[i]*params.A_p) /(dx_eq_20bar_water_2[i]*params.A_p)

#no piston#
x_fluid_eq_20bar_water_no_piston = ms.equilibrate_fluid_positions_no_piston(params.n, params.L0, params.k_s_water, params.m_e_water, params.g, params.p_t0, p_p0)
x_0_20bar_water_no_piston = x_fluid_eq_20bar_water_no_piston[0]

dx_eq_20bar_water_no_piston = np.zeros(params.n)
dx_eq_20bar_water_no_piston[0] = x_fluid_eq_20bar_water_no_piston[1] - x_0_20bar_water_no_piston
for i in range(1, params.n):
    dx_eq_20bar_water_no_piston[i] = x_fluid_eq_20bar_water_no_piston[i] - x_fluid_eq_20bar_water_no_piston[i - 1]

V_p0 = params.A_p * x_0_20bar_water_no_piston  
V_t0 = params.A_p * (params.H_no_piston - x_fluid_eq_20bar_water_no_piston[params.n])  

v_0 = 0  
v_fluid0 = np.zeros(params.n)
y0 = np.concatenate((x_fluid_eq_20bar_water_no_piston, [v_0], v_fluid0))


sol_20bar_water_no_piston = solve_ivp(ms.system_water_no_piston, [0, T_20], y0, args=(V_t0,V_p0, p_p0), atol=1e-6, rtol=1e-6, max_step=1e-6)
sol_20bar_water_no_piston.t=np.array(sol_20bar_water_no_piston.t)

x_0_20bar_water_no_piston = sol_20bar_water_no_piston.y[0]  
x_fluid_20bar_water_no_piston = sol_20bar_water_no_piston.y[1:params.n + 1]  
v_fluid_20bar_water_no_piston = sol_20bar_water_no_piston.y[params.n + 1:2 * (params.n + 1)]  
a0_values_20bar_water_no_piston = [ms.liquid_acceleration_water(y, params.n, params.k_s_water, params.m_e_water, params.L0, V_p0, p_p0) for y in sol_20bar_water_no_piston.y.T]
a0_min_20bar_water_no_piston = np.min(a0_values_20bar_water_no_piston)
a0_max_20bar_water_no_piston = np.max(a0_values_20bar_water_no_piston)
v_piston_20bar_water_no_piston = sol_20bar_water_no_piston.y[params.n + 1, :]
v_piston_max_20bar_water_no_piston = np.max(v_piston_20bar_water_no_piston)

all_acc_20bar_water_no_piston = np.array([
    ms.all_accelerations_water_no_piston(y, V_t0, V_p0, p_p0)
    for y in sol_20bar_water_no_piston.y.T
])  

a_fluid_20bar_water_no_piston = all_acc_20bar_water_no_piston.T 
V_p_20bar_water_no_piston = params.A_p * x_0_20bar_water_no_piston
V_t_20bar_water_no_piston = params.A_p * (params.H_no_piston - x_fluid_20bar_water_no_piston[params.n-1, :])

p_t_20bar_water_no_piston = []
for i in range(len(sol_20bar_water_no_piston.t)):
    p_ti = ms.pressure_on_top_no_piston(sol_20bar_water_no_piston.y[:, i], V_t0)
    p_t_20bar_water_no_piston.append(p_ti)

p_p_20bar_water_no_piston = []
for i in range(len(sol_20bar_water_no_piston.t)):
    p_pi = ms.pressure_gas_bottom(sol_20bar_water_no_piston.y[:, i], V_p0, p_p0)
    p_p_20bar_water_no_piston.append(p_pi)


dx_20bar_water_no_piston = np.zeros((params.n, len(sol_20bar_water_no_piston.t)))
for j in range(len(sol_20bar_water_no_piston.t)):
    dx_20bar_water_no_piston[0, j] = x_fluid_20bar_water_no_piston[0, j] - x_0_20bar_water_no_piston[j]
    for i in range(1, params.n):
        dx_20bar_water_no_piston[i, j] = x_fluid_20bar_water_no_piston[i, j] - x_fluid_20bar_water_no_piston[i - 1, j]

p_20bar_water_no_piston = np.ones((params.n, len(sol_20bar_water_no_piston.t))) * params.p_0
for i in range(params.n):
    for j in range(len(sol_20bar_water_no_piston.t)):
        p_20bar_water_no_piston[i, j] = params.p_0 - params.rho_l_water * params.c_water**2 * ((dx_20bar_water_no_piston[i, j] * params.A_p) - dx_eq_20bar_water_no_piston[i] * params.A_p) / (dx_eq_20bar_water_no_piston[i] * params.A_p)

#lithium#
#M=1.69#
x_fluid_eq_20bar_lithium= ms.equilibrate_fluid_positions(params.n,params.L0,params.k_s_lithium,params.m_e_lithium,params.m_piston_ref,params.g,params.p_t0,p_p0)
for i in range(len(x_fluid_eq_20bar_lithium)):
    x_fluid_eq_20bar_lithium[i]=x_fluid_eq_20bar_lithium[i]+params.x_air_bottom
x_p0_20bar_lithium=x_fluid_eq_20bar_lithium[0]
offset= x_p0_20bar_lithium - 0.25
for i in range(len(x_fluid_eq_20bar_lithium)):
    x_fluid_eq_20bar_lithium[i]=x_fluid_eq_20bar_lithium[i]-offset
x_p0_20bar_lithium=x_fluid_eq_20bar_lithium[0]

dx_eq_20bar_lithium= np.zeros(params.n)
dx_eq_20bar_lithium[0]=x_fluid_eq_20bar_lithium[1]-x_p0_20bar_lithium
for i in range (1,params.n):
    dx_eq_20bar_lithium[i]=x_fluid_eq_20bar_lithium[i]-x_fluid_eq_20bar_lithium[i-1]

V_p0 = params.A_p * (x_p0_20bar_lithium-params.th_piston_ref) 
V_t0 = params.A_p * (params.H_ref-x_fluid_eq_20bar_lithium[params.n])  

v_p0 = 0 
v_fluid0=np.zeros(params.n)
y0 = np.concatenate((x_fluid_eq_20bar_lithium, [v_p0], v_fluid0))

sol_20bar_lithium = solve_ivp(ms.system_lith, [0, T_20], y0, args=(params.m_piston_ref,params.H_ref, V_t0, V_p0, params.n, p_p0), atol = 1e-6, rtol=1e-6, max_step = 1e-6)

x_p_20bar_lithium = sol_20bar_lithium.y[0]    
x_fluid_20bar_lithium = sol_20bar_lithium.y[1:params.n+1] 
v_fluid_20bar_lithium = sol_20bar_lithium.y[params.n+1:2*(params.n+1)] 
a0_values_20bar_lithium = [ms.piston_acceleration_lith(y, params.m_piston_ref, V_p0, p_p0) for y in sol_20bar_lithium.y.T]
a0_min_20bar_lithium = np.min(a0_values_20bar_lithium)
a0_max_20bar_lithium = np.max(a0_values_20bar_lithium)
v_piston_20bar_lithium = sol_20bar_lithium.y[params.n+1, :]
v_piston_max_20bar_lithium=np.max(v_piston_20bar_lithium)

p_t_20bar_lithium=[]
for i in range(len(sol_20bar_lithium.t)):
    p_ti=ms.pressure_on_top(sol_20bar_lithium.y[:,i], params.H_ref, params.th_piston_ref, V_t0, params.n)
    p_t_20bar_lithium.append(p_ti)


p_p_20bar_lithium=[]
for i in range(len(sol_20bar_lithium.t)):
    p_pi=ms.pressure_under_piston(sol_20bar_lithium.y[:,i], params.H_ref, params.th_piston_ref, V_p0, params.n,p_p0)
    p_p_20bar_lithium.append(p_pi)

dx_20bar_lithium = np.zeros((params.n, len(sol_20bar_lithium.t)))
for j in range(len(sol_20bar_lithium.t)):
    dx_20bar_lithium[0,j]=x_fluid_20bar_lithium[0,j]-x_p_20bar_lithium[j]
    for i in range (1,params.n):
        dx_20bar_lithium[i,j]=x_fluid_20bar_lithium[i,j]-x_fluid_20bar_lithium[i-1,j]
p_20bar_lithium= np.ones((params.n,len(sol_20bar_lithium.t)))*params.p_0
for i in range(params.n):
    for j in range(len(sol_20bar_lithium.t)):
        p_20bar_lithium[i,j]=params.p_0-params.rho_l_lithium*params.c_lithium**2*((dx_20bar_lithium[i,j]*params.A_p)-dx_eq_20bar_lithium[i]*params.A_p) /(dx_eq_20bar_lithium[i]*params.A_p)


#Driving pressure: 3MPa#
p_p0 = 30e5
T_30 = 0.02 

#water#
#M=0.1#
x_fluid_eq_30bar_water_01= ms.equilibrate_fluid_positions(params.n,params.L0,params.k_s_water,params.m_e_water,params.m_piston_01,params.g,params.p_t0,p_p0)
for i in range(len(x_fluid_eq_30bar_water_01)):
    x_fluid_eq_30bar_water_01[i]=x_fluid_eq_30bar_water_01[i]+params.x_air_bottom
x_p0_30bar_water_01=x_fluid_eq_30bar_water_01[0]
offset= x_p0_30bar_water_01 - (params.x_air_bottom+params.th_piston_01)
for i in range(len(x_fluid_eq_30bar_water_01)):
    x_fluid_eq_30bar_water_01[i]=x_fluid_eq_30bar_water_01[i]-offset
x_p0_30bar_water_01=x_fluid_eq_30bar_water_01[0]

dx_eq_30bar_water_01 = np.zeros(params.n)
dx_eq_30bar_water_01[0]=x_fluid_eq_30bar_water_01[1]-x_p0_30bar_water_01
for i in range (1,params.n):
    dx_eq_30bar_water_01[i]=x_fluid_eq_30bar_water_01[i]-x_fluid_eq_30bar_water_01[i-1]

V_p0_30bar_01 = params.A_p * (x_p0_30bar_water_01-params.th_piston_01)
V_t0_30bar_01 = params.A_p * (params.H_01-x_fluid_eq_30bar_water_01[params.n]) 

v_p0 = 0 
v_fluid0=np.zeros(params.n)
y0 = np.concatenate((x_fluid_eq_30bar_water_01, [v_p0], v_fluid0))

p_0_x_30bar_water_01= np.zeros(params.n)
for i in range (params.n):
    p_0_x_30bar_water_01[i] = params.p_t0 + params.rho_water*params.g*(x_fluid_eq_30bar_water_01[params.n]-x_fluid_eq_30bar_water_01[i])

sol_30bar_water_01 = solve_ivp(ms.system_water, [0, T_30], y0, args=(params.m_piston_01,params.H_01, V_t0_30bar_01, V_p0_30bar_01, params.k_s_water, params.m_e_water, params.n, p_p0), atol = 1e-6, rtol=1e-6, max_step = 1e-6)
sol_30bar_water_01.t=np.array(sol_30bar_water_01.t)

x_p_30bar_water_01 = sol_30bar_water_01.y[0]      
x_fluid_30bar_water_01 = sol_30bar_water_01.y[1:params.n+1] 
v_fluid_30bar_water_01 = sol_30bar_water_01.y[params.n+1:2*(params.n+1)] 
a0_values_30bar_water_01 = [ms.piston_acceleration_water(y,params.m_piston_01, params.H_01, V_p0_30bar_01, params.n, params.k_s_water, p_p0) for y in sol_30bar_water_01.y.T]
a0_min_30bar_water_01 = np.min(a0_values_30bar_water_01)
a0_max_30bar_water_01 = np.max(a0_values_30bar_water_01)
v_piston_30bar_water_01 = sol_30bar_water_01.y[params.n+1, :]
v_piston_max_30bar_water_01=np.max(v_piston_30bar_water_01)

all_acc_30bar_water_01 = np.array([
    ms.all_accelerations_water(y, params.m_piston_01, params.H_01, V_t0_30bar_01, V_p0_30bar_01, params.k_s_water, params.m_e_water, params.n, p_p0)
    for y in sol_30bar_water_01.y.T
])

a_piston_30bar_water_01 = all_acc_30bar_water_01[:, 0]     
a_fluid_30bar_water_01  = all_acc_30bar_water_01[:, 1:].T   
V_p_30bar_water_01 = params.A_p * (x_p_30bar_water_01 - params.th_piston_01)
V_t_30bar_water_01 = params.A_p * (params.H_01 - x_fluid_30bar_water_01[params.n-1, :])

p_t_30bar_water_01=[]
for i in range(len(sol_30bar_water_01.t)):
    p_ti=ms.pressure_on_top(sol_30bar_water_01.y[:,i], params.H_01, params.th_piston_01, V_t0_30bar_01, params.n)
    p_t_30bar_water_01.append(p_ti)
p_t_30bar_water_01 = np.array(p_t_30bar_water_01)
p_t_30bar_water_max_01 = np.max(p_t_30bar_water_01)

p_p_30bar_water_01=[]
for i in range(len(sol_30bar_water_01.t)):
    p_pi=ms.pressure_under_piston(sol_30bar_water_01.y[:,i], params.H_01, params.th_piston_01, V_p0_30bar_01, params.n,p_p0)
    p_p_30bar_water_01.append(p_pi)
p_p_30bar_water_01 = np.array(p_p_30bar_water_01)

dx_30bar_water_01 = np.zeros((params.n, len(sol_30bar_water_01.t)))
for j in range(len(sol_30bar_water_01.t)):
    dx_30bar_water_01[0,j]=x_fluid_30bar_water_01[0,j]-x_p_30bar_water_01[j]
    for i in range (1,params.n):
        dx_30bar_water_01[i,j]=x_fluid_30bar_water_01[i,j]-x_fluid_30bar_water_01[i-1,j]
p_30bar_water_01= np.ones((params.n,len(sol_30bar_water_01.t)))*params.p_0
for i in range(params.n):
    for j in range(len(sol_30bar_water_01.t)):
        p_30bar_water_01[i,j]=params.p_0-params.rho_l_water*params.c_water**2*((dx_30bar_water_01[i,j]*params.A_p)-dx_eq_30bar_water_01[i]*params.A_p) /(dx_eq_30bar_water_01[i]*params.A_p)

#M=0.5#
x_fluid_eq_30bar_water_05= ms.equilibrate_fluid_positions(params.n,params.L0,params.k_s_water,params.m_e_water,params.m_piston_05,params.g,params.p_t0,p_p0)
for i in range(len(x_fluid_eq_30bar_water_05)):
    x_fluid_eq_30bar_water_05[i]=x_fluid_eq_30bar_water_05[i]+params.x_air_bottom
x_p0_30bar_water_05=x_fluid_eq_30bar_water_05[0]
offset= x_p0_30bar_water_05 - (params.x_air_bottom+params.th_piston_05)
for i in range(len(x_fluid_eq_30bar_water_05)):
    x_fluid_eq_30bar_water_05[i]=x_fluid_eq_30bar_water_05[i]-offset
x_p0_30bar_water_05=x_fluid_eq_30bar_water_05[0]

dx_eq_30bar_water_05 = np.zeros(params.n)
dx_eq_30bar_water_05[0]=x_fluid_eq_30bar_water_05[1]-x_p0_30bar_water_05
for i in range (1,params.n):
    dx_eq_30bar_water_05[i]=x_fluid_eq_30bar_water_05[i]-x_fluid_eq_30bar_water_05[i-1]

V_p0_30bar_05 = params.A_p * (x_p0_30bar_water_05-params.th_piston_05) 
V_t0_30bar_05 = params.A_p * (params.H_05-x_fluid_eq_30bar_water_05[params.n])  

v_p0 = 0 
v_fluid0=np.zeros(params.n)
y0 = np.concatenate((x_fluid_eq_30bar_water_05, [v_p0], v_fluid0))

p_0_x_30bar_water_05= np.zeros(params.n)
for i in range (params.n):
    p_0_x_30bar_water_05[i] = params.p_t0 + params.rho_water*params.g*(x_fluid_eq_30bar_water_05[params.n]-x_fluid_eq_30bar_water_05[i])


sol_30bar_water_05 = solve_ivp(ms.system_water, [0, T_30], y0, args=(params.m_piston_05,params.H_05, V_t0_30bar_05, V_p0_30bar_05, params.k_s_water, params.m_e_water, params.n, p_p0), atol = 1e-6, rtol=1e-6, max_step = 1e-6)
sol_30bar_water_05.t=np.array(sol_30bar_water_05.t)

x_p_30bar_water_05 = sol_30bar_water_05.y[0]     
x_fluid_30bar_water_05 = sol_30bar_water_05.y[1:params.n+1] 
v_fluid_30bar_water_05 = sol_30bar_water_05.y[params.n+1:2*(params.n+1)]
a0_values_30bar_water_05 = [ms.piston_acceleration_water(y,params.m_piston_05, params.H_05, V_p0_30bar_05, params.n, params.k_s_water, p_p0) for y in sol_30bar_water_05.y.T]
a0_min_30bar_water_05 = np.min(a0_values_30bar_water_05)
a0_max_30bar_water_05 = np.max(a0_values_30bar_water_05)
v_piston_30bar_water_05 = sol_30bar_water_05.y[params.n+1, :]
v_piston_max_30bar_water_05=np.max(v_piston_30bar_water_05)

all_acc_30bar_water_05 = np.array([
    ms.all_accelerations_water(y, params.m_piston_05, params.H_05, V_t0_30bar_05, V_p0_30bar_05, params.k_s_water, params.m_e_water, params.n, p_p0)
    for y in sol_30bar_water_05.y.T
])

a_piston_30bar_water_05 = all_acc_30bar_water_05[:, 0]     
a_fluid_30bar_water_05  = all_acc_30bar_water_05[:, 1:].T  
V_p_30bar_water_05 = params.A_p * (x_p_30bar_water_05 - params.th_piston_05)
V_t_30bar_water_05 = params.A_p * (params.H_05 - x_fluid_30bar_water_05[params.n-1, :])

p_t_30bar_water_05=[]
for i in range(len(sol_30bar_water_05.t)):
    p_ti=ms.pressure_on_top(sol_30bar_water_05.y[:,i], params.H_05, params.th_piston_05, V_t0_30bar_05, params.n)
    p_t_30bar_water_05.append(p_ti)
p_t_30bar_water_05 = np.array(p_t_30bar_water_05)
p_t_30bar_water_max_05 = np.max(p_t_30bar_water_05)

p_p_30bar_water_05=[]
for i in range(len(sol_30bar_water_05.t)):
    p_pi=ms.pressure_under_piston(sol_30bar_water_05.y[:,i], params.H_05, params.th_piston_05, V_p0_30bar_05, params.n,p_p0)
    p_p_30bar_water_05.append(p_pi)
p_p_30bar_water_05 = np.array(p_p_30bar_water_05)

dx_30bar_water_05 = np.zeros((params.n, len(sol_30bar_water_05.t)))
for j in range(len(sol_30bar_water_05.t)):
    dx_30bar_water_05[0,j]=x_fluid_30bar_water_05[0,j]-x_p_30bar_water_05[j]
    for i in range (1,params.n):
        dx_30bar_water_05[i,j]=x_fluid_30bar_water_05[i,j]-x_fluid_30bar_water_05[i-1,j]
p_30bar_water_05= np.ones((params.n,len(sol_30bar_water_05.t)))*params.p_0
for i in range(params.n):
    for j in range(len(sol_30bar_water_05.t)):
        p_30bar_water_05[i,j]=params.p_0-params.rho_l_water*params.c_water**2*((dx_30bar_water_05[i,j]*params.A_p)-dx_eq_30bar_water_05[i]*params.A_p) /(dx_eq_30bar_water_05[i]*params.A_p)

#M=0.9#
x_fluid_eq_30bar_water_09= ms.equilibrate_fluid_positions(params.n,params.L0,params.k_s_water,params.m_e_water,params.m_piston_09,params.g,params.p_t0,p_p0)
for i in range(len(x_fluid_eq_30bar_water_09)):
    x_fluid_eq_30bar_water_09[i]=x_fluid_eq_30bar_water_09[i]+params.x_air_bottom
x_p0_30bar_water_09=x_fluid_eq_30bar_water_09[0]
offset= x_p0_30bar_water_09 - (params.x_air_bottom+params.th_piston_09)
for i in range(len(x_fluid_eq_30bar_water_09)):
    x_fluid_eq_30bar_water_09[i]=x_fluid_eq_30bar_water_09[i]-offset
x_p0_30bar_water_09=x_fluid_eq_30bar_water_09[0]

dx_eq_30bar_water_09 = np.zeros(params.n)
dx_eq_30bar_water_09[0]=x_fluid_eq_30bar_water_09[1]-x_p0_30bar_water_09
for i in range (1,params.n):
    dx_eq_30bar_water_09[i]=x_fluid_eq_30bar_water_09[i]-x_fluid_eq_30bar_water_09[i-1]

V_p0_30bar_09 = params.A_p * (x_p0_30bar_water_09-params.th_piston_09) 
V_t0_30bar_09 = params.A_p * (params.H_09-x_fluid_eq_30bar_water_09[params.n]) 

v_p0 = 0 
v_fluid0=np.zeros(params.n)
y0 = np.concatenate((x_fluid_eq_30bar_water_09, [v_p0], v_fluid0))

p_0_x_30bar_water_09= np.zeros(params.n)
for i in range (params.n):
    p_0_x_30bar_water_09[i] = params.p_t0 + params.rho_water*params.g*(x_fluid_eq_30bar_water_09[params.n]-x_fluid_eq_30bar_water_09[i])


sol_30bar_water_09 = solve_ivp(ms.system_water, [0, T_30], y0, args=(params.m_piston_09,params.H_09, V_t0_30bar_09, V_p0_30bar_09, params.k_s_water, params.m_e_water, params.n, p_p0), atol = 1e-6, rtol=1e-6, max_step = 1e-6)
sol_30bar_water_09.t=np.array(sol_30bar_water_09.t)

x_p_30bar_water_09 = sol_30bar_water_09.y[0]      
x_fluid_30bar_water_09 = sol_30bar_water_09.y[1:params.n+1] 
v_fluid_30bar_water_09 = sol_30bar_water_09.y[params.n+1:2*(params.n+1)]
a0_values_30bar_water_09 = [ms.piston_acceleration_water(y,params.m_piston_09, params.H_09, V_p0_30bar_09, params.n, params.k_s_water, p_p0) for y in sol_30bar_water_09.y.T]
a0_min_30bar_water_09 = np.min(a0_values_30bar_water_09)
a0_max_30bar_water_09 = np.max(a0_values_30bar_water_09)
v_piston_30bar_water_09 = sol_30bar_water_09.y[params.n+1, :]
v_piston_max_30bar_water_09=np.max(v_piston_30bar_water_09)

all_acc_30bar_water_09 = np.array([
    ms.all_accelerations_water(y, params.m_piston_09, params.H_09, V_t0_30bar_09, V_p0_30bar_09, params.k_s_water, params.m_e_water, params.n, p_p0)
    for y in sol_30bar_water_09.y.T
])

a_piston_30bar_water_09 = all_acc_30bar_water_09[:, 0]     
a_fluid_30bar_water_09  = all_acc_30bar_water_09[:, 1:].T   
V_p_30bar_water_09 = params.A_p * (x_p_30bar_water_09 - params.th_piston_09)
V_t_30bar_water_09 = params.A_p * (params.H_09 - x_fluid_30bar_water_09[params.n-1, :])

p_t_30bar_water_09=[]
for i in range(len(sol_30bar_water_09.t)):
    p_ti=ms.pressure_on_top(sol_30bar_water_09.y[:,i], params.H_09, params.th_piston_09, V_t0_30bar_09, params.n)
    p_t_30bar_water_09.append(p_ti)
p_t_30bar_water_09 = np.array(p_t_30bar_water_09)
p_t_30bar_water_max_09 = np.max(p_t_30bar_water_09)

p_p_30bar_water_09=[]
for i in range(len(sol_30bar_water_09.t)):
    p_pi=ms.pressure_under_piston(sol_30bar_water_09.y[:,i], params.H_09, params.th_piston_09, V_p0_30bar_09, params.n,p_p0)
    p_p_30bar_water_09.append(p_pi)
p_p_30bar_water_09 = np.array(p_p_30bar_water_09)

dx_30bar_water_09 = np.zeros((params.n, len(sol_30bar_water_09.t)))
for j in range(len(sol_30bar_water_09.t)):
    dx_30bar_water_09[0,j]=x_fluid_30bar_water_09[0,j]-x_p_30bar_water_09[j]
    for i in range (1,params.n):
        dx_30bar_water_09[i,j]=x_fluid_30bar_water_09[i,j]-x_fluid_30bar_water_09[i-1,j]
p_30bar_water_09= np.ones((params.n,len(sol_30bar_water_09.t)))*params.p_0
for i in range(params.n):
    for j in range(len(sol_30bar_water_09.t)):
        p_30bar_water_09[i,j]=params.p_0-params.rho_l_water*params.c_water**2*((dx_30bar_water_09[i,j]*params.A_p)-dx_eq_30bar_water_09[i]*params.A_p) /(dx_eq_30bar_water_09[i]*params.A_p)

#M=1#
x_fluid_eq_30bar_water_1= ms.equilibrate_fluid_positions(params.n,params.L0,params.k_s_water,params.m_e_water,params.m_piston_1,params.g,params.p_t0,p_p0)
for i in range(len(x_fluid_eq_30bar_water_1)):
    x_fluid_eq_30bar_water_1[i]=x_fluid_eq_30bar_water_1[i]+params.x_air_bottom
x_p0_30bar_water_1=x_fluid_eq_30bar_water_1[0]
offset= x_p0_30bar_water_1 - (params.x_air_bottom+params.th_piston_1)
for i in range(len(x_fluid_eq_30bar_water_1)):
    x_fluid_eq_30bar_water_1[i]=x_fluid_eq_30bar_water_1[i]-offset
x_p0_30bar_water_1=x_fluid_eq_30bar_water_1[0]

dx_eq_30bar_water_1 = np.zeros(params.n)
dx_eq_30bar_water_1[0]=x_fluid_eq_30bar_water_1[1]-x_p0_30bar_water_1
for i in range (1,params.n):
    dx_eq_30bar_water_1[i]=x_fluid_eq_30bar_water_1[i]-x_fluid_eq_30bar_water_1[i-1]

V_p0_30bar_1 = params.A_p * (x_p0_30bar_water_1-params.th_piston_1) 
V_t0_30bar_1 = params.A_p * (params.H_1-x_fluid_eq_30bar_water_1[params.n]) 

v_p0 = 0 
v_fluid0=np.zeros(params.n)
y0 = np.concatenate((x_fluid_eq_30bar_water_1, [v_p0], v_fluid0))

p_0_x_30bar_water_1= np.zeros(params.n)
for i in range (params.n):
    p_0_x_30bar_water_1[i] = params.p_t0 + params.rho_water*params.g*(x_fluid_eq_30bar_water_1[params.n]-x_fluid_eq_30bar_water_1[i])


sol_30bar_water_1 = solve_ivp(ms.system_water, [0, T_30], y0, args=(params.m_piston_1,params.H_1, V_t0_30bar_1, V_p0_30bar_1, params.k_s_water, params.m_e_water, params.n, p_p0), atol = 1e-6, rtol=1e-6, max_step = 1e-6)
sol_30bar_water_1.t=np.array(sol_30bar_water_1.t)

x_p_30bar_water_1 = sol_30bar_water_1.y[0]    
x_fluid_30bar_water_1 = sol_30bar_water_1.y[1:params.n+1] 
v_fluid_30bar_water_1 = sol_30bar_water_1.y[params.n+1:2*(params.n+1)] 
a0_values_30bar_water_1 = [ms.piston_acceleration_water(y,params.m_piston_1, params.H_1, V_p0_30bar_1, params.n, params.k_s_water, p_p0) for y in sol_30bar_water_1.y.T]
a0_min_30bar_water_1 = np.min(a0_values_30bar_water_1)
a0_max_30bar_water_1 = np.max(a0_values_30bar_water_1)
v_piston_30bar_water_1 = sol_30bar_water_1.y[params.n+1, :]
v_piston_max_30bar_water_1=np.max(v_piston_30bar_water_1)

all_acc_30bar_water_1 = np.array([
    ms.all_accelerations_water(y, params.m_piston_1, params.H_1, V_t0_30bar_1, V_p0_30bar_1, params.k_s_water, params.m_e_water, params.n, p_p0)
    for y in sol_30bar_water_1.y.T
])

a_piston_30bar_water_1 = all_acc_30bar_water_1[:, 0]      
a_fluid_30bar_water_1  = all_acc_30bar_water_1[:, 1:].T   
V_p_30bar_water_1 = params.A_p * (x_p_30bar_water_1 - params.th_piston_1)
V_t_30bar_water_1 = params.A_p * (params.H_1 - x_fluid_30bar_water_1[params.n-1, :])

p_t_30bar_water_1=[]
for i in range(len(sol_30bar_water_1.t)):
    p_ti=ms.pressure_on_top(sol_30bar_water_1.y[:,i], params.H_1, params.th_piston_1, V_t0_30bar_1, params.n)
    p_t_30bar_water_1.append(p_ti)
p_t_30bar_water_1 = np.array(p_t_30bar_water_1)
p_t_30bar_water_max_1 = np.max(p_t_30bar_water_1)

p_p_30bar_water_1=[]
for i in range(len(sol_30bar_water_1.t)):
    p_pi=ms.pressure_under_piston(sol_30bar_water_1.y[:,i], params.H_1, params.th_piston_1, V_p0_30bar_1, params.n,p_p0)
    p_p_30bar_water_1.append(p_pi)
p_p_30bar_water_1 = np.array(p_p_30bar_water_1)

dx_30bar_water_1 = np.zeros((params.n, len(sol_30bar_water_1.t)))
for j in range(len(sol_30bar_water_1.t)):
    dx_30bar_water_1[0,j]=x_fluid_30bar_water_1[0,j]-x_p_30bar_water_1[j]
    for i in range (1,params.n):
        dx_30bar_water_1[i,j]=x_fluid_30bar_water_1[i,j]-x_fluid_30bar_water_1[i-1,j]
p_30bar_water_1= np.ones((params.n,len(sol_30bar_water_1.t)))*params.p_0
for i in range(params.n):
    for j in range(len(sol_30bar_water_1.t)):
        p_30bar_water_1[i,j]=params.p_0-params.rho_l_water*params.c_water**2*((dx_30bar_water_1[i,j]*params.A_p)-dx_eq_30bar_water_1[i]*params.A_p) /(dx_eq_30bar_water_1[i]*params.A_p)

#M=1.5#
x_fluid_eq_30bar_water_15= ms.equilibrate_fluid_positions(params.n,params.L0,params.k_s_water,params.m_e_water,params.m_piston_15,params.g,params.p_t0,p_p0)
for i in range(len(x_fluid_eq_30bar_water_15)):
    x_fluid_eq_30bar_water_15[i]=x_fluid_eq_30bar_water_15[i]+params.x_air_bottom
x_p0_30bar_water_15=x_fluid_eq_30bar_water_15[0]
offset= x_p0_30bar_water_15 - (params.x_air_bottom+params.th_piston_15)
for i in range(len(x_fluid_eq_30bar_water_15)):
    x_fluid_eq_30bar_water_15[i]=x_fluid_eq_30bar_water_15[i]-offset
x_p0_30bar_water_15=x_fluid_eq_30bar_water_15[0]

dx_eq_30bar_water_15 = np.zeros(params.n)
dx_eq_30bar_water_15[0]=x_fluid_eq_30bar_water_15[1]-x_p0_30bar_water_15
for i in range (1,params.n):
    dx_eq_30bar_water_15[i]=x_fluid_eq_30bar_water_15[i]-x_fluid_eq_30bar_water_15[i-1]

V_p0_30bar_15 = params.A_p * (x_p0_30bar_water_15-params.th_piston_15) 
V_t0_30bar_15 = params.A_p * (params.H_15-x_fluid_eq_30bar_water_15[params.n])  

v_p0 = 0 
v_fluid0=np.zeros(params.n)
y0 = np.concatenate((x_fluid_eq_30bar_water_15, [v_p0], v_fluid0))

p_0_x_30bar_water_15= np.zeros(params.n)
for i in range (params.n):
    p_0_x_30bar_water_15[i] = params.p_t0 + params.rho_water*params.g*(x_fluid_eq_30bar_water_15[params.n]-x_fluid_eq_30bar_water_15[i])


sol_30bar_water_15 = solve_ivp(ms.system_water, [0, T_30], y0, args=(params.m_piston_15,params.H_15, V_t0_30bar_15, V_p0_30bar_15, params.k_s_water, params.m_e_water, params.n, p_p0), atol = 1e-6, rtol=1e-6, max_step = 1e-6)
sol_30bar_water_15.t=np.array(sol_30bar_water_15.t)

x_p_30bar_water_15 = sol_30bar_water_15.y[0]      
x_fluid_30bar_water_15 = sol_30bar_water_15.y[1:params.n+1] 
v_fluid_30bar_water_15 = sol_30bar_water_15.y[params.n+1:2*(params.n+1)] 
a0_values_30bar_water_15 = [ms.piston_acceleration_water(y,params.m_piston_15, params.H_15, V_p0_30bar_15, params.n, params.k_s_water, p_p0) for y in sol_30bar_water_15.y.T]
a0_min_30bar_water_15 = np.min(a0_values_30bar_water_15)
a0_max_30bar_water_15 = np.max(a0_values_30bar_water_15)
v_piston_30bar_water_15 = sol_30bar_water_15.y[params.n+1, :]
v_piston_max_30bar_water_15=np.max(v_piston_30bar_water_15)

all_acc_30bar_water_15 = np.array([
    ms.all_accelerations_water(y, params.m_piston_15, params.H_15, V_t0_30bar_15, V_p0_30bar_15, params.k_s_water, params.m_e_water, params.n, p_p0)
    for y in sol_30bar_water_15.y.T
])

a_piston_30bar_water_15 = all_acc_30bar_water_15[:, 0]      
a_fluid_30bar_water_15  = all_acc_30bar_water_15[:, 1:].T   
V_p_30bar_water_15 = params.A_p * (x_p_30bar_water_15 - params.th_piston_15)
V_t_30bar_water_15 = params.A_p * (params.H_15 - x_fluid_30bar_water_15[params.n-1, :])

p_t_30bar_water_15=[]
for i in range(len(sol_30bar_water_15.t)):
    p_ti=ms.pressure_on_top(sol_30bar_water_15.y[:,i], params.H_15, params.th_piston_15, V_t0_30bar_15, params.n)
    p_t_30bar_water_15.append(p_ti)
p_t_30bar_water_15 = np.array(p_t_30bar_water_15)
p_t_30bar_water_max_15 = np.max(p_t_30bar_water_15)

p_p_30bar_water_15=[]
for i in range(len(sol_30bar_water_15.t)):
    p_pi=ms.pressure_under_piston(sol_30bar_water_15.y[:,i], params.H_15, params.th_piston_15, V_p0_30bar_15, params.n,p_p0)
    p_p_30bar_water_15.append(p_pi)
p_p_30bar_water_15 = np.array(p_p_30bar_water_15)

dx_30bar_water_15 = np.zeros((params.n, len(sol_30bar_water_15.t)))
for j in range(len(sol_30bar_water_15.t)):
    dx_30bar_water_15[0,j]=x_fluid_30bar_water_15[0,j]-x_p_30bar_water_15[j]
    for i in range (1,params.n):
        dx_30bar_water_15[i,j]=x_fluid_30bar_water_15[i,j]-x_fluid_30bar_water_15[i-1,j]
p_30bar_water_15= np.ones((params.n,len(sol_30bar_water_15.t)))*params.p_0
for i in range(params.n):
    for j in range(len(sol_30bar_water_15.t)):
        p_30bar_water_15[i,j]=params.p_0-params.rho_l_water*params.c_water**2*((dx_30bar_water_15[i,j]*params.A_p)-dx_eq_30bar_water_15[i]*params.A_p) /(dx_eq_30bar_water_15[i]*params.A_p)

#M=2#
x_fluid_eq_30bar_water_2= ms.equilibrate_fluid_positions(params.n,params.L0,params.k_s_water,params.m_e_water,params.m_piston_2,params.g,params.p_t0,p_p0)
for i in range(len(x_fluid_eq_30bar_water_2)):
    x_fluid_eq_30bar_water_2[i]=x_fluid_eq_30bar_water_2[i]+params.x_air_bottom
x_p0_30bar_water_2=x_fluid_eq_30bar_water_2[0]
offset= x_p0_30bar_water_2 - (params.x_air_bottom+params.th_piston_2)
for i in range(len(x_fluid_eq_30bar_water_2)):
    x_fluid_eq_30bar_water_2[i]=x_fluid_eq_30bar_water_2[i]-offset
x_p0_30bar_water_2=x_fluid_eq_30bar_water_2[0]

dx_eq_30bar_water_2 = np.zeros(params.n)
dx_eq_30bar_water_2[0]=x_fluid_eq_30bar_water_2[1]-x_p0_30bar_water_2
for i in range (1,params.n):
    dx_eq_30bar_water_2[i]=x_fluid_eq_30bar_water_2[i]-x_fluid_eq_30bar_water_2[i-1]

V_p0_30bar_2 = params.A_p * (x_p0_30bar_water_2-params.th_piston_2) 
V_t0_30bar_2 = params.A_p * (params.H_2-x_fluid_eq_30bar_water_2[params.n])  

v_p0 = 0 
v_fluid0=np.zeros(params.n)
y0 = np.concatenate((x_fluid_eq_30bar_water_2, [v_p0], v_fluid0))

p_0_x_30bar_water_2= np.zeros(params.n)
for i in range (params.n):
    p_0_x_30bar_water_2[i] = params.p_t0 + params.rho_water*params.g*(x_fluid_eq_30bar_water_2[params.n]-x_fluid_eq_30bar_water_2[i])


sol_30bar_water_2 = solve_ivp(ms.system_water, [0, T_30], y0, args=(params.m_piston_2,params.H_2, V_t0_30bar_2, V_p0_30bar_2, params.k_s_water, params.m_e_water, params.n, p_p0), atol = 1e-6, rtol=1e-6, max_step = 1e-6)
sol_30bar_water_2.t=np.array(sol_30bar_water_2.t)

x_p_30bar_water_2 = sol_30bar_water_2.y[0]      
x_fluid_30bar_water_2 = sol_30bar_water_2.y[1:params.n+1] 
v_fluid_30bar_water_2 = sol_30bar_water_2.y[params.n+1:2*(params.n+1)] 
a0_values_30bar_water_2 = [ms.piston_acceleration_water(y,params.m_piston_2, params.H_2, V_p0_30bar_2, params.n, params.k_s_water, p_p0) for y in sol_30bar_water_2.y.T]
a0_min_30bar_water_2 = np.min(a0_values_30bar_water_2)
a0_max_30bar_water_2 = np.max(a0_values_30bar_water_2)
v_piston_30bar_water_2 = sol_30bar_water_2.y[params.n+1, :]
v_piston_max_30bar_water_2=np.max(v_piston_30bar_water_2)

all_acc_30bar_water_2 = np.array([
    ms.all_accelerations_water(y, params.m_piston_2, params.H_2, V_t0_30bar_2, V_p0_30bar_2, params.k_s_water, params.m_e_water, params.n, p_p0)
    for y in sol_30bar_water_2.y.T
])

a_piston_30bar_water_2 = all_acc_30bar_water_2[:, 0]      
a_fluid_30bar_water_2  = all_acc_30bar_water_2[:, 1:].T   
V_p_30bar_water_2 = params.A_p * (x_p_30bar_water_2 - params.th_piston_2)
V_t_30bar_water_2 = params.A_p * (params.H_2 - x_fluid_30bar_water_2[params.n-1, :])

p_t_30bar_water_2=[]
for i in range(len(sol_30bar_water_2.t)):
    p_ti=ms.pressure_on_top(sol_30bar_water_2.y[:,i], params.H_2, params.th_piston_2, V_t0_30bar_2, params.n)
    p_t_30bar_water_2.append(p_ti)
p_t_30bar_water_2 = np.array(p_t_30bar_water_2)
p_t_30bar_water_max_2 = np.max(p_t_30bar_water_2)

p_p_30bar_water_2=[]
for i in range(len(sol_30bar_water_2.t)):
    p_pi=ms.pressure_under_piston(sol_30bar_water_2.y[:,i], params.H_2, params.th_piston_2, V_p0_30bar_2, params.n,p_p0)
    p_p_30bar_water_2.append(p_pi)
p_p_30bar_water_2 = np.array(p_p_30bar_water_2)

dx_30bar_water_2 = np.zeros((params.n, len(sol_30bar_water_2.t)))
for j in range(len(sol_30bar_water_2.t)):
    dx_30bar_water_2[0,j]=x_fluid_30bar_water_2[0,j]-x_p_30bar_water_2[j]
    for i in range (1,params.n):
        dx_30bar_water_2[i,j]=x_fluid_30bar_water_2[i,j]-x_fluid_30bar_water_2[i-1,j]
p_30bar_water_2= np.ones((params.n,len(sol_30bar_water_2.t)))*params.p_0
for i in range(params.n):
    for j in range(len(sol_30bar_water_2.t)):
        p_30bar_water_2[i,j]=params.p_0-params.rho_l_water*params.c_water**2*((dx_30bar_water_2[i,j]*params.A_p)-dx_eq_30bar_water_2[i]*params.A_p) /(dx_eq_30bar_water_2[i]*params.A_p)

#no piston#
x_fluid_eq_30bar_water_no_piston = ms.equilibrate_fluid_positions_no_piston(params.n, params.L0, params.k_s_water, params.m_e_water, params.g, params.p_t0, p_p0)
x_0_30bar_water_no_piston = x_fluid_eq_30bar_water_no_piston[0]

dx_eq_30bar_water_no_piston = np.zeros(params.n)
dx_eq_30bar_water_no_piston[0] = x_fluid_eq_30bar_water_no_piston[1] - x_0_30bar_water_no_piston
for i in range(1, params.n):
    dx_eq_30bar_water_no_piston[i] = x_fluid_eq_30bar_water_no_piston[i] - x_fluid_eq_30bar_water_no_piston[i - 1]

V_p0 = params.A_p * x_0_30bar_water_no_piston  
V_t0 = params.A_p * (params.H_no_piston - x_fluid_eq_30bar_water_no_piston[params.n]) 

v_0 = 0  
v_fluid0 = np.zeros(params.n)
y0 = np.concatenate((x_fluid_eq_30bar_water_no_piston, [v_0], v_fluid0))


sol_30bar_water_no_piston = solve_ivp(ms.system_water_no_piston, [0, T_30], y0, args=(V_t0,V_p0, p_p0), atol=1e-6, rtol=1e-6, max_step=1e-6)
sol_30bar_water_no_piston.t=np.array(sol_30bar_water_no_piston.t)

x_0_30bar_water_no_piston = sol_30bar_water_no_piston.y[0]  
x_fluid_30bar_water_no_piston = sol_30bar_water_no_piston.y[1:params.n + 1]  
v_fluid_30bar_water_no_piston = sol_30bar_water_no_piston.y[params.n + 1:2 * (params.n + 1)]  
a0_values_30bar_water_no_piston = [ms.liquid_acceleration_water(y, params.n, params.k_s_water, params.m_e_water, params.L0, V_p0, p_p0) for y in sol_30bar_water_no_piston.y.T]
a0_min_30bar_water_no_piston = np.min(a0_values_30bar_water_no_piston)
a0_max_30bar_water_no_piston = np.max(a0_values_30bar_water_no_piston)
v_piston_30bar_water_no_piston = sol_30bar_water_no_piston.y[params.n + 1, :]
v_piston_max_30bar_water_no_piston = np.max(v_piston_30bar_water_no_piston)

all_acc_30bar_water_no_piston = np.array([
    ms.all_accelerations_water_no_piston(y, V_t0, V_p0, p_p0)
    for y in sol_30bar_water_no_piston.y.T
])  

a_fluid_30bar_water_no_piston = all_acc_30bar_water_no_piston.T 
V_p_30bar_water_no_piston = params.A_p * x_0_30bar_water_no_piston
V_t_30bar_water_no_piston = params.A_p * (params.H_no_piston - x_fluid_30bar_water_no_piston[params.n-1, :])

p_t_30bar_water_no_piston = []
for i in range(len(sol_30bar_water_no_piston.t)):
    p_ti = ms.pressure_on_top_no_piston(sol_30bar_water_no_piston.y[:, i], V_t0)
    p_t_30bar_water_no_piston.append(p_ti)


p_p_30bar_water_no_piston = []
for i in range(len(sol_30bar_water_no_piston.t)):
    p_pi = ms.pressure_gas_bottom(sol_30bar_water_no_piston.y[:, i], V_p0, p_p0)
    p_p_30bar_water_no_piston.append(p_pi)


dx_30bar_water_no_piston = np.zeros((params.n, len(sol_30bar_water_no_piston.t)))
for j in range(len(sol_30bar_water_no_piston.t)):
    dx_30bar_water_no_piston[0, j] = x_fluid_30bar_water_no_piston[0, j] - x_0_30bar_water_no_piston[j]
    for i in range(1, params.n):
        dx_30bar_water_no_piston[i, j] = x_fluid_30bar_water_no_piston[i, j] - x_fluid_30bar_water_no_piston[i - 1, j]

p_30bar_water_no_piston = np.ones((params.n, len(sol_30bar_water_no_piston.t))) * params.p_0
for i in range(params.n):
    for j in range(len(sol_30bar_water_no_piston.t)):
        p_30bar_water_no_piston[i, j] = params.p_0 - params.rho_l_water * params.c_water**2 * ((dx_30bar_water_no_piston[i, j] * params.A_p) - dx_eq_30bar_water_no_piston[i] * params.A_p) / (dx_eq_30bar_water_no_piston[i] * params.A_p)

#lithium#
#M=1.69#
x_fluid_eq_30bar_lithium= ms.equilibrate_fluid_positions(params.n,params.L0,params.k_s_lithium,params.m_e_lithium,params.m_piston_ref,params.g,params.p_t0,p_p0)
for i in range(len(x_fluid_eq_30bar_lithium)):
    x_fluid_eq_30bar_lithium[i]=x_fluid_eq_30bar_lithium[i]+params.x_air_bottom
x_p0_30bar_lithium=x_fluid_eq_30bar_lithium[0]
offset= x_p0_30bar_lithium - 0.25
for i in range(len(x_fluid_eq_30bar_lithium)):
    x_fluid_eq_30bar_lithium[i]=x_fluid_eq_30bar_lithium[i]-offset
x_p0_30bar_lithium=x_fluid_eq_30bar_lithium[0]

dx_eq_30bar_lithium= np.zeros(params.n)
dx_eq_30bar_lithium[0]=x_fluid_eq_30bar_lithium[1]-x_p0_30bar_lithium
for i in range (1,params.n):
    dx_eq_30bar_lithium[i]=x_fluid_eq_30bar_lithium[i]-x_fluid_eq_30bar_lithium[i-1]

V_p0 = params.A_p * (x_p0_30bar_lithium-params.th_piston_ref) 
V_t0 = params.A_p * (params.H_ref-x_fluid_eq_30bar_lithium[params.n])

v_p0 = 0 
v_fluid0=np.zeros(params.n)
y0 = np.concatenate((x_fluid_eq_30bar_lithium, [v_p0], v_fluid0))


sol_30bar_lithium = solve_ivp(ms.system_lith, [0, T_30], y0, args=(params.m_piston_ref,params.H_ref, V_t0, V_p0, params.n, p_p0), atol = 1e-6, rtol=1e-6, max_step = 1e-6)


x_p_30bar_lithium = sol_30bar_lithium.y[0]     
x_fluid_30bar_lithium = sol_30bar_lithium.y[1:params.n+1] 
v_fluid_30bar_lithium = sol_30bar_lithium.y[params.n+1:2*(params.n+1)] 
a0_values_30bar_lithium = [ms.piston_acceleration_lith(y, params.m_piston_ref, V_p0, p_p0) for y in sol_30bar_lithium.y.T]
a0_min_30bar_lithium = np.min(a0_values_30bar_lithium)
a0_max_30bar_lithium = np.max(a0_values_30bar_lithium)
v_piston_30bar_lithium = sol_30bar_lithium.y[params.n+1, :]
v_piston_max_30bar_lithium=np.max(v_piston_30bar_lithium)

p_t_30bar_lithium=[]
for i in range(len(sol_30bar_lithium.t)):
    p_ti=ms.pressure_on_top(sol_30bar_lithium.y[:,i], params.H_ref, params.th_piston_ref, V_t0, params.n)
    p_t_30bar_lithium.append(p_ti)


p_p_30bar_lithium=[]
for i in range(len(sol_30bar_lithium.t)):
    p_pi=ms.pressure_under_piston(sol_30bar_lithium.y[:,i], params.H_ref, params.th_piston_ref, V_p0, params.n,p_p0)
    p_p_30bar_lithium.append(p_pi)


dx_30bar_lithium = np.zeros((params.n, len(sol_30bar_lithium.t)))
for j in range(len(sol_30bar_lithium.t)):
    dx_30bar_lithium[0,j]=x_fluid_30bar_lithium[0,j]-x_p_30bar_lithium[j]
    for i in range (1,params.n):
        dx_30bar_lithium[i,j]=x_fluid_30bar_lithium[i,j]-x_fluid_30bar_lithium[i-1,j]
p_30bar_lithium= np.ones((params.n,len(sol_30bar_lithium.t)))*params.p_0
for i in range(params.n):
    for j in range(len(sol_30bar_lithium.t)):
        p_30bar_lithium[i,j]=params.p_0-params.rho_l_lithium*params.c_lithium**2*((dx_30bar_lithium[i,j]*params.A_p)-dx_eq_30bar_lithium[i]*params.A_p) /(dx_eq_30bar_lithium[i]*params.A_p)

#Driving pressure: 4MPa#
p_p0 = 40e5
T_40 = 0.02  

#water#
#M=0.1#
x_fluid_eq_40bar_water_01= ms.equilibrate_fluid_positions(params.n,params.L0,params.k_s_water,params.m_e_water,params.m_piston_01,params.g,params.p_t0,p_p0)
for i in range(len(x_fluid_eq_40bar_water_01)):
    x_fluid_eq_40bar_water_01[i]=x_fluid_eq_40bar_water_01[i]+params.x_air_bottom
x_p0_40bar_water_01=x_fluid_eq_40bar_water_01[0]
offset= x_p0_40bar_water_01 - (params.x_air_bottom+params.th_piston_01)
for i in range(len(x_fluid_eq_40bar_water_01)):
    x_fluid_eq_40bar_water_01[i]=x_fluid_eq_40bar_water_01[i]-offset
x_p0_40bar_water_01=x_fluid_eq_40bar_water_01[0]

dx_eq_40bar_water_01 = np.zeros(params.n)
dx_eq_40bar_water_01[0]=x_fluid_eq_40bar_water_01[1]-x_p0_40bar_water_01
for i in range (1,params.n):
    dx_eq_40bar_water_01[i]=x_fluid_eq_40bar_water_01[i]-x_fluid_eq_40bar_water_01[i-1]

V_p0_40bar_01 = params.A_p * (x_p0_40bar_water_01-params.th_piston_01) 
V_t0_40bar_01 = params.A_p * (params.H_01-x_fluid_eq_40bar_water_01[params.n])  

v_p0 = 0 
v_fluid0=np.zeros(params.n)
y0 = np.concatenate((x_fluid_eq_40bar_water_01, [v_p0], v_fluid0))

p_0_x_40bar_water_01= np.zeros(params.n)
for i in range (params.n):
    p_0_x_40bar_water_01[i] = params.p_t0 + params.rho_water*params.g*(x_fluid_eq_40bar_water_01[params.n]-x_fluid_eq_40bar_water_01[i])

sol_40bar_water_01 = solve_ivp(ms.system_water, [0, T_40], y0, args=(params.m_piston_01,params.H_01, V_t0_40bar_01, V_p0_40bar_01, params.k_s_water, params.m_e_water, params.n, p_p0), atol = 1e-6, rtol=1e-6, max_step = 1e-6)
sol_40bar_water_01.t=np.array(sol_40bar_water_01.t)

x_p_40bar_water_01 = sol_40bar_water_01.y[0]      
x_fluid_40bar_water_01 = sol_40bar_water_01.y[1:params.n+1] 
v_fluid_40bar_water_01 = sol_40bar_water_01.y[params.n+1:2*(params.n+1)] 
a0_values_40bar_water_01 = [ms.piston_acceleration_water(y,params.m_piston_01, params.H_01, V_p0_40bar_01, params.n, params.k_s_water, p_p0) for y in sol_40bar_water_01.y.T]
a0_min_40bar_water_01 = np.min(a0_values_40bar_water_01)
a0_max_40bar_water_01 = np.max(a0_values_40bar_water_01)
v_piston_40bar_water_01 = sol_40bar_water_01.y[params.n+1, :]
v_piston_max_40bar_water_01=np.max(v_piston_40bar_water_01)

all_acc_40bar_water_01 = np.array([
    ms.all_accelerations_water(y, params.m_piston_01, params.H_01, V_t0_40bar_01, V_p0_40bar_01, params.k_s_water, params.m_e_water, params.n, p_p0)
    for y in sol_40bar_water_01.y.T
])

a_piston_40bar_water_01 = all_acc_40bar_water_01[:, 0]      
a_fluid_40bar_water_01  = all_acc_40bar_water_01[:, 1:].T   
V_p_40bar_water_01 = params.A_p * (x_p_40bar_water_01 - params.th_piston_01)
V_t_40bar_water_01 = params.A_p * (params.H_01 - x_fluid_40bar_water_01[params.n-1, :])

p_t_40bar_water_01=[]
for i in range(len(sol_40bar_water_01.t)):
    p_ti=ms.pressure_on_top(sol_40bar_water_01.y[:,i], params.H_01, params.th_piston_01, V_t0_40bar_01, params.n)
    p_t_40bar_water_01.append(p_ti)
p_t_40bar_water_01 = np.array(p_t_40bar_water_01)
p_t_40bar_water_max_01 = np.max(p_t_40bar_water_01)

p_p_40bar_water_01=[]
for i in range(len(sol_40bar_water_01.t)):
    p_pi=ms.pressure_under_piston(sol_40bar_water_01.y[:,i], params.H_01, params.th_piston_01, V_p0_40bar_01, params.n,p_p0)
    p_p_40bar_water_01.append(p_pi)
p_p_40bar_water_01 = np.array(p_p_40bar_water_01)

dx_40bar_water_01 = np.zeros((params.n, len(sol_40bar_water_01.t)))
for j in range(len(sol_40bar_water_01.t)):
    dx_40bar_water_01[0,j]=x_fluid_40bar_water_01[0,j]-x_p_40bar_water_01[j]
    for i in range (1,params.n):
        dx_40bar_water_01[i,j]=x_fluid_40bar_water_01[i,j]-x_fluid_40bar_water_01[i-1,j]
p_40bar_water_01= np.ones((params.n,len(sol_40bar_water_01.t)))*params.p_0
for i in range(params.n):
    for j in range(len(sol_40bar_water_01.t)):
        p_40bar_water_01[i,j]=params.p_0-params.rho_l_water*params.c_water**2*((dx_40bar_water_01[i,j]*params.A_p)-dx_eq_40bar_water_01[i]*params.A_p) /(dx_eq_40bar_water_01[i]*params.A_p)

#M=0.5#
x_fluid_eq_40bar_water_05= ms.equilibrate_fluid_positions(params.n,params.L0,params.k_s_water,params.m_e_water,params.m_piston_05,params.g,params.p_t0,p_p0)
for i in range(len(x_fluid_eq_40bar_water_05)):
    x_fluid_eq_40bar_water_05[i]=x_fluid_eq_40bar_water_05[i]+params.x_air_bottom
x_p0_40bar_water_05=x_fluid_eq_40bar_water_05[0]
offset= x_p0_40bar_water_05 - (params.x_air_bottom+params.th_piston_05)
for i in range(len(x_fluid_eq_40bar_water_05)):
    x_fluid_eq_40bar_water_05[i]=x_fluid_eq_40bar_water_05[i]-offset
x_p0_40bar_water_05=x_fluid_eq_40bar_water_05[0]

dx_eq_40bar_water_05 = np.zeros(params.n)
dx_eq_40bar_water_05[0]=x_fluid_eq_40bar_water_05[1]-x_p0_40bar_water_05
for i in range (1,params.n):
    dx_eq_40bar_water_05[i]=x_fluid_eq_40bar_water_05[i]-x_fluid_eq_40bar_water_05[i-1]

V_p0_40bar_05 = params.A_p * (x_p0_40bar_water_05-params.th_piston_05) 
V_t0_40bar_05 = params.A_p * (params.H_05-x_fluid_eq_40bar_water_05[params.n])  

v_p0 = 0
v_fluid0=np.zeros(params.n)
y0 = np.concatenate((x_fluid_eq_40bar_water_05, [v_p0], v_fluid0))

p_0_x_40bar_water_05= np.zeros(params.n)
for i in range (params.n):
    p_0_x_40bar_water_05[i] = params.p_t0 + params.rho_water*params.g*(x_fluid_eq_40bar_water_05[params.n]-x_fluid_eq_40bar_water_05[i])


sol_40bar_water_05 = solve_ivp(ms.system_water, [0, T_40], y0, args=(params.m_piston_05,params.H_05, V_t0_40bar_05, V_p0_40bar_05, params.k_s_water, params.m_e_water, params.n, p_p0), atol = 1e-6, rtol=1e-6, max_step = 1e-6)
sol_40bar_water_05.t=np.array(sol_40bar_water_05.t)

x_p_40bar_water_05 = sol_40bar_water_05.y[0]      
x_fluid_40bar_water_05 = sol_40bar_water_05.y[1:params.n+1] 
v_fluid_40bar_water_05 = sol_40bar_water_05.y[params.n+1:2*(params.n+1)] 
a0_values_40bar_water_05 = [ms.piston_acceleration_water(y,params.m_piston_05, params.H_05, V_p0_40bar_05, params.n, params.k_s_water, p_p0) for y in sol_40bar_water_05.y.T]
a0_min_40bar_water_05 = np.min(a0_values_40bar_water_05)
a0_max_40bar_water_05 = np.max(a0_values_40bar_water_05)
v_piston_40bar_water_05 = sol_40bar_water_05.y[params.n+1, :]
v_piston_max_40bar_water_05=np.max(v_piston_40bar_water_05)

all_acc_40bar_water_05 = np.array([
    ms.all_accelerations_water(y, params.m_piston_05, params.H_05, V_t0_40bar_05, V_p0_40bar_05, params.k_s_water, params.m_e_water, params.n, p_p0)
    for y in sol_40bar_water_05.y.T
])

a_piston_40bar_water_05 = all_acc_40bar_water_05[:, 0]     
a_fluid_40bar_water_05  = all_acc_40bar_water_05[:, 1:].T   
V_p_40bar_water_05 = params.A_p * (x_p_40bar_water_05 - params.th_piston_05)
V_t_40bar_water_05 = params.A_p * (params.H_05 - x_fluid_40bar_water_05[params.n-1, :])

p_t_40bar_water_05=[]
for i in range(len(sol_40bar_water_05.t)):
    p_ti=ms.pressure_on_top(sol_40bar_water_05.y[:,i], params.H_05, params.th_piston_05, V_t0_40bar_05, params.n)
    p_t_40bar_water_05.append(p_ti)
p_t_40bar_water_05 = np.array(p_t_40bar_water_05)
p_t_40bar_water_max_05 = np.max(p_t_40bar_water_05)

p_p_40bar_water_05=[]
for i in range(len(sol_40bar_water_05.t)):
    p_pi=ms.pressure_under_piston(sol_40bar_water_05.y[:,i], params.H_05, params.th_piston_05, V_p0_40bar_05, params.n,p_p0)
    p_p_40bar_water_05.append(p_pi)
p_p_40bar_water_05 = np.array(p_p_40bar_water_05)

dx_40bar_water_05 = np.zeros((params.n, len(sol_40bar_water_05.t)))
for j in range(len(sol_40bar_water_05.t)):
    dx_40bar_water_05[0,j]=x_fluid_40bar_water_05[0,j]-x_p_40bar_water_05[j]
    for i in range (1,params.n):
        dx_40bar_water_05[i,j]=x_fluid_40bar_water_05[i,j]-x_fluid_40bar_water_05[i-1,j]
p_40bar_water_05= np.ones((params.n,len(sol_40bar_water_05.t)))*params.p_0
for i in range(params.n):
    for j in range(len(sol_40bar_water_05.t)):
        p_40bar_water_05[i,j]=params.p_0-params.rho_l_water*params.c_water**2*((dx_40bar_water_05[i,j]*params.A_p)-dx_eq_40bar_water_05[i]*params.A_p) /(dx_eq_40bar_water_05[i]*params.A_p)

#M=0.9#
x_fluid_eq_40bar_water_09= ms.equilibrate_fluid_positions(params.n,params.L0,params.k_s_water,params.m_e_water,params.m_piston_09,params.g,params.p_t0,p_p0)
for i in range(len(x_fluid_eq_40bar_water_09)):
    x_fluid_eq_40bar_water_09[i]=x_fluid_eq_40bar_water_09[i]+params.x_air_bottom
x_p0_40bar_water_09=x_fluid_eq_40bar_water_09[0]
offset= x_p0_40bar_water_09 - (params.x_air_bottom+params.th_piston_09)
for i in range(len(x_fluid_eq_40bar_water_09)):
    x_fluid_eq_40bar_water_09[i]=x_fluid_eq_40bar_water_09[i]-offset
x_p0_40bar_water_09=x_fluid_eq_40bar_water_09[0]

dx_eq_40bar_water_09 = np.zeros(params.n)
dx_eq_40bar_water_09[0]=x_fluid_eq_40bar_water_09[1]-x_p0_40bar_water_09
for i in range (1,params.n):
    dx_eq_40bar_water_09[i]=x_fluid_eq_40bar_water_09[i]-x_fluid_eq_40bar_water_09[i-1]

V_p0_40bar_09 = params.A_p * (x_p0_40bar_water_09-params.th_piston_09) 
V_t0_40bar_09 = params.A_p * (params.H_09-x_fluid_eq_40bar_water_09[params.n]) 

v_p0 = 0 
v_fluid0=np.zeros(params.n)
y0 = np.concatenate((x_fluid_eq_40bar_water_09, [v_p0], v_fluid0))

p_0_x_40bar_water_09= np.zeros(params.n)
for i in range (params.n):
    p_0_x_40bar_water_09[i] = params.p_t0 + params.rho_water*params.g*(x_fluid_eq_40bar_water_09[params.n]-x_fluid_eq_40bar_water_09[i])


sol_40bar_water_09 = solve_ivp(ms.system_water, [0, T_40], y0, args=(params.m_piston_09,params.H_09, V_t0_40bar_09, V_p0_40bar_09, params.k_s_water, params.m_e_water, params.n, p_p0), atol = 1e-6, rtol=1e-6, max_step = 1e-6)
sol_40bar_water_09.t=np.array(sol_40bar_water_09.t)

x_p_40bar_water_09 = sol_40bar_water_09.y[0]      
x_fluid_40bar_water_09 = sol_40bar_water_09.y[1:params.n+1] 
v_fluid_40bar_water_09 = sol_40bar_water_09.y[params.n+1:2*(params.n+1)] 
a0_values_40bar_water_09 = [ms.piston_acceleration_water(y,params.m_piston_09, params.H_09, V_p0_40bar_09, params.n, params.k_s_water, p_p0) for y in sol_40bar_water_09.y.T]
a0_min_40bar_water_09 = np.min(a0_values_40bar_water_09)
a0_max_40bar_water_09 = np.max(a0_values_40bar_water_09)
v_piston_40bar_water_09 = sol_40bar_water_09.y[params.n+1, :]
v_piston_max_40bar_water_09=np.max(v_piston_40bar_water_09)

all_acc_40bar_water_09 = np.array([
    ms.all_accelerations_water(y, params.m_piston_09, params.H_09, V_t0_40bar_09, V_p0_40bar_09, params.k_s_water, params.m_e_water, params.n, p_p0)
    for y in sol_40bar_water_09.y.T
])

a_piston_40bar_water_09 = all_acc_40bar_water_09[:, 0]      
a_fluid_40bar_water_09  = all_acc_40bar_water_09[:, 1:].T   
V_p_40bar_water_09 = params.A_p * (x_p_40bar_water_09 - params.th_piston_09)
V_t_40bar_water_09 = params.A_p * (params.H_09 - x_fluid_40bar_water_09[params.n-1, :])

p_t_40bar_water_09=[]
for i in range(len(sol_40bar_water_09.t)):
    p_ti=ms.pressure_on_top(sol_40bar_water_09.y[:,i], params.H_09, params.th_piston_09, V_t0_40bar_09, params.n)
    p_t_40bar_water_09.append(p_ti)
p_t_40bar_water_09 = np.array(p_t_40bar_water_09)
p_t_40bar_water_max_09 = np.max(p_t_40bar_water_09)

p_p_40bar_water_09=[]
for i in range(len(sol_40bar_water_09.t)):
    p_pi=ms.pressure_under_piston(sol_40bar_water_09.y[:,i], params.H_09, params.th_piston_09, V_p0_40bar_09, params.n,p_p0)
    p_p_40bar_water_09.append(p_pi)
p_p_40bar_water_09 = np.array(p_p_40bar_water_09)

dx_40bar_water_09 = np.zeros((params.n, len(sol_40bar_water_09.t)))
for j in range(len(sol_40bar_water_09.t)):
    dx_40bar_water_09[0,j]=x_fluid_40bar_water_09[0,j]-x_p_40bar_water_09[j]
    for i in range (1,params.n):
        dx_40bar_water_09[i,j]=x_fluid_40bar_water_09[i,j]-x_fluid_40bar_water_09[i-1,j]
p_40bar_water_09= np.ones((params.n,len(sol_40bar_water_09.t)))*params.p_0
for i in range(params.n):
    for j in range(len(sol_40bar_water_09.t)):
        p_40bar_water_09[i,j]=params.p_0-params.rho_l_water*params.c_water**2*((dx_40bar_water_09[i,j]*params.A_p)-dx_eq_40bar_water_09[i]*params.A_p) /(dx_eq_40bar_water_09[i]*params.A_p)

#M=1#
x_fluid_eq_40bar_water_1= ms.equilibrate_fluid_positions(params.n,params.L0,params.k_s_water,params.m_e_water,params.m_piston_1,params.g,params.p_t0,p_p0)
for i in range(len(x_fluid_eq_40bar_water_1)):
    x_fluid_eq_40bar_water_1[i]=x_fluid_eq_40bar_water_1[i]+params.x_air_bottom
x_p0_40bar_water_1=x_fluid_eq_40bar_water_1[0]
offset= x_p0_40bar_water_1 - (params.x_air_bottom+params.th_piston_1)
for i in range(len(x_fluid_eq_40bar_water_1)):
    x_fluid_eq_40bar_water_1[i]=x_fluid_eq_40bar_water_1[i]-offset
x_p0_40bar_water_1=x_fluid_eq_40bar_water_1[0]

dx_eq_40bar_water_1 = np.zeros(params.n)
dx_eq_40bar_water_1[0]=x_fluid_eq_40bar_water_1[1]-x_p0_40bar_water_1
for i in range (1,params.n):
    dx_eq_40bar_water_1[i]=x_fluid_eq_40bar_water_1[i]-x_fluid_eq_40bar_water_1[i-1]

V_p0_40bar_1 = params.A_p * (x_p0_40bar_water_1-params.th_piston_1) 
V_t0_40bar_1 = params.A_p * (params.H_1-x_fluid_eq_40bar_water_1[params.n])  

v_p0 = 0 
v_fluid0=np.zeros(params.n)
y0 = np.concatenate((x_fluid_eq_40bar_water_1, [v_p0], v_fluid0))

p_0_x_40bar_water_1= np.zeros(params.n)
for i in range (params.n):
    p_0_x_40bar_water_1[i] = params.p_t0 + params.rho_water*params.g*(x_fluid_eq_40bar_water_1[params.n]-x_fluid_eq_40bar_water_1[i])


sol_40bar_water_1 = solve_ivp(ms.system_water, [0, T_40], y0, args=(params.m_piston_1,params.H_1, V_t0_40bar_1, V_p0_40bar_1, params.k_s_water, params.m_e_water, params.n, p_p0), atol = 1e-6, rtol=1e-6, max_step = 1e-6)
sol_40bar_water_1.t=np.array(sol_40bar_water_1.t)

x_p_40bar_water_1 = sol_40bar_water_1.y[0]      
x_fluid_40bar_water_1 = sol_40bar_water_1.y[1:params.n+1] 
v_fluid_40bar_water_1 = sol_40bar_water_1.y[params.n+1:2*(params.n+1)] 
a0_values_40bar_water_1 = [ms.piston_acceleration_water(y,params.m_piston_1, params.H_1, V_p0_40bar_1, params.n, params.k_s_water, p_p0) for y in sol_40bar_water_1.y.T]
a0_min_40bar_water_1 = np.min(a0_values_40bar_water_1)
a0_max_40bar_water_1 = np.max(a0_values_40bar_water_1)
v_piston_40bar_water_1 = sol_40bar_water_1.y[params.n+1, :]
v_piston_max_40bar_water_1=np.max(v_piston_40bar_water_1)

all_acc_40bar_water_1 = np.array([
    ms.all_accelerations_water(y, params.m_piston_1, params.H_1, V_t0_40bar_1, V_p0_40bar_1, params.k_s_water, params.m_e_water, params.n, p_p0)
    for y in sol_40bar_water_1.y.T
])

a_piston_40bar_water_1 = all_acc_40bar_water_1[:, 0]      
a_fluid_40bar_water_1  = all_acc_40bar_water_1[:, 1:].T   
V_p_40bar_water_1 = params.A_p * (x_p_40bar_water_1 - params.th_piston_1)
V_t_40bar_water_1 = params.A_p * (params.H_1 - x_fluid_40bar_water_1[params.n-1, :])

p_t_40bar_water_1=[]
for i in range(len(sol_40bar_water_1.t)):
    p_ti=ms.pressure_on_top(sol_40bar_water_1.y[:,i], params.H_1, params.th_piston_1, V_t0_40bar_1, params.n)
    p_t_40bar_water_1.append(p_ti)
p_t_40bar_water_1 = np.array(p_t_40bar_water_1)
p_t_40bar_water_max_1 = np.max(p_t_40bar_water_1)

p_p_40bar_water_1=[]
for i in range(len(sol_40bar_water_1.t)):
    p_pi=ms.pressure_under_piston(sol_40bar_water_1.y[:,i], params.H_1, params.th_piston_1, V_p0_40bar_1, params.n,p_p0)
    p_p_40bar_water_1.append(p_pi)
p_p_40bar_water_1 = np.array(p_p_40bar_water_1)

dx_40bar_water_1 = np.zeros((params.n, len(sol_40bar_water_1.t)))
for j in range(len(sol_40bar_water_1.t)):
    dx_40bar_water_1[0,j]=x_fluid_40bar_water_1[0,j]-x_p_40bar_water_1[j]
    for i in range (1,params.n):
        dx_40bar_water_1[i,j]=x_fluid_40bar_water_1[i,j]-x_fluid_40bar_water_1[i-1,j]
p_40bar_water_1= np.ones((params.n,len(sol_40bar_water_1.t)))*params.p_0
for i in range(params.n):
    for j in range(len(sol_40bar_water_1.t)):
        p_40bar_water_1[i,j]=params.p_0-params.rho_l_water*params.c_water**2*((dx_40bar_water_1[i,j]*params.A_p)-dx_eq_40bar_water_1[i]*params.A_p) /(dx_eq_40bar_water_1[i]*params.A_p)

#M=1.5#
x_fluid_eq_40bar_water_15= ms.equilibrate_fluid_positions(params.n,params.L0,params.k_s_water,params.m_e_water,params.m_piston_15,params.g,params.p_t0,p_p0)
for i in range(len(x_fluid_eq_40bar_water_15)):
    x_fluid_eq_40bar_water_15[i]=x_fluid_eq_40bar_water_15[i]+params.x_air_bottom
x_p0_40bar_water_15=x_fluid_eq_40bar_water_15[0]
offset= x_p0_40bar_water_15 - (params.x_air_bottom+params.th_piston_15)
for i in range(len(x_fluid_eq_40bar_water_15)):
    x_fluid_eq_40bar_water_15[i]=x_fluid_eq_40bar_water_15[i]-offset
x_p0_40bar_water_15=x_fluid_eq_40bar_water_15[0]

dx_eq_40bar_water_15 = np.zeros(params.n)
dx_eq_40bar_water_15[0]=x_fluid_eq_40bar_water_15[1]-x_p0_40bar_water_15
for i in range (1,params.n):
    dx_eq_40bar_water_15[i]=x_fluid_eq_40bar_water_15[i]-x_fluid_eq_40bar_water_15[i-1]

V_p0_40bar_15 = params.A_p * (x_p0_40bar_water_15-params.th_piston_15)
V_t0_40bar_15 = params.A_p * (params.H_15-x_fluid_eq_40bar_water_15[params.n])  

v_p0 = 0 
v_fluid0=np.zeros(params.n)
y0 = np.concatenate((x_fluid_eq_40bar_water_15, [v_p0], v_fluid0))

p_0_x_40bar_water_15= np.zeros(params.n)
for i in range (params.n):
    p_0_x_40bar_water_15[i] = params.p_t0 + params.rho_water*params.g*(x_fluid_eq_40bar_water_15[params.n]-x_fluid_eq_40bar_water_15[i])


sol_40bar_water_15 = solve_ivp(ms.system_water, [0, T_40], y0, args=(params.m_piston_15,params.H_15, V_t0_40bar_15, V_p0_40bar_15, params.k_s_water, params.m_e_water, params.n, p_p0), atol = 1e-6, rtol=1e-6, max_step = 1e-6)
sol_40bar_water_15.t=np.array(sol_40bar_water_15.t)

x_p_40bar_water_15 = sol_40bar_water_15.y[0]      
x_fluid_40bar_water_15 = sol_40bar_water_15.y[1:params.n+1] 
v_fluid_40bar_water_15 = sol_40bar_water_15.y[params.n+1:2*(params.n+1)]
a0_values_40bar_water_15 = [ms.piston_acceleration_water(y,params.m_piston_15, params.H_15, V_p0_40bar_15, params.n, params.k_s_water, p_p0) for y in sol_40bar_water_15.y.T]
a0_min_40bar_water_15 = np.min(a0_values_40bar_water_15)
a0_max_40bar_water_15 = np.max(a0_values_40bar_water_15)
v_piston_40bar_water_15 = sol_40bar_water_15.y[params.n+1, :]
v_piston_max_40bar_water_15=np.max(v_piston_40bar_water_15)

all_acc_40bar_water_15 = np.array([
    ms.all_accelerations_water(y, params.m_piston_15, params.H_15, V_t0_40bar_15, V_p0_40bar_15, params.k_s_water, params.m_e_water, params.n, p_p0)
    for y in sol_40bar_water_15.y.T
])

a_piston_40bar_water_15 = all_acc_40bar_water_15[:, 0]      
a_fluid_40bar_water_15  = all_acc_40bar_water_15[:, 1:].T   
V_p_40bar_water_15 = params.A_p * (x_p_40bar_water_15 - params.th_piston_15)
V_t_40bar_water_15 = params.A_p * (params.H_15 - x_fluid_40bar_water_15[params.n-1, :])

p_t_40bar_water_15=[]
for i in range(len(sol_40bar_water_15.t)):
    p_ti=ms.pressure_on_top(sol_40bar_water_15.y[:,i], params.H_15, params.th_piston_15, V_t0_40bar_15, params.n)
    p_t_40bar_water_15.append(p_ti)
p_t_40bar_water_15 = np.array(p_t_40bar_water_15)
p_t_40bar_water_max_15 = np.max(p_t_40bar_water_15)

p_p_40bar_water_15=[]
for i in range(len(sol_40bar_water_15.t)):
    p_pi=ms.pressure_under_piston(sol_40bar_water_15.y[:,i], params.H_15, params.th_piston_15, V_p0_40bar_15, params.n,p_p0)
    p_p_40bar_water_15.append(p_pi)
p_p_40bar_water_15 = np.array(p_p_40bar_water_15)

dx_40bar_water_15 = np.zeros((params.n, len(sol_40bar_water_15.t)))
for j in range(len(sol_40bar_water_15.t)):
    dx_40bar_water_15[0,j]=x_fluid_40bar_water_15[0,j]-x_p_40bar_water_15[j]
    for i in range (1,params.n):
        dx_40bar_water_15[i,j]=x_fluid_40bar_water_15[i,j]-x_fluid_40bar_water_15[i-1,j]
p_40bar_water_15= np.ones((params.n,len(sol_40bar_water_15.t)))*params.p_0
for i in range(params.n):
    for j in range(len(sol_40bar_water_15.t)):
        p_40bar_water_15[i,j]=params.p_0-params.rho_l_water*params.c_water**2*((dx_40bar_water_15[i,j]*params.A_p)-dx_eq_40bar_water_15[i]*params.A_p) /(dx_eq_40bar_water_15[i]*params.A_p)

#M=2#
x_fluid_eq_40bar_water_2= ms.equilibrate_fluid_positions(params.n,params.L0,params.k_s_water,params.m_e_water,params.m_piston_2,params.g,params.p_t0,p_p0)
for i in range(len(x_fluid_eq_40bar_water_2)):
    x_fluid_eq_40bar_water_2[i]=x_fluid_eq_40bar_water_2[i]+params.x_air_bottom
x_p0_40bar_water_2=x_fluid_eq_40bar_water_2[0]
offset= x_p0_40bar_water_2 - (params.x_air_bottom+params.th_piston_2)
for i in range(len(x_fluid_eq_40bar_water_2)):
    x_fluid_eq_40bar_water_2[i]=x_fluid_eq_40bar_water_2[i]-offset
x_p0_40bar_water_2=x_fluid_eq_40bar_water_2[0]

dx_eq_40bar_water_2 = np.zeros(params.n)
dx_eq_40bar_water_2[0]=x_fluid_eq_40bar_water_2[1]-x_p0_40bar_water_2
for i in range (1,params.n):
    dx_eq_40bar_water_2[i]=x_fluid_eq_40bar_water_2[i]-x_fluid_eq_40bar_water_2[i-1]

V_p0_40bar_2 = params.A_p * (x_p0_40bar_water_2-params.th_piston_2) 
V_t0_40bar_2 = params.A_p * (params.H_2-x_fluid_eq_40bar_water_2[params.n]) 

v_p0 = 0 
v_fluid0=np.zeros(params.n)
y0 = np.concatenate((x_fluid_eq_40bar_water_2, [v_p0], v_fluid0))

p_0_x_40bar_water_2= np.zeros(params.n)
for i in range (params.n):
    p_0_x_40bar_water_2[i] = params.p_t0 + params.rho_water*params.g*(x_fluid_eq_40bar_water_2[params.n]-x_fluid_eq_40bar_water_2[i])


sol_40bar_water_2 = solve_ivp(ms.system_water, [0, T_40], y0, args=(params.m_piston_2,params.H_2, V_t0_40bar_2, V_p0_40bar_2, params.k_s_water, params.m_e_water, params.n, p_p0), atol = 1e-6, rtol=1e-6, max_step = 1e-6)
sol_40bar_water_2.t=np.array(sol_40bar_water_2.t)

x_p_40bar_water_2 = sol_40bar_water_2.y[0]      
x_fluid_40bar_water_2 = sol_40bar_water_2.y[1:params.n+1] 
v_fluid_40bar_water_2 = sol_40bar_water_2.y[params.n+1:2*(params.n+1)] 
a0_values_40bar_water_2 = [ms.piston_acceleration_water(y,params.m_piston_2, params.H_2, V_p0_40bar_2, params.n, params.k_s_water, p_p0) for y in sol_40bar_water_2.y.T]
a0_min_40bar_water_2 = np.min(a0_values_40bar_water_2)
a0_max_40bar_water_2 = np.max(a0_values_40bar_water_2)
v_piston_40bar_water_2 = sol_40bar_water_2.y[params.n+1, :]
v_piston_max_40bar_water_2=np.max(v_piston_40bar_water_2)

all_acc_40bar_water_2 = np.array([
    ms.all_accelerations_water(y, params.m_piston_2, params.H_2, V_t0_40bar_2, V_p0_40bar_2, params.k_s_water, params.m_e_water, params.n, p_p0)
    for y in sol_40bar_water_2.y.T
])

a_piston_40bar_water_2 = all_acc_40bar_water_2[:, 0]      
a_fluid_40bar_water_2  = all_acc_40bar_water_2[:, 1:].T   
V_p_40bar_water_2 = params.A_p * (x_p_40bar_water_2 - params.th_piston_2)
V_t_40bar_water_2 = params.A_p * (params.H_2 - x_fluid_40bar_water_2[params.n-1, :])

p_t_40bar_water_2=[]
for i in range(len(sol_40bar_water_2.t)):
    p_ti=ms.pressure_on_top(sol_40bar_water_2.y[:,i], params.H_2, params.th_piston_2, V_t0_40bar_2, params.n)
    p_t_40bar_water_2.append(p_ti)
p_t_40bar_water_2 = np.array(p_t_40bar_water_2)
p_t_40bar_water_max_2 = np.max(p_t_40bar_water_2)

p_p_40bar_water_2=[]
for i in range(len(sol_40bar_water_2.t)):
    p_pi=ms.pressure_under_piston(sol_40bar_water_2.y[:,i], params.H_2, params.th_piston_2, V_p0_40bar_2, params.n,p_p0)
    p_p_40bar_water_2.append(p_pi)
p_p_40bar_water_2 = np.array(p_p_40bar_water_2)

dx_40bar_water_2 = np.zeros((params.n, len(sol_40bar_water_2.t)))
for j in range(len(sol_40bar_water_2.t)):
    dx_40bar_water_2[0,j]=x_fluid_40bar_water_2[0,j]-x_p_40bar_water_2[j]
    for i in range (1,params.n):
        dx_40bar_water_2[i,j]=x_fluid_40bar_water_2[i,j]-x_fluid_40bar_water_2[i-1,j]
p_40bar_water_2= np.ones((params.n,len(sol_40bar_water_2.t)))*params.p_0
for i in range(params.n):
    for j in range(len(sol_40bar_water_2.t)):
        p_40bar_water_2[i,j]=params.p_0-params.rho_l_water*params.c_water**2*((dx_40bar_water_2[i,j]*params.A_p)-dx_eq_40bar_water_2[i]*params.A_p) /(dx_eq_40bar_water_2[i]*params.A_p)

#no piston#
x_fluid_eq_40bar_water_no_piston = ms.equilibrate_fluid_positions_no_piston(params.n, params.L0, params.k_s_water, params.m_e_water, params.g, params.p_t0, p_p0)
x_0_40bar_water_no_piston = x_fluid_eq_40bar_water_no_piston[0]

dx_eq_40bar_water_no_piston = np.zeros(params.n)
dx_eq_40bar_water_no_piston[0] = x_fluid_eq_40bar_water_no_piston[1] - x_0_40bar_water_no_piston
for i in range(1, params.n):
    dx_eq_40bar_water_no_piston[i] = x_fluid_eq_40bar_water_no_piston[i] - x_fluid_eq_40bar_water_no_piston[i - 1]

V_p0 = params.A_p * x_0_40bar_water_no_piston  
V_t0 = params.A_p * (params.H_no_piston - x_fluid_eq_40bar_water_no_piston[params.n]) 

v_0 = 0  
v_fluid0 = np.zeros(params.n)
y0 = np.concatenate((x_fluid_eq_40bar_water_no_piston, [v_0], v_fluid0))


sol_40bar_water_no_piston = solve_ivp(ms.system_water_no_piston, [0, T_40], y0, args=(V_t0,V_p0, p_p0), atol=1e-6, rtol=1e-6, max_step=1e-6)
sol_40bar_water_no_piston.t=np.array(sol_40bar_water_no_piston.t)

x_0_40bar_water_no_piston = sol_40bar_water_no_piston.y[0]  
x_fluid_40bar_water_no_piston = sol_40bar_water_no_piston.y[1:params.n + 1]  
v_fluid_40bar_water_no_piston = sol_40bar_water_no_piston.y[params.n + 1:2 * (params.n + 1)]  
a0_values_40bar_water_no_piston = [ms.liquid_acceleration_water(y, params.n, params.k_s_water, params.m_e_water, params.L0, V_p0, p_p0) for y in sol_40bar_water_no_piston.y.T]
a0_min_40bar_water_no_piston = np.min(a0_values_40bar_water_no_piston)
a0_max_40bar_water_no_piston = np.max(a0_values_40bar_water_no_piston)
v_piston_40bar_water_no_piston = sol_40bar_water_no_piston.y[params.n + 1, :]
v_piston_max_40bar_water_no_piston = np.max(v_piston_40bar_water_no_piston)

all_acc_40bar_water_no_piston = np.array([
    ms.all_accelerations_water_no_piston(y, V_t0, V_p0, p_p0)
    for y in sol_40bar_water_no_piston.y.T
])  

a_fluid_40bar_water_no_piston = all_acc_40bar_water_no_piston.T  
V_p_40bar_water_no_piston = params.A_p * x_0_40bar_water_no_piston
V_t_40bar_water_no_piston = params.A_p * (params.H_no_piston - x_fluid_40bar_water_no_piston[params.n-1, :])

p_t_40bar_water_no_piston = []
for i in range(len(sol_40bar_water_no_piston.t)):
    p_ti = ms.pressure_on_top_no_piston(sol_40bar_water_no_piston.y[:, i], V_t0)
    p_t_40bar_water_no_piston.append(p_ti)


p_p_40bar_water_no_piston = []
for i in range(len(sol_40bar_water_no_piston.t)):
    p_pi = ms.pressure_gas_bottom(sol_40bar_water_no_piston.y[:, i], V_p0, p_p0)
    p_p_40bar_water_no_piston.append(p_pi)


dx_40bar_water_no_piston = np.zeros((params.n, len(sol_40bar_water_no_piston.t)))
for j in range(len(sol_40bar_water_no_piston.t)):
    dx_40bar_water_no_piston[0, j] = x_fluid_40bar_water_no_piston[0, j] - x_0_40bar_water_no_piston[j]
    for i in range(1, params.n):
        dx_40bar_water_no_piston[i, j] = x_fluid_40bar_water_no_piston[i, j] - x_fluid_40bar_water_no_piston[i - 1, j]

p_40bar_water_no_piston = np.ones((params.n, len(sol_40bar_water_no_piston.t))) * params.p_0
for i in range(params.n):
    for j in range(len(sol_40bar_water_no_piston.t)):
        p_40bar_water_no_piston[i, j] = params.p_0 - params.rho_l_water * params.c_water**2 * ((dx_40bar_water_no_piston[i, j] * params.A_p) - dx_eq_40bar_water_no_piston[i] * params.A_p) / (dx_eq_40bar_water_no_piston[i] * params.A_p)

#lithium#
#M=1.69#
x_fluid_eq_40bar_lithium= ms.equilibrate_fluid_positions(params.n,params.L0,params.k_s_lithium,params.m_e_lithium,params.m_piston_ref,params.g,params.p_t0,p_p0)
for i in range(len(x_fluid_eq_40bar_lithium)):
    x_fluid_eq_40bar_lithium[i]=x_fluid_eq_40bar_lithium[i]+params.x_air_bottom
x_p0_40bar_lithium=x_fluid_eq_40bar_lithium[0]
offset= x_p0_40bar_lithium - 0.25
for i in range(len(x_fluid_eq_40bar_lithium)):
    x_fluid_eq_40bar_lithium[i]=x_fluid_eq_40bar_lithium[i]-offset
x_p0_40bar_lithium=x_fluid_eq_40bar_lithium[0]

dx_eq_40bar_lithium= np.zeros(params.n)
dx_eq_40bar_lithium[0]=x_fluid_eq_40bar_lithium[1]-x_p0_40bar_lithium
for i in range (1,params.n):
    dx_eq_40bar_lithium[i]=x_fluid_eq_40bar_lithium[i]-x_fluid_eq_40bar_lithium[i-1]

V_p0 = params.A_p * (x_p0_40bar_lithium-params.th_piston_ref) 
V_t0 = params.A_p * (params.H_ref-x_fluid_eq_40bar_lithium[params.n]) 

v_p0 = 0
v_fluid0=np.zeros(params.n)
y0 = np.concatenate((x_fluid_eq_40bar_lithium, [v_p0], v_fluid0))


sol_40bar_lithium = solve_ivp(ms.system_lith, [0, T_40], y0, args=(params.m_piston_ref,params.H_ref, V_t0, V_p0, params.n, p_p0), atol = 1e-6, rtol=1e-6, max_step = 1e-6)


x_p_40bar_lithium = sol_40bar_lithium.y[0]    
x_fluid_40bar_lithium = sol_40bar_lithium.y[1:params.n+1] 
v_fluid_40bar_lithium = sol_40bar_lithium.y[params.n+1:2*(params.n+1)]
a0_values_40bar_lithium = [ms.piston_acceleration_lith(y, params.m_piston_ref, V_p0, p_p0) for y in sol_40bar_lithium.y.T]
a0_min_40bar_lithium = np.min(a0_values_40bar_lithium)
a0_max_40bar_lithium = np.max(a0_values_40bar_lithium)
v_piston_40bar_lithium = sol_40bar_lithium.y[params.n+1, :]
v_piston_max_40bar_lithium=np.max(v_piston_40bar_lithium)


p_t_40bar_lithium=[]
for i in range(len(sol_40bar_lithium.t)):
    p_ti=ms.pressure_on_top(sol_40bar_lithium.y[:,i], params.H_ref, params.th_piston_ref, V_t0, params.n)
    p_t_40bar_lithium.append(p_ti)


p_p_40bar_lithium=[]
for i in range(len(sol_40bar_lithium.t)):
    p_pi=ms.pressure_under_piston(sol_40bar_lithium.y[:,i], params.H_ref, params.th_piston_ref, V_p0, params.n,p_p0)
    p_p_40bar_lithium.append(p_pi)


dx_40bar_lithium = np.zeros((params.n, len(sol_40bar_lithium.t)))
for j in range(len(sol_40bar_lithium.t)):
    dx_40bar_lithium[0,j]=x_fluid_40bar_lithium[0,j]-x_p_40bar_lithium[j]
    for i in range (1,params.n):
        dx_40bar_lithium[i,j]=x_fluid_40bar_lithium[i,j]-x_fluid_40bar_lithium[i-1,j]
p_40bar_lithium= np.ones((params.n,len(sol_40bar_lithium.t)))*params.p_0
for i in range(params.n):
    for j in range(len(sol_40bar_lithium.t)):
        p_40bar_lithium[i,j]=params.p_0-params.rho_l_lithium*params.c_lithium**2*((dx_40bar_lithium[i,j]*params.A_p)-dx_eq_40bar_lithium[i]*params.A_p) /(dx_eq_40bar_lithium[i]*params.A_p)

#Driving pressure: 5MPa#
p_p0 = 50e5
T_50 = 0.03

#water#
#M=0.1#
x_fluid_eq_50bar_water_01= ms.equilibrate_fluid_positions(params.n,params.L0,params.k_s_water,params.m_e_water,params.m_piston_01,params.g,params.p_t0,p_p0)
for i in range(len(x_fluid_eq_50bar_water_01)):
    x_fluid_eq_50bar_water_01[i]=x_fluid_eq_50bar_water_01[i]+params.x_air_bottom
x_p0_50bar_water_01=x_fluid_eq_50bar_water_01[0]
offset= x_p0_50bar_water_01 - (params.x_air_bottom+params.th_piston_01)
for i in range(len(x_fluid_eq_50bar_water_01)):
    x_fluid_eq_50bar_water_01[i]=x_fluid_eq_50bar_water_01[i]-offset
x_p0_50bar_water_01=x_fluid_eq_50bar_water_01[0]

dx_eq_50bar_water_01 = np.zeros(params.n)
dx_eq_50bar_water_01[0]=x_fluid_eq_50bar_water_01[1]-x_p0_50bar_water_01
for i in range (1,params.n):
    dx_eq_50bar_water_01[i]=x_fluid_eq_50bar_water_01[i]-x_fluid_eq_50bar_water_01[i-1]

V_p0_50bar_01 = params.A_p * (x_p0_50bar_water_01-params.th_piston_01)
V_t0_50bar_01 = params.A_p * (params.H_01-x_fluid_eq_50bar_water_01[params.n]) 

v_p0 = 0 
v_fluid0=np.zeros(params.n)
y0 = np.concatenate((x_fluid_eq_50bar_water_01, [v_p0], v_fluid0))

p_0_x_50bar_water_01= np.zeros(params.n)
for i in range (params.n):
    p_0_x_50bar_water_01[i] = params.p_t0 + params.rho_water*params.g*(x_fluid_eq_50bar_water_01[params.n]-x_fluid_eq_50bar_water_01[i])

sol_50bar_water_01 = solve_ivp(ms.system_water, [0, T_50], y0, args=(params.m_piston_01,params.H_01, V_t0_50bar_01, V_p0_50bar_01, params.k_s_water, params.m_e_water, params.n, p_p0), atol = 1e-6, rtol=1e-6, max_step = 1e-6)
sol_50bar_water_01.t=np.array(sol_50bar_water_01.t)

x_p_50bar_water_01 = sol_50bar_water_01.y[0]      
x_fluid_50bar_water_01 = sol_50bar_water_01.y[1:params.n+1] 
v_fluid_50bar_water_01 = sol_50bar_water_01.y[params.n+1:2*(params.n+1)] 
a0_values_50bar_water_01 = [ms.piston_acceleration_water(y,params.m_piston_01, params.H_01, V_p0_50bar_01, params.n, params.k_s_water, p_p0) for y in sol_50bar_water_01.y.T]
a0_min_50bar_water_01 = np.min(a0_values_50bar_water_01)
a0_max_50bar_water_01 = np.max(a0_values_50bar_water_01)
v_piston_50bar_water_01 = sol_50bar_water_01.y[params.n+1, :]
v_piston_max_50bar_water_01=np.max(v_piston_50bar_water_01)

all_acc_50bar_water_01 = np.array([
    ms.all_accelerations_water(y, params.m_piston_01, params.H_01, V_t0_50bar_01, V_p0_50bar_01, params.k_s_water, params.m_e_water, params.n, p_p0)
    for y in sol_50bar_water_01.y.T
])

a_piston_50bar_water_01 = all_acc_50bar_water_01[:, 0]     
a_fluid_50bar_water_01  = all_acc_50bar_water_01[:, 1:].T  
V_p_50bar_water_01 = params.A_p * (x_p_50bar_water_01 - params.th_piston_01)
V_t_50bar_water_01 = params.A_p * (params.H_01 - x_fluid_50bar_water_01[params.n-1, :])

p_t_50bar_water_01=[]
for i in range(len(sol_50bar_water_01.t)):
    p_ti=ms.pressure_on_top(sol_50bar_water_01.y[:,i], params.H_01, params.th_piston_01, V_t0_50bar_01, params.n)
    p_t_50bar_water_01.append(p_ti)
p_t_50bar_water_01 = np.array(p_t_50bar_water_01)
p_t_50bar_water_max_01 = np.max(p_t_50bar_water_01)

p_p_50bar_water_01=[]
for i in range(len(sol_50bar_water_01.t)):
    p_pi=ms.pressure_under_piston(sol_50bar_water_01.y[:,i], params.H_01, params.th_piston_01, V_p0_50bar_01, params.n,p_p0)
    p_p_50bar_water_01.append(p_pi)
p_p_50bar_water_01 = np.array(p_p_50bar_water_01)

dx_50bar_water_01 = np.zeros((params.n, len(sol_50bar_water_01.t)))
for j in range(len(sol_50bar_water_01.t)):
    dx_50bar_water_01[0,j]=x_fluid_50bar_water_01[0,j]-x_p_50bar_water_01[j]
    for i in range (1,params.n):
        dx_50bar_water_01[i,j]=x_fluid_50bar_water_01[i,j]-x_fluid_50bar_water_01[i-1,j]
p_50bar_water_01= np.ones((params.n,len(sol_50bar_water_01.t)))*params.p_0
for i in range(params.n):
    for j in range(len(sol_50bar_water_01.t)):
        p_50bar_water_01[i,j]=params.p_0-params.rho_l_water*params.c_water**2*((dx_50bar_water_01[i,j]*params.A_p)-dx_eq_50bar_water_01[i]*params.A_p) /(dx_eq_50bar_water_01[i]*params.A_p)

#M=0.5#
x_fluid_eq_50bar_water_05= ms.equilibrate_fluid_positions(params.n,params.L0,params.k_s_water,params.m_e_water,params.m_piston_05,params.g,params.p_t0,p_p0)
for i in range(len(x_fluid_eq_50bar_water_05)):
    x_fluid_eq_50bar_water_05[i]=x_fluid_eq_50bar_water_05[i]+params.x_air_bottom
x_p0_50bar_water_05=x_fluid_eq_50bar_water_05[0]
offset= x_p0_50bar_water_05 - (params.x_air_bottom+params.th_piston_05)
for i in range(len(x_fluid_eq_50bar_water_05)):
    x_fluid_eq_50bar_water_05[i]=x_fluid_eq_50bar_water_05[i]-offset
x_p0_50bar_water_05=x_fluid_eq_50bar_water_05[0]

dx_eq_50bar_water_05 = np.zeros(params.n)
dx_eq_50bar_water_05[0]=x_fluid_eq_50bar_water_05[1]-x_p0_50bar_water_05
for i in range (1,params.n):
    dx_eq_50bar_water_05[i]=x_fluid_eq_50bar_water_05[i]-x_fluid_eq_50bar_water_05[i-1]

V_p0_50bar_05 = params.A_p * (x_p0_50bar_water_05-params.th_piston_05) 
V_t0_50bar_05 = params.A_p * (params.H_05-x_fluid_eq_50bar_water_05[params.n])  

v_p0 = 0 
v_fluid0=np.zeros(params.n)
y0 = np.concatenate((x_fluid_eq_50bar_water_05, [v_p0], v_fluid0))

p_0_x_50bar_water_05= np.zeros(params.n)
for i in range (params.n):
    p_0_x_50bar_water_05[i] = params.p_t0 + params.rho_water*params.g*(x_fluid_eq_50bar_water_05[params.n]-x_fluid_eq_50bar_water_05[i])


sol_50bar_water_05 = solve_ivp(ms.system_water, [0, T_50], y0, args=(params.m_piston_05,params.H_05, V_t0_50bar_05, V_p0_50bar_05, params.k_s_water, params.m_e_water, params.n, p_p0), atol = 1e-6, rtol=1e-6, max_step = 1e-6)
sol_50bar_water_05.t=np.array(sol_50bar_water_05.t)

x_p_50bar_water_05 = sol_50bar_water_05.y[0]      
x_fluid_50bar_water_05 = sol_50bar_water_05.y[1:params.n+1] 
v_fluid_50bar_water_05 = sol_50bar_water_05.y[params.n+1:2*(params.n+1)] 
a0_values_50bar_water_05 = [ms.piston_acceleration_water(y,params.m_piston_05, params.H_05, V_p0_50bar_05, params.n, params.k_s_water, p_p0) for y in sol_50bar_water_05.y.T]
a0_min_50bar_water_05 = np.min(a0_values_50bar_water_05)
a0_max_50bar_water_05 = np.max(a0_values_50bar_water_05)
v_piston_50bar_water_05 = sol_50bar_water_05.y[params.n+1, :]
v_piston_max_50bar_water_05=np.max(v_piston_50bar_water_05)

all_acc_50bar_water_05 = np.array([
    ms.all_accelerations_water(y, params.m_piston_05, params.H_05, V_t0_50bar_05, V_p0_50bar_05, params.k_s_water, params.m_e_water, params.n, p_p0)
    for y in sol_50bar_water_05.y.T
])

a_piston_50bar_water_05 = all_acc_50bar_water_05[:, 0]     
a_fluid_50bar_water_05  = all_acc_50bar_water_05[:, 1:].T  
V_p_50bar_water_05 = params.A_p * (x_p_50bar_water_05 - params.th_piston_05)
V_t_50bar_water_05 = params.A_p * (params.H_05 - x_fluid_50bar_water_05[params.n-1, :])

p_t_50bar_water_05=[]
for i in range(len(sol_50bar_water_05.t)):
    p_ti=ms.pressure_on_top(sol_50bar_water_05.y[:,i], params.H_05, params.th_piston_05, V_t0_50bar_05, params.n)
    p_t_50bar_water_05.append(p_ti)
p_t_50bar_water_05 = np.array(p_t_50bar_water_05)
p_t_50bar_water_max_05 = np.max(p_t_50bar_water_05)

p_p_50bar_water_05=[]
for i in range(len(sol_50bar_water_05.t)):
    p_pi=ms.pressure_under_piston(sol_50bar_water_05.y[:,i], params.H_05, params.th_piston_05, V_p0_50bar_05, params.n,p_p0)
    p_p_50bar_water_05.append(p_pi)
p_p_50bar_water_05 = np.array(p_p_50bar_water_05)

dx_50bar_water_05 = np.zeros((params.n, len(sol_50bar_water_05.t)))
for j in range(len(sol_50bar_water_05.t)):
    dx_50bar_water_05[0,j]=x_fluid_50bar_water_05[0,j]-x_p_50bar_water_05[j]
    for i in range (1,params.n):
        dx_50bar_water_05[i,j]=x_fluid_50bar_water_05[i,j]-x_fluid_50bar_water_05[i-1,j]
p_50bar_water_05= np.ones((params.n,len(sol_50bar_water_05.t)))*params.p_0
for i in range(params.n):
    for j in range(len(sol_50bar_water_05.t)):
        p_50bar_water_05[i,j]=params.p_0-params.rho_l_water*params.c_water**2*((dx_50bar_water_05[i,j]*params.A_p)-dx_eq_50bar_water_05[i]*params.A_p) /(dx_eq_50bar_water_05[i]*params.A_p)

#M=0.9#
x_fluid_eq_50bar_water_09= ms.equilibrate_fluid_positions(params.n,params.L0,params.k_s_water,params.m_e_water,params.m_piston_09,params.g,params.p_t0,p_p0)
for i in range(len(x_fluid_eq_50bar_water_09)):
    x_fluid_eq_50bar_water_09[i]=x_fluid_eq_50bar_water_09[i]+params.x_air_bottom
x_p0_50bar_water_09=x_fluid_eq_50bar_water_09[0]
offset= x_p0_50bar_water_09 - (params.x_air_bottom+params.th_piston_09)
for i in range(len(x_fluid_eq_50bar_water_09)):
    x_fluid_eq_50bar_water_09[i]=x_fluid_eq_50bar_water_09[i]-offset
x_p0_50bar_water_09=x_fluid_eq_50bar_water_09[0]

dx_eq_50bar_water_09 = np.zeros(params.n)
dx_eq_50bar_water_09[0]=x_fluid_eq_50bar_water_09[1]-x_p0_50bar_water_09
for i in range (1,params.n):
    dx_eq_50bar_water_09[i]=x_fluid_eq_50bar_water_09[i]-x_fluid_eq_50bar_water_09[i-1]

V_p0_50bar_09 = params.A_p * (x_p0_50bar_water_09-params.th_piston_09) 
V_t0_50bar_09 = params.A_p * (params.H_09-x_fluid_eq_50bar_water_09[params.n])  

v_p0 = 0 
v_fluid0=np.zeros(params.n)
y0 = np.concatenate((x_fluid_eq_50bar_water_09, [v_p0], v_fluid0))

p_0_x_50bar_water_09= np.zeros(params.n)
for i in range (params.n):
    p_0_x_50bar_water_09[i] = params.p_t0 + params.rho_water*params.g*(x_fluid_eq_50bar_water_09[params.n]-x_fluid_eq_50bar_water_09[i])


sol_50bar_water_09 = solve_ivp(ms.system_water, [0, T_50], y0, args=(params.m_piston_09,params.H_09, V_t0_50bar_09, V_p0_50bar_09, params.k_s_water, params.m_e_water, params.n, p_p0), atol = 1e-6, rtol=1e-6, max_step = 1e-6)
sol_50bar_water_09.t=np.array(sol_50bar_water_09.t)

x_p_50bar_water_09 = sol_50bar_water_09.y[0]      
x_fluid_50bar_water_09 = sol_50bar_water_09.y[1:params.n+1] 
v_fluid_50bar_water_09 = sol_50bar_water_09.y[params.n+1:2*(params.n+1)] 
a0_values_50bar_water_09 = [ms.piston_acceleration_water(y,params.m_piston_09, params.H_09, V_p0_50bar_09, params.n, params.k_s_water, p_p0) for y in sol_50bar_water_09.y.T]
a0_min_50bar_water_09 = np.min(a0_values_50bar_water_09)
a0_max_50bar_water_09 = np.max(a0_values_50bar_water_09)
v_piston_50bar_water_09 = sol_50bar_water_09.y[params.n+1, :]
v_piston_max_50bar_water_09=np.max(v_piston_50bar_water_09)

all_acc_50bar_water_09 = np.array([
    ms.all_accelerations_water(y, params.m_piston_09, params.H_09, V_t0_50bar_09, V_p0_50bar_09, params.k_s_water, params.m_e_water, params.n, p_p0)
    for y in sol_50bar_water_09.y.T
])

a_piston_50bar_water_09 = all_acc_50bar_water_09[:, 0]      
a_fluid_50bar_water_09  = all_acc_50bar_water_09[:, 1:].T  
V_p_50bar_water_09 = params.A_p * (x_p_50bar_water_09 - params.th_piston_09)
V_t_50bar_water_09 = params.A_p * (params.H_09 - x_fluid_50bar_water_09[params.n-1, :])

p_t_50bar_water_09=[]
for i in range(len(sol_50bar_water_09.t)):
    p_ti=ms.pressure_on_top(sol_50bar_water_09.y[:,i], params.H_09, params.th_piston_09, V_t0_50bar_09, params.n)
    p_t_50bar_water_09.append(p_ti)
p_t_50bar_water_09 = np.array(p_t_50bar_water_09)
p_t_50bar_water_max_09 = np.max(p_t_50bar_water_09)

p_p_50bar_water_09=[]
for i in range(len(sol_50bar_water_09.t)):
    p_pi=ms.pressure_under_piston(sol_50bar_water_09.y[:,i], params.H_09, params.th_piston_09, V_p0_50bar_09, params.n,p_p0)
    p_p_50bar_water_09.append(p_pi)
p_p_50bar_water_09 = np.array(p_p_50bar_water_09)

dx_50bar_water_09 = np.zeros((params.n, len(sol_50bar_water_09.t)))
for j in range(len(sol_50bar_water_09.t)):
    dx_50bar_water_09[0,j]=x_fluid_50bar_water_09[0,j]-x_p_50bar_water_09[j]
    for i in range (1,params.n):
        dx_50bar_water_09[i,j]=x_fluid_50bar_water_09[i,j]-x_fluid_50bar_water_09[i-1,j]
p_50bar_water_09= np.ones((params.n,len(sol_50bar_water_09.t)))*params.p_0
for i in range(params.n):
    for j in range(len(sol_50bar_water_09.t)):
        p_50bar_water_09[i,j]=params.p_0-params.rho_l_water*params.c_water**2*((dx_50bar_water_09[i,j]*params.A_p)-dx_eq_50bar_water_09[i]*params.A_p) /(dx_eq_50bar_water_09[i]*params.A_p)

#M=1#
x_fluid_eq_50bar_water_1= ms.equilibrate_fluid_positions(params.n,params.L0,params.k_s_water,params.m_e_water,params.m_piston_1,params.g,params.p_t0,p_p0)
for i in range(len(x_fluid_eq_50bar_water_1)):
    x_fluid_eq_50bar_water_1[i]=x_fluid_eq_50bar_water_1[i]+params.x_air_bottom
x_p0_50bar_water_1=x_fluid_eq_50bar_water_1[0]
offset= x_p0_50bar_water_1 - (params.x_air_bottom+params.th_piston_1)
for i in range(len(x_fluid_eq_50bar_water_1)):
    x_fluid_eq_50bar_water_1[i]=x_fluid_eq_50bar_water_1[i]-offset
x_p0_50bar_water_1=x_fluid_eq_50bar_water_1[0]

dx_eq_50bar_water_1 = np.zeros(params.n)
dx_eq_50bar_water_1[0]=x_fluid_eq_50bar_water_1[1]-x_p0_50bar_water_1
for i in range (1,params.n):
    dx_eq_50bar_water_1[i]=x_fluid_eq_50bar_water_1[i]-x_fluid_eq_50bar_water_1[i-1]

V_p0_50bar_1 = params.A_p * (x_p0_50bar_water_1-params.th_piston_1) 
V_t0_50bar_1 = params.A_p * (params.H_1-x_fluid_eq_50bar_water_1[params.n]) 

v_p0 = 0 
v_fluid0=np.zeros(params.n)
y0 = np.concatenate((x_fluid_eq_50bar_water_1, [v_p0], v_fluid0))

p_0_x_50bar_water_1= np.zeros(params.n)
for i in range (params.n):
    p_0_x_50bar_water_1[i] = params.p_t0 + params.rho_water*params.g*(x_fluid_eq_50bar_water_1[params.n]-x_fluid_eq_50bar_water_1[i])


sol_50bar_water_1 = solve_ivp(ms.system_water, [0, T_50], y0, args=(params.m_piston_1,params.H_1, V_t0_50bar_1, V_p0_50bar_1, params.k_s_water, params.m_e_water, params.n, p_p0), atol = 1e-6, rtol=1e-6, max_step = 1e-6)
sol_50bar_water_1.t=np.array(sol_50bar_water_1.t)

x_p_50bar_water_1 = sol_50bar_water_1.y[0]    
x_fluid_50bar_water_1 = sol_50bar_water_1.y[1:params.n+1]
v_fluid_50bar_water_1 = sol_50bar_water_1.y[params.n+1:2*(params.n+1)] 
a0_values_50bar_water_1 = [ms.piston_acceleration_water(y,params.m_piston_1, params.H_1, V_p0_50bar_1, params.n, params.k_s_water, p_p0) for y in sol_50bar_water_1.y.T]
a0_min_50bar_water_1 = np.min(a0_values_50bar_water_1)
a0_max_50bar_water_1 = np.max(a0_values_50bar_water_1)
v_piston_50bar_water_1 = sol_50bar_water_1.y[params.n+1, :]
v_piston_max_50bar_water_1=np.max(v_piston_50bar_water_1)

all_acc_50bar_water_1 = np.array([
    ms.all_accelerations_water(y, params.m_piston_1, params.H_1, V_t0_50bar_1, V_p0_50bar_1, params.k_s_water, params.m_e_water, params.n, p_p0)
    for y in sol_50bar_water_1.y.T
])

a_piston_50bar_water_1 = all_acc_50bar_water_1[:, 0]      
a_fluid_50bar_water_1  = all_acc_50bar_water_1[:, 1:].T  
V_p_50bar_water_1 = params.A_p * (x_p_50bar_water_1 - params.th_piston_1)
V_t_50bar_water_1 = params.A_p * (params.H_1 - x_fluid_50bar_water_1[params.n-1, :])

p_t_50bar_water_1=[]
for i in range(len(sol_50bar_water_1.t)):
    p_ti=ms.pressure_on_top(sol_50bar_water_1.y[:,i], params.H_1, params.th_piston_1, V_t0_50bar_1, params.n)
    p_t_50bar_water_1.append(p_ti)
p_t_50bar_water_1 = np.array(p_t_50bar_water_1)
p_t_50bar_water_max_1 = np.max(p_t_50bar_water_1)

p_p_50bar_water_1=[]
for i in range(len(sol_50bar_water_1.t)):
    p_pi=ms.pressure_under_piston(sol_50bar_water_1.y[:,i], params.H_1, params.th_piston_1, V_p0_50bar_1, params.n,p_p0)
    p_p_50bar_water_1.append(p_pi)
p_p_50bar_water_1 = np.array(p_p_50bar_water_1)

dx_50bar_water_1 = np.zeros((params.n, len(sol_50bar_water_1.t)))
for j in range(len(sol_50bar_water_1.t)):
    dx_50bar_water_1[0,j]=x_fluid_50bar_water_1[0,j]-x_p_50bar_water_1[j]
    for i in range (1,params.n):
        dx_50bar_water_1[i,j]=x_fluid_50bar_water_1[i,j]-x_fluid_50bar_water_1[i-1,j]
p_50bar_water_1= np.ones((params.n,len(sol_50bar_water_1.t)))*params.p_0
for i in range(params.n):
    for j in range(len(sol_50bar_water_1.t)):
        p_50bar_water_1[i,j]=params.p_0-params.rho_l_water*params.c_water**2*((dx_50bar_water_1[i,j]*params.A_p)-dx_eq_50bar_water_1[i]*params.A_p) /(dx_eq_50bar_water_1[i]*params.A_p)

#M=1.5#
x_fluid_eq_50bar_water_15= ms.equilibrate_fluid_positions(params.n,params.L0,params.k_s_water,params.m_e_water,params.m_piston_15,params.g,params.p_t0,p_p0)
for i in range(len(x_fluid_eq_50bar_water_15)):
    x_fluid_eq_50bar_water_15[i]=x_fluid_eq_50bar_water_15[i]+params.x_air_bottom
x_p0_50bar_water_15=x_fluid_eq_50bar_water_15[0]
offset= x_p0_50bar_water_15 - (params.x_air_bottom+params.th_piston_15)
for i in range(len(x_fluid_eq_50bar_water_15)):
    x_fluid_eq_50bar_water_15[i]=x_fluid_eq_50bar_water_15[i]-offset
x_p0_50bar_water_15=x_fluid_eq_50bar_water_15[0]

dx_eq_50bar_water_15 = np.zeros(params.n)
dx_eq_50bar_water_15[0]=x_fluid_eq_50bar_water_15[1]-x_p0_50bar_water_15
for i in range (1,params.n):
    dx_eq_50bar_water_15[i]=x_fluid_eq_50bar_water_15[i]-x_fluid_eq_50bar_water_15[i-1]

V_p0_50bar_15 = params.A_p * (x_p0_50bar_water_15-params.th_piston_15) 
V_t0_50bar_15 = params.A_p * (params.H_15-x_fluid_eq_50bar_water_15[params.n])  

v_p0 = 0 
v_fluid0=np.zeros(params.n)
y0 = np.concatenate((x_fluid_eq_50bar_water_15, [v_p0], v_fluid0))

p_0_x_50bar_water_15= np.zeros(params.n)
for i in range (params.n):
    p_0_x_50bar_water_15[i] = params.p_t0 + params.rho_water*params.g*(x_fluid_eq_50bar_water_15[params.n]-x_fluid_eq_50bar_water_15[i])


sol_50bar_water_15 = solve_ivp(ms.system_water, [0, T_50], y0, args=(params.m_piston_15,params.H_15, V_t0_50bar_15, V_p0_50bar_15, params.k_s_water, params.m_e_water, params.n, p_p0), atol = 1e-6, rtol=1e-6, max_step = 1e-6)
sol_50bar_water_15.t=np.array(sol_50bar_water_15.t)

x_p_50bar_water_15 = sol_50bar_water_15.y[0]      
x_fluid_50bar_water_15 = sol_50bar_water_15.y[1:params.n+1]
v_fluid_50bar_water_15 = sol_50bar_water_15.y[params.n+1:2*(params.n+1)] 
a0_values_50bar_water_15 = [ms.piston_acceleration_water(y,params.m_piston_15, params.H_15, V_p0_50bar_15, params.n, params.k_s_water, p_p0) for y in sol_50bar_water_15.y.T]
a0_min_50bar_water_15 = np.min(a0_values_50bar_water_15)
a0_max_50bar_water_15 = np.max(a0_values_50bar_water_15)
v_piston_50bar_water_15 = sol_50bar_water_15.y[params.n+1, :]
v_piston_max_50bar_water_15=np.max(v_piston_50bar_water_15)

all_acc_50bar_water_15 = np.array([
    ms.all_accelerations_water(y, params.m_piston_15, params.H_15, V_t0_50bar_15, V_p0_50bar_15, params.k_s_water, params.m_e_water, params.n, p_p0)
    for y in sol_50bar_water_15.y.T
])

a_piston_50bar_water_15 = all_acc_50bar_water_15[:, 0]      
a_fluid_50bar_water_15  = all_acc_50bar_water_15[:, 1:].T  
V_p_50bar_water_15 = params.A_p * (x_p_50bar_water_15 - params.th_piston_15)
V_t_50bar_water_15 = params.A_p * (params.H_15 - x_fluid_50bar_water_15[params.n-1, :])

p_t_50bar_water_15=[]
for i in range(len(sol_50bar_water_15.t)):
    p_ti=ms.pressure_on_top(sol_50bar_water_15.y[:,i], params.H_15, params.th_piston_15, V_t0_50bar_15, params.n)
    p_t_50bar_water_15.append(p_ti)
p_t_50bar_water_15 = np.array(p_t_50bar_water_15)
p_t_50bar_water_max_15 = np.max(p_t_50bar_water_15)

p_p_50bar_water_15=[]
for i in range(len(sol_50bar_water_15.t)):
    p_pi=ms.pressure_under_piston(sol_50bar_water_15.y[:,i], params.H_15, params.th_piston_15, V_p0_50bar_15, params.n,p_p0)
    p_p_50bar_water_15.append(p_pi)
p_p_50bar_water_15 = np.array(p_p_50bar_water_15)

dx_50bar_water_15 = np.zeros((params.n, len(sol_50bar_water_15.t)))
for j in range(len(sol_50bar_water_15.t)):
    dx_50bar_water_15[0,j]=x_fluid_50bar_water_15[0,j]-x_p_50bar_water_15[j]
    for i in range (1,params.n):
        dx_50bar_water_15[i,j]=x_fluid_50bar_water_15[i,j]-x_fluid_50bar_water_15[i-1,j]
p_50bar_water_15= np.ones((params.n,len(sol_50bar_water_15.t)))*params.p_0
for i in range(params.n):
    for j in range(len(sol_50bar_water_15.t)):
        p_50bar_water_15[i,j]=params.p_0-params.rho_l_water*params.c_water**2*((dx_50bar_water_15[i,j]*params.A_p)-dx_eq_50bar_water_15[i]*params.A_p) /(dx_eq_50bar_water_15[i]*params.A_p)

#M=2#
x_fluid_eq_50bar_water_2= ms.equilibrate_fluid_positions(params.n,params.L0,params.k_s_water,params.m_e_water,params.m_piston_2,params.g,params.p_t0,p_p0)
for i in range(len(x_fluid_eq_50bar_water_2)):
    x_fluid_eq_50bar_water_2[i]=x_fluid_eq_50bar_water_2[i]+params.x_air_bottom
x_p0_50bar_water_2=x_fluid_eq_50bar_water_2[0]
offset= x_p0_50bar_water_2 - (params.x_air_bottom+params.th_piston_2)
for i in range(len(x_fluid_eq_50bar_water_2)):
    x_fluid_eq_50bar_water_2[i]=x_fluid_eq_50bar_water_2[i]-offset
x_p0_50bar_water_2=x_fluid_eq_50bar_water_2[0]

dx_eq_50bar_water_2 = np.zeros(params.n)
dx_eq_50bar_water_2[0]=x_fluid_eq_50bar_water_2[1]-x_p0_50bar_water_2
for i in range (1,params.n):
    dx_eq_50bar_water_2[i]=x_fluid_eq_50bar_water_2[i]-x_fluid_eq_50bar_water_2[i-1]

V_p0_50bar_2 = params.A_p * (x_p0_50bar_water_2-params.th_piston_2) 
V_t0_50bar_2 = params.A_p * (params.H_2-x_fluid_eq_50bar_water_2[params.n])  

v_p0 = 0 
v_fluid0=np.zeros(params.n)
y0 = np.concatenate((x_fluid_eq_50bar_water_2, [v_p0], v_fluid0))

p_0_x_50bar_water_2= np.zeros(params.n)
for i in range (params.n):
    p_0_x_50bar_water_2[i] = params.p_t0 + params.rho_water*params.g*(x_fluid_eq_50bar_water_2[params.n]-x_fluid_eq_50bar_water_2[i])


sol_50bar_water_2 = solve_ivp(ms.system_water, [0, T_50], y0, args=(params.m_piston_2,params.H_2, V_t0_50bar_2, V_p0_50bar_2, params.k_s_water, params.m_e_water, params.n, p_p0), atol = 1e-6, rtol=1e-6, max_step = 1e-6)
sol_50bar_water_2.t=np.array(sol_50bar_water_2.t)

x_p_50bar_water_2 = sol_50bar_water_2.y[0]    
x_fluid_50bar_water_2 = sol_50bar_water_2.y[1:params.n+1] 
v_fluid_50bar_water_2 = sol_50bar_water_2.y[params.n+1:2*(params.n+1)]
a0_values_50bar_water_2 = [ms.piston_acceleration_water(y,params.m_piston_2, params.H_2, V_p0_50bar_2, params.n, params.k_s_water, p_p0) for y in sol_50bar_water_2.y.T]
a0_min_50bar_water_2 = np.min(a0_values_50bar_water_2)
a0_max_50bar_water_2 = np.max(a0_values_50bar_water_2)
v_piston_50bar_water_2 = sol_50bar_water_2.y[params.n+1, :]
v_piston_max_50bar_water_2=np.max(v_piston_50bar_water_2)

all_acc_50bar_water_2 = np.array([
    ms.all_accelerations_water(y, params.m_piston_2, params.H_2, V_t0_50bar_2, V_p0_50bar_2, params.k_s_water, params.m_e_water, params.n, p_p0)
    for y in sol_50bar_water_2.y.T
])

a_piston_50bar_water_2 = all_acc_50bar_water_2[:, 0]      
a_fluid_50bar_water_2  = all_acc_50bar_water_2[:, 1:].T   
V_p_50bar_water_2 = params.A_p * (x_p_50bar_water_2 - params.th_piston_2)
V_t_50bar_water_2 = params.A_p * (params.H_2 - x_fluid_50bar_water_2[params.n-1, :])

p_t_50bar_water_2=[]
for i in range(len(sol_50bar_water_2.t)):
    p_ti=ms.pressure_on_top(sol_50bar_water_2.y[:,i], params.H_2, params.th_piston_2, V_t0_50bar_2, params.n)
    p_t_50bar_water_2.append(p_ti)
p_t_50bar_water_2 = np.array(p_t_50bar_water_2)
p_t_50bar_water_max_2 = np.max(p_t_50bar_water_2)

p_p_50bar_water_2=[]
for i in range(len(sol_50bar_water_2.t)):
    p_pi=ms.pressure_under_piston(sol_50bar_water_2.y[:,i], params.H_2, params.th_piston_2, V_p0_50bar_2, params.n,p_p0)
    p_p_50bar_water_2.append(p_pi)
p_p_50bar_water_2 = np.array(p_p_50bar_water_2)

dx_50bar_water_2 = np.zeros((params.n, len(sol_50bar_water_2.t)))
for j in range(len(sol_50bar_water_2.t)):
    dx_50bar_water_2[0,j]=x_fluid_50bar_water_2[0,j]-x_p_50bar_water_2[j]
    for i in range (1,params.n):
        dx_50bar_water_2[i,j]=x_fluid_50bar_water_2[i,j]-x_fluid_50bar_water_2[i-1,j]
p_50bar_water_2= np.ones((params.n,len(sol_50bar_water_2.t)))*params.p_0
for i in range(params.n):
    for j in range(len(sol_50bar_water_2.t)):
        p_50bar_water_2[i,j]=params.p_0-params.rho_l_water*params.c_water**2*((dx_50bar_water_2[i,j]*params.A_p)-dx_eq_50bar_water_2[i]*params.A_p) /(dx_eq_50bar_water_2[i]*params.A_p)

#no piston#
x_fluid_eq_50bar_water_no_piston = ms.equilibrate_fluid_positions_no_piston(params.n, params.L0, params.k_s_water, params.m_e_water, params.g, params.p_t0, p_p0)
x_0_50bar_water_no_piston = x_fluid_eq_50bar_water_no_piston[0]

dx_eq_50bar_water_no_piston = np.zeros(params.n)
dx_eq_50bar_water_no_piston[0] = x_fluid_eq_50bar_water_no_piston[1] - x_0_50bar_water_no_piston
for i in range(1, params.n):
    dx_eq_50bar_water_no_piston[i] = x_fluid_eq_50bar_water_no_piston[i] - x_fluid_eq_50bar_water_no_piston[i - 1]

V_p0 = params.A_p * x_0_50bar_water_no_piston  
V_t0 = params.A_p * (params.H_no_piston - x_fluid_eq_50bar_water_no_piston[params.n])  

v_0 = 0  
v_fluid0 = np.zeros(params.n)
y0 = np.concatenate((x_fluid_eq_50bar_water_no_piston, [v_0], v_fluid0))


sol_50bar_water_no_piston = solve_ivp(ms.system_water_no_piston, [0, T_50], y0, args=(V_t0,V_p0, p_p0), atol=1e-6, rtol=1e-6, max_step=1e-6)
sol_50bar_water_no_piston.t=np.array(sol_50bar_water_no_piston.t)

x_0_50bar_water_no_piston = sol_50bar_water_no_piston.y[0]  
x_fluid_50bar_water_no_piston = sol_50bar_water_no_piston.y[1:params.n + 1]  
v_fluid_50bar_water_no_piston = sol_50bar_water_no_piston.y[params.n + 1:2 * (params.n + 1)]  
a0_values_50bar_water_no_piston = [ms.liquid_acceleration_water(y, params.n, params.k_s_water, params.m_e_water, params.L0, V_p0, p_p0) for y in sol_50bar_water_no_piston.y.T]
a0_min_50bar_water_no_piston = np.min(a0_values_50bar_water_no_piston)
a0_max_50bar_water_no_piston = np.max(a0_values_50bar_water_no_piston)
v_piston_50bar_water_no_piston = sol_50bar_water_no_piston.y[params.n + 1, :]
v_piston_max_50bar_water_no_piston = np.max(v_piston_50bar_water_no_piston)

all_acc_50bar_water_no_piston = np.array([
    ms.all_accelerations_water_no_piston(y,V_t0, V_p0, p_p0)
    for y in sol_50bar_water_no_piston.y.T
])  

a_fluid_50bar_water_no_piston = all_acc_50bar_water_no_piston.T  
V_p_50bar_water_no_piston = params.A_p * x_0_50bar_water_no_piston
V_t_50bar_water_no_piston = params.A_p * (params.H_no_piston - x_fluid_50bar_water_no_piston[params.n-1, :])

p_t_50bar_water_no_piston = []
for i in range(len(sol_50bar_water_no_piston.t)):
    p_ti = ms.pressure_on_top_no_piston(sol_50bar_water_no_piston.y[:, i], V_t0)
    p_t_50bar_water_no_piston.append(p_ti)


p_p_50bar_water_no_piston = []
for i in range(len(sol_50bar_water_no_piston.t)):
    p_pi = ms.pressure_gas_bottom(sol_50bar_water_no_piston.y[:, i], V_p0, p_p0)
    p_p_50bar_water_no_piston.append(p_pi)


dx_50bar_water_no_piston = np.zeros((params.n, len(sol_50bar_water_no_piston.t)))
for j in range(len(sol_50bar_water_no_piston.t)):
    dx_50bar_water_no_piston[0, j] = x_fluid_50bar_water_no_piston[0, j] - x_0_50bar_water_no_piston[j]
    for i in range(1, params.n):
        dx_50bar_water_no_piston[i, j] = x_fluid_50bar_water_no_piston[i, j] - x_fluid_50bar_water_no_piston[i - 1, j]

p_50bar_water_no_piston = np.ones((params.n, len(sol_50bar_water_no_piston.t))) * params.p_0
for i in range(params.n):
    for j in range(len(sol_50bar_water_no_piston.t)):
        p_50bar_water_no_piston[i, j] = params.p_0 - params.rho_l_water * params.c_water**2 * ((dx_50bar_water_no_piston[i, j] * params.A_p) - dx_eq_50bar_water_no_piston[i] * params.A_p) / (dx_eq_50bar_water_no_piston[i] * params.A_p)

#lithium#
#M=1.69#
x_fluid_eq_50bar_lithium= ms.equilibrate_fluid_positions(params.n,params.L0,params.k_s_lithium,params.m_e_lithium,params.m_piston_ref,params.g,params.p_t0,p_p0)
for i in range(len(x_fluid_eq_50bar_lithium)):
    x_fluid_eq_50bar_lithium[i]=x_fluid_eq_50bar_lithium[i]+params.x_air_bottom
x_p0_50bar_lithium=x_fluid_eq_50bar_lithium[0]
offset= x_p0_50bar_lithium - 0.25
for i in range(len(x_fluid_eq_50bar_lithium)):
    x_fluid_eq_50bar_lithium[i]=x_fluid_eq_50bar_lithium[i]-offset
x_p0_50bar_lithium=x_fluid_eq_50bar_lithium[0]

dx_eq_50bar_lithium= np.zeros(params.n)
dx_eq_50bar_lithium[0]=x_fluid_eq_50bar_lithium[1]-x_p0_50bar_lithium
for i in range (1,params.n):
    dx_eq_50bar_lithium[i]=x_fluid_eq_50bar_lithium[i]-x_fluid_eq_50bar_lithium[i-1]

V_p0 = params.A_p * (x_p0_50bar_lithium-params.th_piston_ref) 
V_t0 = params.A_p * (params.H_ref-x_fluid_eq_50bar_lithium[params.n])  

v_p0 = 0 
v_fluid0=np.zeros(params.n)
y0 = np.concatenate((x_fluid_eq_50bar_lithium, [v_p0], v_fluid0))


sol_50bar_lithium = solve_ivp(ms.system_lith, [0, T_50], y0, args=(params.m_piston_ref,params.H_ref, V_t0, V_p0, params.n, p_p0), atol = 1e-6, rtol=1e-6, max_step = 1e-6)


x_p_50bar_lithium = sol_50bar_lithium.y[0]      
x_fluid_50bar_lithium = sol_50bar_lithium.y[1:params.n+1] 
v_fluid_50bar_lithium = sol_50bar_lithium.y[params.n+1:2*(params.n+1)] 
a0_values_50bar_lithium = [ms.piston_acceleration_lith(y, params.m_piston_ref, V_p0, p_p0) for y in sol_50bar_lithium.y.T]
a0_min_50bar_lithium = np.min(a0_values_50bar_lithium)
a0_max_50bar_lithium = np.max(a0_values_50bar_lithium)
v_piston_50bar_lithium = sol_50bar_lithium.y[params.n+1, :]
v_piston_max_50bar_lithium=np.max(v_piston_50bar_lithium)

p_t_50bar_lithium=[]
for i in range(len(sol_50bar_lithium.t)):
    p_ti=ms.pressure_on_top(sol_50bar_lithium.y[:,i], params.H_ref, params.th_piston_ref, V_t0, params.n)
    p_t_50bar_lithium.append(p_ti)


p_p_50bar_lithium=[]
for i in range(len(sol_50bar_lithium.t)):
    p_pi=ms.pressure_under_piston(sol_50bar_lithium.y[:,i], params.H_ref, params.th_piston_ref, V_p0, params.n,p_p0)
    p_p_50bar_lithium.append(p_pi)


dx_50bar_lithium = np.zeros((params.n, len(sol_50bar_lithium.t)))
for j in range(len(sol_50bar_lithium.t)):
    dx_50bar_lithium[0,j]=x_fluid_50bar_lithium[0,j]-x_p_50bar_lithium[j]
    for i in range (1,params.n):
        dx_50bar_lithium[i,j]=x_fluid_50bar_lithium[i,j]-x_fluid_50bar_lithium[i-1,j]
p_50bar_lithium= np.ones((params.n,len(sol_50bar_lithium.t)))*params.p_0
for i in range(params.n):
    for j in range(len(sol_50bar_lithium.t)):
        p_50bar_lithium[i,j]=params.p_0-params.rho_l_lithium*params.c_lithium**2*((dx_50bar_lithium[i,j]*params.A_p)-dx_eq_50bar_lithium[i]*params.A_p) /(dx_eq_50bar_lithium[i]*params.A_p)


#Driving pressure: 6MPa#
p_p0 = 60e5
T_60 = 0.02  

#water#
#M=0.1#
x_fluid_eq_60bar_water_01= ms.equilibrate_fluid_positions(params.n,params.L0,params.k_s_water,params.m_e_water,params.m_piston_01,params.g,params.p_t0,p_p0)
for i in range(len(x_fluid_eq_60bar_water_01)):
    x_fluid_eq_60bar_water_01[i]=x_fluid_eq_60bar_water_01[i]+params.x_air_bottom
x_p0_60bar_water_01=x_fluid_eq_60bar_water_01[0]
offset= x_p0_60bar_water_01 - (params.x_air_bottom+params.th_piston_01)
for i in range(len(x_fluid_eq_60bar_water_01)):
    x_fluid_eq_60bar_water_01[i]=x_fluid_eq_60bar_water_01[i]-offset
x_p0_60bar_water_01=x_fluid_eq_60bar_water_01[0]

dx_eq_60bar_water_01 = np.zeros(params.n)
dx_eq_60bar_water_01[0]=x_fluid_eq_60bar_water_01[1]-x_p0_60bar_water_01
for i in range (1,params.n):
    dx_eq_60bar_water_01[i]=x_fluid_eq_60bar_water_01[i]-x_fluid_eq_60bar_water_01[i-1]

V_p0_60bar_01 = params.A_p * (x_p0_60bar_water_01-params.th_piston_01) 
V_t0_60bar_01 = params.A_p * (params.H_01-x_fluid_eq_60bar_water_01[params.n])  

v_p0 = 0 
v_fluid0=np.zeros(params.n)
y0 = np.concatenate((x_fluid_eq_60bar_water_01, [v_p0], v_fluid0))

p_0_x_60bar_water_01= np.zeros(params.n)
for i in range (params.n):
    p_0_x_60bar_water_01[i] = params.p_t0 + params.rho_water*params.g*(x_fluid_eq_60bar_water_01[params.n]-x_fluid_eq_60bar_water_01[i])


sol_60bar_water_01 = solve_ivp(ms.system_water, [0, T_60], y0, args=(params.m_piston_01,params.H_01, V_t0_60bar_01, V_p0_60bar_01, params.k_s_water, params.m_e_water, params.n, p_p0), atol = 1e-6, rtol=1e-6, max_step = 1e-6)
sol_60bar_water_01.t=np.array(sol_60bar_water_01.t)

x_p_60bar_water_01 = sol_60bar_water_01.y[0]      
x_fluid_60bar_water_01 = sol_60bar_water_01.y[1:params.n+1] 
v_fluid_60bar_water_01 = sol_60bar_water_01.y[params.n+1:2*(params.n+1)] 
a0_values_60bar_water_01 = [ms.piston_acceleration_water(y,params.m_piston_01, params.H_01, V_p0_60bar_01, params.n, params.k_s_water, p_p0) for y in sol_60bar_water_01.y.T]
a0_min_60bar_water_01 = np.min(a0_values_60bar_water_01)
a0_max_60bar_water_01 = np.max(a0_values_60bar_water_01)
v_piston_60bar_water_01 = sol_60bar_water_01.y[params.n+1, :]
v_piston_max_60bar_water_01=np.max(v_piston_60bar_water_01)

all_acc_60bar_water_01 = np.array([
    ms.all_accelerations_water(y, params.m_piston_01, params.H_01, V_t0_60bar_01, V_p0_60bar_01, params.k_s_water, params.m_e_water, params.n, p_p0)
    for y in sol_60bar_water_01.y.T
])

a_piston_60bar_water_01 = all_acc_60bar_water_01[:, 0]      
a_fluid_60bar_water_01  = all_acc_60bar_water_01[:, 1:].T   
V_p_60bar_water_01 = params.A_p * (x_p_60bar_water_01 - params.th_piston_01)
V_t_60bar_water_01 = params.A_p * (params.H_01 - x_fluid_60bar_water_01[params.n-1, :])

p_t_60bar_water_01=[]
for i in range(len(sol_60bar_water_01.t)):
    p_ti=ms.pressure_on_top(sol_60bar_water_01.y[:,i], params.H_01, params.th_piston_01, V_t0_60bar_01, params.n)
    p_t_60bar_water_01.append(p_ti)
p_t_60bar_water_01 = np.array(p_t_60bar_water_01)
p_t_60bar_water_max_01 = np.max(p_t_60bar_water_01)

p_p_60bar_water_01=[]
for i in range(len(sol_60bar_water_01.t)):
    p_pi=ms.pressure_under_piston(sol_60bar_water_01.y[:,i], params.H_01, params.th_piston_01, V_p0_60bar_01, params.n,p_p0)
    p_p_60bar_water_01.append(p_pi)
p_p_60bar_water_01 = np.array(p_p_60bar_water_01)

dx_60bar_water_01 = np.zeros((params.n, len(sol_60bar_water_01.t)))
for j in range(len(sol_60bar_water_01.t)):
    dx_60bar_water_01[0,j]=x_fluid_60bar_water_01[0,j]-x_p_60bar_water_01[j]
    for i in range (1,params.n):
        dx_60bar_water_01[i,j]=x_fluid_60bar_water_01[i,j]-x_fluid_60bar_water_01[i-1,j]
p_60bar_water_01= np.ones((params.n,len(sol_60bar_water_01.t)))*params.p_0
for i in range(params.n):
    for j in range(len(sol_60bar_water_01.t)):
        p_60bar_water_01[i,j]=params.p_0-params.rho_l_water*params.c_water**2*((dx_60bar_water_01[i,j]*params.A_p)-dx_eq_60bar_water_01[i]*params.A_p) /(dx_eq_60bar_water_01[i]*params.A_p)

#M=0.5#
x_fluid_eq_60bar_water_05= ms.equilibrate_fluid_positions(params.n,params.L0,params.k_s_water,params.m_e_water,params.m_piston_05,params.g,params.p_t0,p_p0)
for i in range(len(x_fluid_eq_60bar_water_05)):
    x_fluid_eq_60bar_water_05[i]=x_fluid_eq_60bar_water_05[i]+params.x_air_bottom
x_p0_60bar_water_05=x_fluid_eq_60bar_water_05[0]
offset= x_p0_60bar_water_05 - (params.x_air_bottom+params.th_piston_05)
for i in range(len(x_fluid_eq_60bar_water_05)):
    x_fluid_eq_60bar_water_05[i]=x_fluid_eq_60bar_water_05[i]-offset
x_p0_60bar_water_05=x_fluid_eq_60bar_water_05[0]

dx_eq_60bar_water_05 = np.zeros(params.n)
dx_eq_60bar_water_05[0]=x_fluid_eq_60bar_water_05[1]-x_p0_60bar_water_05
for i in range (1,params.n):
    dx_eq_60bar_water_05[i]=x_fluid_eq_60bar_water_05[i]-x_fluid_eq_60bar_water_05[i-1]

V_p0_60bar_05 = params.A_p * (x_p0_60bar_water_05-params.th_piston_05) 
V_t0_60bar_05 = params.A_p * (params.H_05-x_fluid_eq_60bar_water_05[params.n])  

v_p0 = 0
v_fluid0=np.zeros(params.n)
y0 = np.concatenate((x_fluid_eq_60bar_water_05, [v_p0], v_fluid0))

p_0_x_60bar_water_05= np.zeros(params.n)
for i in range (params.n):
    p_0_x_60bar_water_05[i] = params.p_t0 + params.rho_water*params.g*(x_fluid_eq_60bar_water_05[params.n]-x_fluid_eq_60bar_water_05[i])


sol_60bar_water_05 = solve_ivp(ms.system_water, [0, T_60], y0, args=(params.m_piston_05,params.H_05, V_t0_60bar_05, V_p0_60bar_05, params.k_s_water, params.m_e_water, params.n, p_p0), atol = 1e-6, rtol=1e-6, max_step = 1e-6)
sol_60bar_water_05.t=np.array(sol_60bar_water_05.t)

x_p_60bar_water_05 = sol_60bar_water_05.y[0]      
x_fluid_60bar_water_05 = sol_60bar_water_05.y[1:params.n+1] 
v_fluid_60bar_water_05 = sol_60bar_water_05.y[params.n+1:2*(params.n+1)] 
a0_values_60bar_water_05 = [ms.piston_acceleration_water(y,params.m_piston_05, params.H_05, V_p0_60bar_05, params.n, params.k_s_water, p_p0) for y in sol_60bar_water_05.y.T]
a0_min_60bar_water_05 = np.min(a0_values_60bar_water_05)
a0_max_60bar_water_05 = np.max(a0_values_60bar_water_05)
v_piston_60bar_water_05 = sol_60bar_water_05.y[params.n+1, :]
v_piston_max_60bar_water_05=np.max(v_piston_60bar_water_05)

all_acc_60bar_water_05 = np.array([
    ms.all_accelerations_water(y, params.m_piston_05, params.H_05, V_t0_60bar_05, V_p0_60bar_05, params.k_s_water, params.m_e_water, params.n, p_p0)
    for y in sol_60bar_water_05.y.T
])

a_piston_60bar_water_05 = all_acc_60bar_water_05[:, 0]     
a_fluid_60bar_water_05  = all_acc_60bar_water_05[:, 1:].T   
V_p_60bar_water_05 = params.A_p * (x_p_60bar_water_05 - params.th_piston_05)
V_t_60bar_water_05 = params.A_p * (params.H_05 - x_fluid_60bar_water_05[params.n-1, :])

p_t_60bar_water_05=[]
for i in range(len(sol_60bar_water_05.t)):
    p_ti=ms.pressure_on_top(sol_60bar_water_05.y[:,i], params.H_05, params.th_piston_05, V_t0_60bar_05, params.n)
    p_t_60bar_water_05.append(p_ti)
p_t_60bar_water_05 = np.array(p_t_60bar_water_05)
p_t_60bar_water_max_05 = np.max(p_t_60bar_water_05)

p_p_60bar_water_05=[]
for i in range(len(sol_60bar_water_05.t)):
    p_pi=ms.pressure_under_piston(sol_60bar_water_05.y[:,i], params.H_05, params.th_piston_05, V_p0_60bar_05, params.n,p_p0)
    p_p_60bar_water_05.append(p_pi)
p_p_60bar_water_05 = np.array(p_p_60bar_water_05)

dx_60bar_water_05 = np.zeros((params.n, len(sol_60bar_water_05.t)))
for j in range(len(sol_60bar_water_05.t)):
    dx_60bar_water_05[0,j]=x_fluid_60bar_water_05[0,j]-x_p_60bar_water_05[j]
    for i in range (1,params.n):
        dx_60bar_water_05[i,j]=x_fluid_60bar_water_05[i,j]-x_fluid_60bar_water_05[i-1,j]
p_60bar_water_05= np.ones((params.n,len(sol_60bar_water_05.t)))*params.p_0
for i in range(params.n):
    for j in range(len(sol_60bar_water_05.t)):
        p_60bar_water_05[i,j]=params.p_0-params.rho_l_water*params.c_water**2*((dx_60bar_water_05[i,j]*params.A_p)-dx_eq_60bar_water_05[i]*params.A_p) /(dx_eq_60bar_water_05[i]*params.A_p)

#M=0.9#
x_fluid_eq_60bar_water_09= ms.equilibrate_fluid_positions(params.n,params.L0,params.k_s_water,params.m_e_water,params.m_piston_09,params.g,params.p_t0,p_p0)
for i in range(len(x_fluid_eq_60bar_water_09)):
    x_fluid_eq_60bar_water_09[i]=x_fluid_eq_60bar_water_09[i]+params.x_air_bottom
x_p0_60bar_water_09=x_fluid_eq_60bar_water_09[0]
offset= x_p0_60bar_water_09 - (params.x_air_bottom+params.th_piston_09)
for i in range(len(x_fluid_eq_60bar_water_09)):
    x_fluid_eq_60bar_water_09[i]=x_fluid_eq_60bar_water_09[i]-offset
x_p0_60bar_water_09=x_fluid_eq_60bar_water_09[0]

dx_eq_60bar_water_09 = np.zeros(params.n)
dx_eq_60bar_water_09[0]=x_fluid_eq_60bar_water_09[1]-x_p0_60bar_water_09
for i in range (1,params.n):
    dx_eq_60bar_water_09[i]=x_fluid_eq_60bar_water_09[i]-x_fluid_eq_60bar_water_09[i-1]

V_p0_60bar_09 = params.A_p * (x_p0_60bar_water_09-params.th_piston_09) 
V_t0_60bar_09 = params.A_p * (params.H_09-x_fluid_eq_60bar_water_09[params.n])  

v_p0 = 0 
v_fluid0=np.zeros(params.n)
y0 = np.concatenate((x_fluid_eq_60bar_water_09, [v_p0], v_fluid0))

p_0_x_60bar_water_09= np.zeros(params.n)
for i in range (params.n):
    p_0_x_60bar_water_09[i] = params.p_t0 + params.rho_water*params.g*(x_fluid_eq_60bar_water_09[params.n]-x_fluid_eq_60bar_water_09[i])


sol_60bar_water_09 = solve_ivp(ms.system_water, [0, T_60], y0, args=(params.m_piston_09,params.H_09, V_t0_60bar_09, V_p0_60bar_09, params.k_s_water, params.m_e_water, params.n, p_p0), atol = 1e-6, rtol=1e-6, max_step = 1e-6)
sol_60bar_water_09.t=np.array(sol_60bar_water_09.t)

x_p_60bar_water_09 = sol_60bar_water_09.y[0]      
x_fluid_60bar_water_09 = sol_60bar_water_09.y[1:params.n+1] 
v_fluid_60bar_water_09 = sol_60bar_water_09.y[params.n+1:2*(params.n+1)] 
a0_values_60bar_water_09 = [ms.piston_acceleration_water(y,params.m_piston_09, params.H_09, V_p0_60bar_09, params.n, params.k_s_water, p_p0) for y in sol_60bar_water_09.y.T]
a0_min_60bar_water_09 = np.min(a0_values_60bar_water_09)
a0_max_60bar_water_09 = np.max(a0_values_60bar_water_09)
v_piston_60bar_water_09 = sol_60bar_water_09.y[params.n+1, :]
v_piston_max_60bar_water_09=np.max(v_piston_60bar_water_09)

all_acc_60bar_water_09 = np.array([
    ms.all_accelerations_water(y, params.m_piston_09, params.H_09, V_t0_60bar_09, V_p0_60bar_09, params.k_s_water, params.m_e_water, params.n, p_p0)
    for y in sol_60bar_water_09.y.T
])

a_piston_60bar_water_09 = all_acc_60bar_water_09[:, 0]      
a_fluid_60bar_water_09  = all_acc_60bar_water_09[:, 1:].T  
V_p_60bar_water_09 = params.A_p * (x_p_60bar_water_09 - params.th_piston_09)
V_t_60bar_water_09 = params.A_p * (params.H_09 - x_fluid_60bar_water_09[params.n-1, :])

p_t_60bar_water_09=[]
for i in range(len(sol_60bar_water_09.t)):
    p_ti=ms.pressure_on_top(sol_60bar_water_09.y[:,i], params.H_09, params.th_piston_09, V_t0_60bar_09, params.n)
    p_t_60bar_water_09.append(p_ti)
p_t_60bar_water_09 = np.array(p_t_60bar_water_09)
p_t_60bar_water_max_09 = np.max(p_t_60bar_water_09)

p_p_60bar_water_09=[]
for i in range(len(sol_60bar_water_09.t)):
    p_pi=ms.pressure_under_piston(sol_60bar_water_09.y[:,i], params.H_09, params.th_piston_09, V_p0_60bar_09, params.n,p_p0)
    p_p_60bar_water_09.append(p_pi)
p_p_60bar_water_09 = np.array(p_p_60bar_water_09)

dx_60bar_water_09 = np.zeros((params.n, len(sol_60bar_water_09.t)))
for j in range(len(sol_60bar_water_09.t)):
    dx_60bar_water_09[0,j]=x_fluid_60bar_water_09[0,j]-x_p_60bar_water_09[j]
    for i in range (1,params.n):
        dx_60bar_water_09[i,j]=x_fluid_60bar_water_09[i,j]-x_fluid_60bar_water_09[i-1,j]
p_60bar_water_09= np.ones((params.n,len(sol_60bar_water_09.t)))*params.p_0
for i in range(params.n):
    for j in range(len(sol_60bar_water_09.t)):
        p_60bar_water_09[i,j]=params.p_0-params.rho_l_water*params.c_water**2*((dx_60bar_water_09[i,j]*params.A_p)-dx_eq_60bar_water_09[i]*params.A_p) /(dx_eq_60bar_water_09[i]*params.A_p)

#M=1#
x_fluid_eq_60bar_water_1= ms.equilibrate_fluid_positions(params.n,params.L0,params.k_s_water,params.m_e_water,params.m_piston_1,params.g,params.p_t0,p_p0)
for i in range(len(x_fluid_eq_60bar_water_1)):
    x_fluid_eq_60bar_water_1[i]=x_fluid_eq_60bar_water_1[i]+params.x_air_bottom
x_p0_60bar_water_1=x_fluid_eq_60bar_water_1[0]
offset= x_p0_60bar_water_1 - (params.x_air_bottom+params.th_piston_1)
for i in range(len(x_fluid_eq_60bar_water_1)):
    x_fluid_eq_60bar_water_1[i]=x_fluid_eq_60bar_water_1[i]-offset
x_p0_60bar_water_1=x_fluid_eq_60bar_water_1[0]

dx_eq_60bar_water_1 = np.zeros(params.n)
dx_eq_60bar_water_1[0]=x_fluid_eq_60bar_water_1[1]-x_p0_60bar_water_1
for i in range (1,params.n):
    dx_eq_60bar_water_1[i]=x_fluid_eq_60bar_water_1[i]-x_fluid_eq_60bar_water_1[i-1]

V_p0_60bar_1 = params.A_p * (x_p0_60bar_water_1-params.th_piston_1) 
V_t0_60bar_1 = params.A_p * (params.H_1-x_fluid_eq_60bar_water_1[params.n])  

v_p0 = 0 
v_fluid0=np.zeros(params.n)
y0 = np.concatenate((x_fluid_eq_60bar_water_1, [v_p0], v_fluid0))

p_0_x_60bar_water_1= np.zeros(params.n)
for i in range (params.n):
    p_0_x_60bar_water_1[i] = params.p_t0 + params.rho_water*params.g*(x_fluid_eq_60bar_water_1[params.n]-x_fluid_eq_60bar_water_1[i])


sol_60bar_water_1 = solve_ivp(ms.system_water, [0, T_60], y0, args=(params.m_piston_1,params.H_1, V_t0_60bar_1, V_p0_60bar_1, params.k_s_water, params.m_e_water, params.n, p_p0), atol = 1e-6, rtol=1e-6, max_step = 1e-6)
sol_60bar_water_1.t=np.array(sol_60bar_water_1.t)

x_p_60bar_water_1 = sol_60bar_water_1.y[0]     
x_fluid_60bar_water_1 = sol_60bar_water_1.y[1:params.n+1] 
v_fluid_60bar_water_1 = sol_60bar_water_1.y[params.n+1:2*(params.n+1)] 
a0_values_60bar_water_1 = [ms.piston_acceleration_water(y,params.m_piston_1, params.H_1, V_p0_60bar_1, params.n, params.k_s_water, p_p0) for y in sol_60bar_water_1.y.T]
a0_min_60bar_water_1 = np.min(a0_values_60bar_water_1)
a0_max_60bar_water_1 = np.max(a0_values_60bar_water_1)
v_piston_60bar_water_1 = sol_60bar_water_1.y[params.n+1, :]
v_piston_max_60bar_water_1=np.max(v_piston_60bar_water_1)

all_acc_60bar_water_1 = np.array([
    ms.all_accelerations_water(y, params.m_piston_1, params.H_1, V_t0_60bar_1, V_p0_60bar_1, params.k_s_water, params.m_e_water, params.n, p_p0)
    for y in sol_60bar_water_1.y.T
])

a_piston_60bar_water_1 = all_acc_60bar_water_1[:, 0]      
a_fluid_60bar_water_1  = all_acc_60bar_water_1[:, 1:].T   
V_p_60bar_water_1 = params.A_p * (x_p_60bar_water_1 - params.th_piston_1)
V_t_60bar_water_1 = params.A_p * (params.H_1 - x_fluid_60bar_water_1[params.n-1, :])

p_t_60bar_water_1=[]
for i in range(len(sol_60bar_water_1.t)):
    p_ti=ms.pressure_on_top(sol_60bar_water_1.y[:,i], params.H_1, params.th_piston_1, V_t0_60bar_1, params.n)
    p_t_60bar_water_1.append(p_ti)
p_t_60bar_water_1 = np.array(p_t_60bar_water_1)
p_t_60bar_water_max_1 = np.max(p_t_60bar_water_1)

p_p_60bar_water_1=[]
for i in range(len(sol_60bar_water_1.t)):
    p_pi=ms.pressure_under_piston(sol_60bar_water_1.y[:,i], params.H_1, params.th_piston_1, V_p0_60bar_1, params.n,p_p0)
    p_p_60bar_water_1.append(p_pi)
p_p_60bar_water_1 = np.array(p_p_60bar_water_1)

dx_60bar_water_1 = np.zeros((params.n, len(sol_60bar_water_1.t)))
for j in range(len(sol_60bar_water_1.t)):
    dx_60bar_water_1[0,j]=x_fluid_60bar_water_1[0,j]-x_p_60bar_water_1[j]
    for i in range (1,params.n):
        dx_60bar_water_1[i,j]=x_fluid_60bar_water_1[i,j]-x_fluid_60bar_water_1[i-1,j]
p_60bar_water_1= np.ones((params.n,len(sol_60bar_water_1.t)))*params.p_0
for i in range(params.n):
    for j in range(len(sol_60bar_water_1.t)):
        p_60bar_water_1[i,j]=params.p_0-params.rho_l_water*params.c_water**2*((dx_60bar_water_1[i,j]*params.A_p)-dx_eq_60bar_water_1[i]*params.A_p) /(dx_eq_60bar_water_1[i]*params.A_p)

#M=1.5#
x_fluid_eq_60bar_water_15= ms.equilibrate_fluid_positions(params.n,params.L0,params.k_s_water,params.m_e_water,params.m_piston_15,params.g,params.p_t0,p_p0)
for i in range(len(x_fluid_eq_60bar_water_15)):
    x_fluid_eq_60bar_water_15[i]=x_fluid_eq_60bar_water_15[i]+params.x_air_bottom
x_p0_60bar_water_15=x_fluid_eq_60bar_water_15[0]
offset= x_p0_60bar_water_15 - (params.x_air_bottom+params.th_piston_15)
for i in range(len(x_fluid_eq_60bar_water_15)):
    x_fluid_eq_60bar_water_15[i]=x_fluid_eq_60bar_water_15[i]-offset
x_p0_60bar_water_15=x_fluid_eq_60bar_water_15[0]

dx_eq_60bar_water_15 = np.zeros(params.n)
dx_eq_60bar_water_15[0]=x_fluid_eq_60bar_water_15[1]-x_p0_60bar_water_15
for i in range (1,params.n):
    dx_eq_60bar_water_15[i]=x_fluid_eq_60bar_water_15[i]-x_fluid_eq_60bar_water_15[i-1]

V_p0_60bar_15 = params.A_p * (x_p0_60bar_water_15-params.th_piston_15)
V_t0_60bar_15 = params.A_p * (params.H_15-x_fluid_eq_60bar_water_15[params.n]) 

v_p0 = 0 
v_fluid0=np.zeros(params.n)
y0 = np.concatenate((x_fluid_eq_60bar_water_15, [v_p0], v_fluid0))

p_0_x_60bar_water_15= np.zeros(params.n)
for i in range (params.n):
    p_0_x_60bar_water_15[i] = params.p_t0 + params.rho_water*params.g*(x_fluid_eq_60bar_water_15[params.n]-x_fluid_eq_60bar_water_15[i])


sol_60bar_water_15 = solve_ivp(ms.system_water, [0, T_60], y0, args=(params.m_piston_15,params.H_15, V_t0_60bar_15, V_p0_60bar_15, params.k_s_water, params.m_e_water, params.n, p_p0), atol = 1e-6, rtol=1e-6, max_step = 1e-6)
sol_60bar_water_15.t=np.array(sol_60bar_water_15.t)

x_p_60bar_water_15 = sol_60bar_water_15.y[0]      
x_fluid_60bar_water_15 = sol_60bar_water_15.y[1:params.n+1] 
v_fluid_60bar_water_15 = sol_60bar_water_15.y[params.n+1:2*(params.n+1)]
a0_values_60bar_water_15 = [ms.piston_acceleration_water(y,params.m_piston_15, params.H_15, V_p0_60bar_15, params.n, params.k_s_water, p_p0) for y in sol_60bar_water_15.y.T]
a0_min_60bar_water_15 = np.min(a0_values_60bar_water_15)
a0_max_60bar_water_15 = np.max(a0_values_60bar_water_15)
v_piston_60bar_water_15 = sol_60bar_water_15.y[params.n+1, :]
v_piston_max_60bar_water_15=np.max(v_piston_60bar_water_15)

all_acc_60bar_water_15 = np.array([
    ms.all_accelerations_water(y, params.m_piston_15, params.H_15, V_t0_60bar_15, V_p0_60bar_15, params.k_s_water, params.m_e_water, params.n, p_p0)
    for y in sol_60bar_water_15.y.T
])

a_piston_60bar_water_15 = all_acc_60bar_water_15[:, 0]      
a_fluid_60bar_water_15  = all_acc_60bar_water_15[:, 1:].T   
V_p_60bar_water_15 = params.A_p * (x_p_60bar_water_15 - params.th_piston_15)
V_t_60bar_water_15 = params.A_p * (params.H_15 - x_fluid_60bar_water_15[params.n-1, :])

p_t_60bar_water_15=[]
for i in range(len(sol_60bar_water_15.t)):
    p_ti=ms.pressure_on_top(sol_60bar_water_15.y[:,i], params.H_15, params.th_piston_15, V_t0_60bar_15, params.n)
    p_t_60bar_water_15.append(p_ti)
p_t_60bar_water_15 = np.array(p_t_60bar_water_15)
p_t_60bar_water_max_15 = np.max(p_t_60bar_water_15)

p_p_60bar_water_15=[]
for i in range(len(sol_60bar_water_15.t)):
    p_pi=ms.pressure_under_piston(sol_60bar_water_15.y[:,i], params.H_15, params.th_piston_15, V_p0_60bar_15, params.n,p_p0)
    p_p_60bar_water_15.append(p_pi)
p_p_60bar_water_15 = np.array(p_p_60bar_water_15)

dx_60bar_water_15 = np.zeros((params.n, len(sol_60bar_water_15.t)))
for j in range(len(sol_60bar_water_15.t)):
    dx_60bar_water_15[0,j]=x_fluid_60bar_water_15[0,j]-x_p_60bar_water_15[j]
    for i in range (1,params.n):
        dx_60bar_water_15[i,j]=x_fluid_60bar_water_15[i,j]-x_fluid_60bar_water_15[i-1,j]
p_60bar_water_15= np.ones((params.n,len(sol_60bar_water_15.t)))*params.p_0
for i in range(params.n):
    for j in range(len(sol_60bar_water_15.t)):
        p_60bar_water_15[i,j]=params.p_0-params.rho_l_water*params.c_water**2*((dx_60bar_water_15[i,j]*params.A_p)-dx_eq_60bar_water_15[i]*params.A_p) /(dx_eq_60bar_water_15[i]*params.A_p)

#M=2#
x_fluid_eq_60bar_water_2= ms.equilibrate_fluid_positions(params.n,params.L0,params.k_s_water,params.m_e_water,params.m_piston_2,params.g,params.p_t0,p_p0)
for i in range(len(x_fluid_eq_60bar_water_2)):
    x_fluid_eq_60bar_water_2[i]=x_fluid_eq_60bar_water_2[i]+params.x_air_bottom
x_p0_60bar_water_2=x_fluid_eq_60bar_water_2[0]
offset= x_p0_60bar_water_2 - (params.x_air_bottom+params.th_piston_2)
for i in range(len(x_fluid_eq_60bar_water_2)):
    x_fluid_eq_60bar_water_2[i]=x_fluid_eq_60bar_water_2[i]-offset
x_p0_60bar_water_2=x_fluid_eq_60bar_water_2[0]

dx_eq_60bar_water_2 = np.zeros(params.n)
dx_eq_60bar_water_2[0]=x_fluid_eq_60bar_water_2[1]-x_p0_60bar_water_2
for i in range (1,params.n):
    dx_eq_60bar_water_2[i]=x_fluid_eq_60bar_water_2[i]-x_fluid_eq_60bar_water_2[i-1]

V_p0_60bar_2 = params.A_p * (x_p0_60bar_water_2-params.th_piston_2) 
V_t0_60bar_2 = params.A_p * (params.H_2-x_fluid_eq_60bar_water_2[params.n])  

v_p0 = 0 
v_fluid0=np.zeros(params.n)
y0 = np.concatenate((x_fluid_eq_60bar_water_2, [v_p0], v_fluid0))

p_0_x_60bar_water_2= np.zeros(params.n)
for i in range (params.n):
    p_0_x_60bar_water_2[i] = params.p_t0 + params.rho_water*params.g*(x_fluid_eq_60bar_water_2[params.n]-x_fluid_eq_60bar_water_2[i])


sol_60bar_water_2 = solve_ivp(ms.system_water, [0, T_60], y0, args=(params.m_piston_2,params.H_2, V_t0_60bar_2, V_p0_60bar_2, params.k_s_water, params.m_e_water, params.n, p_p0), atol = 1e-6, rtol=1e-6, max_step = 1e-6)
sol_60bar_water_2.t=np.array(sol_60bar_water_2.t)

x_p_60bar_water_2 = sol_60bar_water_2.y[0]      
x_fluid_60bar_water_2 = sol_60bar_water_2.y[1:params.n+1] 
v_fluid_60bar_water_2 = sol_60bar_water_2.y[params.n+1:2*(params.n+1)] 
a0_values_60bar_water_2 = [ms.piston_acceleration_water(y,params.m_piston_2, params.H_2, V_p0_60bar_2, params.n, params.k_s_water, p_p0) for y in sol_60bar_water_2.y.T]
a0_min_60bar_water_2 = np.min(a0_values_60bar_water_2)
a0_max_60bar_water_2 = np.max(a0_values_60bar_water_2)
v_piston_60bar_water_2 = sol_60bar_water_2.y[params.n+1, :]
v_piston_max_60bar_water_2=np.max(v_piston_60bar_water_2)

all_acc_60bar_water_2 = np.array([
    ms.all_accelerations_water(y, params.m_piston_2, params.H_2, V_t0_60bar_2, V_p0_60bar_2, params.k_s_water, params.m_e_water, params.n, p_p0)
    for y in sol_60bar_water_2.y.T
])

a_piston_60bar_water_2 = all_acc_60bar_water_2[:, 0]     
a_fluid_60bar_water_2  = all_acc_60bar_water_2[:, 1:].T   
V_p_60bar_water_2 = params.A_p * (x_p_60bar_water_2 - params.th_piston_2)
V_t_60bar_water_2 = params.A_p * (params.H_2 - x_fluid_60bar_water_2[params.n-1, :])

p_t_60bar_water_2=[]
for i in range(len(sol_60bar_water_2.t)):
    p_ti=ms.pressure_on_top(sol_60bar_water_2.y[:,i], params.H_2, params.th_piston_2, V_t0_60bar_2, params.n)
    p_t_60bar_water_2.append(p_ti)
p_t_60bar_water_2 = np.array(p_t_60bar_water_2)
p_t_60bar_water_max_2 = np.max(p_t_60bar_water_2)

p_p_60bar_water_2=[]
for i in range(len(sol_60bar_water_2.t)):
    p_pi=ms.pressure_under_piston(sol_60bar_water_2.y[:,i], params.H_2, params.th_piston_2, V_p0_60bar_2, params.n,p_p0)
    p_p_60bar_water_2.append(p_pi)
p_p_60bar_water_2 = np.array(p_p_60bar_water_2)

dx_60bar_water_2 = np.zeros((params.n, len(sol_60bar_water_2.t)))
for j in range(len(sol_60bar_water_2.t)):
    dx_60bar_water_2[0,j]=x_fluid_60bar_water_2[0,j]-x_p_60bar_water_2[j]
    for i in range (1,params.n):
        dx_60bar_water_2[i,j]=x_fluid_60bar_water_2[i,j]-x_fluid_60bar_water_2[i-1,j]
p_60bar_water_2= np.ones((params.n,len(sol_60bar_water_2.t)))*params.p_0
for i in range(params.n):
    for j in range(len(sol_60bar_water_2.t)):
        p_60bar_water_2[i,j]=params.p_0-params.rho_l_water*params.c_water**2*((dx_60bar_water_2[i,j]*params.A_p)-dx_eq_60bar_water_2[i]*params.A_p) /(dx_eq_60bar_water_2[i]*params.A_p)

#no piston#
x_fluid_eq_60bar_water_no_piston = ms.equilibrate_fluid_positions_no_piston(params.n, params.L0, params.k_s_water, params.m_e_water, params.g, params.p_t0, p_p0)
x_0_60bar_water_no_piston = x_fluid_eq_60bar_water_no_piston[0]

dx_eq_60bar_water_no_piston = np.zeros(params.n)
dx_eq_60bar_water_no_piston[0] = x_fluid_eq_60bar_water_no_piston[1] - x_0_60bar_water_no_piston
for i in range(1, params.n):
    dx_eq_60bar_water_no_piston[i] = x_fluid_eq_60bar_water_no_piston[i] - x_fluid_eq_60bar_water_no_piston[i - 1]

V_p0 = params.A_p * x_0_60bar_water_no_piston  
V_t0 = params.A_p * (params.H_no_piston - x_fluid_eq_60bar_water_no_piston[params.n])  

v_0 = 0  
v_fluid0 = np.zeros(params.n)
y0 = np.concatenate((x_fluid_eq_60bar_water_no_piston, [v_0], v_fluid0))

sol_60bar_water_no_piston = solve_ivp(ms.system_water_no_piston, [0, T_60], y0, args=(V_t0,V_p0, p_p0), atol=1e-6, rtol=1e-6, max_step=1e-6)
sol_60bar_water_no_piston.t=np.array(sol_60bar_water_no_piston.t)

x_0_60bar_water_no_piston = sol_60bar_water_no_piston.y[0]  
x_fluid_60bar_water_no_piston = sol_60bar_water_no_piston.y[1:params.n + 1]  
v_fluid_60bar_water_no_piston = sol_60bar_water_no_piston.y[params.n + 1:2 * (params.n + 1)]  
a0_values_60bar_water_no_piston = [ms.liquid_acceleration_water(y, params.n, params.k_s_water, params.m_e_water, params.L0, V_p0, p_p0) for y in sol_60bar_water_no_piston.y.T]
a0_min_60bar_water_no_piston = np.min(a0_values_60bar_water_no_piston)
a0_max_60bar_water_no_piston = np.max(a0_values_60bar_water_no_piston)
v_piston_60bar_water_no_piston = sol_60bar_water_no_piston.y[params.n + 1, :]
v_piston_max_60bar_water_no_piston = np.max(v_piston_60bar_water_no_piston)

all_acc_60bar_water_no_piston = np.array([
    ms.all_accelerations_water_no_piston(y, V_t0, V_p0, p_p0)
    for y in sol_60bar_water_no_piston.y.T
])  

a_fluid_60bar_water_no_piston = all_acc_60bar_water_no_piston.T 
V_p_60bar_water_no_piston = params.A_p * x_0_60bar_water_no_piston
V_t_60bar_water_no_piston = params.A_p * (params.H_no_piston - x_fluid_60bar_water_no_piston[params.n-1, :])

p_t_60bar_water_no_piston = []
for i in range(len(sol_60bar_water_no_piston.t)):
    p_ti = ms.pressure_on_top_no_piston(sol_60bar_water_no_piston.y[:, i], V_t0)
    p_t_60bar_water_no_piston.append(p_ti)


p_p_60bar_water_no_piston = []
for i in range(len(sol_60bar_water_no_piston.t)):
    p_pi = ms.pressure_gas_bottom(sol_60bar_water_no_piston.y[:, i], V_p0, p_p0)
    p_p_60bar_water_no_piston.append(p_pi)


dx_60bar_water_no_piston = np.zeros((params.n, len(sol_60bar_water_no_piston.t)))
for j in range(len(sol_60bar_water_no_piston.t)):
    dx_60bar_water_no_piston[0, j] = x_fluid_60bar_water_no_piston[0, j] - x_0_60bar_water_no_piston[j]
    for i in range(1, params.n):
        dx_60bar_water_no_piston[i, j] = x_fluid_60bar_water_no_piston[i, j] - x_fluid_60bar_water_no_piston[i - 1, j]

p_60bar_water_no_piston = np.ones((params.n, len(sol_60bar_water_no_piston.t))) * params.p_0
for i in range(params.n):
    for j in range(len(sol_60bar_water_no_piston.t)):
        p_60bar_water_no_piston[i, j] = params.p_0 - params.rho_l_water * params.c_water**2 * ((dx_60bar_water_no_piston[i, j] * params.A_p) - dx_eq_60bar_water_no_piston[i] * params.A_p) / (dx_eq_60bar_water_no_piston[i] * params.A_p)

#lithium#
#M=1.69#
x_fluid_eq_60bar_lithium= ms.equilibrate_fluid_positions(params.n,params.L0,params.k_s_lithium,params.m_e_lithium,params.m_piston_ref,params.g,params.p_t0,p_p0)
for i in range(len(x_fluid_eq_60bar_lithium)):
    x_fluid_eq_60bar_lithium[i]=x_fluid_eq_60bar_lithium[i]+params.x_air_bottom
x_p0_60bar_lithium=x_fluid_eq_60bar_lithium[0]
offset= x_p0_60bar_lithium - 0.25
for i in range(len(x_fluid_eq_60bar_lithium)):
    x_fluid_eq_60bar_lithium[i]=x_fluid_eq_60bar_lithium[i]-offset
x_p0_60bar_lithium=x_fluid_eq_60bar_lithium[0]

dx_eq_60bar_lithium= np.zeros(params.n)
dx_eq_60bar_lithium[0]=x_fluid_eq_60bar_lithium[1]-x_p0_60bar_lithium
for i in range (1,params.n):
    dx_eq_60bar_lithium[i]=x_fluid_eq_60bar_lithium[i]-x_fluid_eq_60bar_lithium[i-1]

V_p0 = params.A_p * (x_p0_60bar_lithium-params.th_piston_ref) 
V_t0 = params.A_p * (params.H_ref-x_fluid_eq_60bar_lithium[params.n])  

v_p0 = 0 
v_fluid0=np.zeros(params.n)
y0 = np.concatenate((x_fluid_eq_60bar_lithium, [v_p0], v_fluid0))


sol_60bar_lithium = solve_ivp(ms.system_lith, [0, T_60], y0, args=(params.m_piston_ref,params.H_ref, V_t0, V_p0, params.n, p_p0), atol = 1e-6, rtol=1e-6, max_step = 1e-6)


x_p_60bar_lithium = sol_60bar_lithium.y[0]      
x_fluid_60bar_lithium = sol_60bar_lithium.y[1:params.n+1]
v_fluid_60bar_lithium = sol_60bar_lithium.y[params.n+1:2*(params.n+1)] 
a0_values_60bar_lithium = [ms.piston_acceleration_lith(y, params.m_piston_ref, V_p0, p_p0) for y in sol_60bar_lithium.y.T]
a0_min_60bar_lithium = np.min(a0_values_60bar_lithium)
a0_max_60bar_lithium = np.max(a0_values_60bar_lithium)
v_piston_60bar_lithium = sol_60bar_lithium.y[params.n+1, :]
v_piston_max_60bar_lithium=np.max(v_piston_60bar_lithium)


p_t_60bar_lithium=[]
for i in range(len(sol_60bar_lithium.t)):
    p_ti=ms.pressure_on_top(sol_60bar_lithium.y[:,i], params.H_ref, params.th_piston_ref, V_t0, params.n)
    p_t_60bar_lithium.append(p_ti)


p_p_60bar_lithium=[]
for i in range(len(sol_60bar_lithium.t)):
    p_pi=ms.pressure_under_piston(sol_60bar_lithium.y[:,i], params.H_ref, params.th_piston_ref, V_p0, params.n,p_p0)
    p_p_60bar_lithium.append(p_pi)


dx_60bar_lithium = np.zeros((params.n, len(sol_60bar_lithium.t)))
for j in range(len(sol_60bar_lithium.t)):
    dx_60bar_lithium[0,j]=x_fluid_60bar_lithium[0,j]-x_p_60bar_lithium[j]
    for i in range (1,params.n):
        dx_60bar_lithium[i,j]=x_fluid_60bar_lithium[i,j]-x_fluid_60bar_lithium[i-1,j]
p_60bar_lithium= np.ones((params.n,len(sol_60bar_lithium.t)))*params.p_0
for i in range(params.n):
    for j in range(len(sol_60bar_lithium.t)):
        p_60bar_lithium[i,j]=params.p_0-params.rho_l_lithium*params.c_lithium**2*((dx_60bar_lithium[i,j]*params.A_p)-dx_eq_60bar_lithium[i]*params.A_p) /(dx_eq_60bar_lithium[i]*params.A_p)

#Driving pressure: 7.5MPa#
p_p0 = 75e5
T_75 = 0.012  

#water#
#M=0.1#
x_fluid_eq_75bar_water_01= ms.equilibrate_fluid_positions(params.n,params.L0,params.k_s_water,params.m_e_water,params.m_piston_01,params.g,params.p_t0,p_p0)
for i in range(len(x_fluid_eq_75bar_water_01)):
    x_fluid_eq_75bar_water_01[i]=x_fluid_eq_75bar_water_01[i]+params.x_air_bottom
x_p0_75bar_water_01=x_fluid_eq_75bar_water_01[0]
offset= x_p0_75bar_water_01 - (params.x_air_bottom+params.th_piston_01)
for i in range(len(x_fluid_eq_75bar_water_01)):
    x_fluid_eq_75bar_water_01[i]=x_fluid_eq_75bar_water_01[i]-offset
x_p0_75bar_water_01=x_fluid_eq_75bar_water_01[0]

dx_eq_75bar_water_01 = np.zeros(params.n)
dx_eq_75bar_water_01[0]=x_fluid_eq_75bar_water_01[1]-x_p0_75bar_water_01
for i in range (1,params.n):
    dx_eq_75bar_water_01[i]=x_fluid_eq_75bar_water_01[i]-x_fluid_eq_75bar_water_01[i-1]

V_p0_75bar_01 = params.A_p * (x_p0_75bar_water_01-params.th_piston_01) 
V_t0_75bar_01 = params.A_p * (params.H_01-x_fluid_eq_75bar_water_01[params.n])  

v_p0 = 0 
v_fluid0=np.zeros(params.n)
y0 = np.concatenate((x_fluid_eq_75bar_water_01, [v_p0], v_fluid0))

p_0_x_75bar_water_01= np.zeros(params.n)
for i in range (params.n):
    p_0_x_75bar_water_01[i] = params.p_t0 + params.rho_water*params.g*(x_fluid_eq_75bar_water_01[params.n]-x_fluid_eq_75bar_water_01[i])

sol_75bar_water_01 = solve_ivp(ms.system_water, [0, T_75], y0, args=(params.m_piston_01,params.H_01, V_t0_75bar_01, V_p0_75bar_01, params.k_s_water, params.m_e_water, params.n, p_p0), atol = 1e-6, rtol=1e-6, max_step = 1e-6)
sol_75bar_water_01.t=np.array(sol_75bar_water_01.t)

x_p_75bar_water_01 = sol_75bar_water_01.y[0]      
x_fluid_75bar_water_01 = sol_75bar_water_01.y[1:params.n+1] 
v_fluid_75bar_water_01 = sol_75bar_water_01.y[params.n+1:2*(params.n+1)] 
a0_values_75bar_water_01 = [ms.piston_acceleration_water(y,params.m_piston_01, params.H_01, V_p0_75bar_01, params.n, params.k_s_water, p_p0) for y in sol_75bar_water_01.y.T]
a0_min_75bar_water_01 = np.min(a0_values_75bar_water_01)
a0_max_75bar_water_01 = np.max(a0_values_75bar_water_01)
v_piston_75bar_water_01 = sol_75bar_water_01.y[params.n+1, :]
v_piston_max_75bar_water_01=np.max(v_piston_75bar_water_01)

all_acc_75bar_water_01 = np.array([
    ms.all_accelerations_water(y, params.m_piston_01, params.H_01, V_t0_75bar_01, V_p0_75bar_01, params.k_s_water, params.m_e_water, params.n, p_p0)
    for y in sol_75bar_water_01.y.T
])

a_piston_75bar_water_01 = all_acc_75bar_water_01[:, 0]      
a_fluid_75bar_water_01  = all_acc_75bar_water_01[:, 1:].T   
V_p_75bar_water_01 = params.A_p * (x_p_75bar_water_01 - params.th_piston_01)
V_t_75bar_water_01 = params.A_p * (params.H_01 - x_fluid_75bar_water_01[params.n-1, :])

p_t_75bar_water_01=[]
for i in range(len(sol_75bar_water_01.t)):
    p_ti=ms.pressure_on_top(sol_75bar_water_01.y[:,i], params.H_01, params.th_piston_01, V_t0_75bar_01, params.n)
    p_t_75bar_water_01.append(p_ti)
p_t_75bar_water_01 = np.array(p_t_75bar_water_01)
p_t_75bar_water_max_01 = np.max(p_t_75bar_water_01)

p_p_75bar_water_01=[]
for i in range(len(sol_75bar_water_01.t)):
    p_pi=ms.pressure_under_piston(sol_75bar_water_01.y[:,i], params.H_01, params.th_piston_01, V_p0_75bar_01, params.n,p_p0)
    p_p_75bar_water_01.append(p_pi)
p_p_75bar_water_01 = np.array(p_p_75bar_water_01)

dx_75bar_water_01 = np.zeros((params.n, len(sol_75bar_water_01.t)))
for j in range(len(sol_75bar_water_01.t)):
    dx_75bar_water_01[0,j]=x_fluid_75bar_water_01[0,j]-x_p_75bar_water_01[j]
    for i in range (1,params.n):
        dx_75bar_water_01[i,j]=x_fluid_75bar_water_01[i,j]-x_fluid_75bar_water_01[i-1,j]
p_75bar_water_01= np.ones((params.n,len(sol_75bar_water_01.t)))*params.p_0
for i in range(params.n):
    for j in range(len(sol_75bar_water_01.t)):
        p_75bar_water_01[i,j]=params.p_0-params.rho_l_water*params.c_water**2*((dx_75bar_water_01[i,j]*params.A_p)-dx_eq_75bar_water_01[i]*params.A_p) /(dx_eq_75bar_water_01[i]*params.A_p)

#M=0.5#
x_fluid_eq_75bar_water_05= ms.equilibrate_fluid_positions(params.n,params.L0,params.k_s_water,params.m_e_water,params.m_piston_05,params.g,params.p_t0,p_p0)
for i in range(len(x_fluid_eq_75bar_water_05)):
    x_fluid_eq_75bar_water_05[i]=x_fluid_eq_75bar_water_05[i]+params.x_air_bottom
x_p0_75bar_water_05=x_fluid_eq_75bar_water_05[0]
offset= x_p0_75bar_water_05 - (params.x_air_bottom+params.th_piston_05)
for i in range(len(x_fluid_eq_75bar_water_05)):
    x_fluid_eq_75bar_water_05[i]=x_fluid_eq_75bar_water_05[i]-offset
x_p0_75bar_water_05=x_fluid_eq_75bar_water_05[0]

dx_eq_75bar_water_05 = np.zeros(params.n)
dx_eq_75bar_water_05[0]=x_fluid_eq_75bar_water_05[1]-x_p0_75bar_water_05
for i in range (1,params.n):
    dx_eq_75bar_water_05[i]=x_fluid_eq_75bar_water_05[i]-x_fluid_eq_75bar_water_05[i-1]

V_p0_75bar_05 = params.A_p * (x_p0_75bar_water_05-params.th_piston_05) 
V_t0_75bar_05 = params.A_p * (params.H_05-x_fluid_eq_75bar_water_05[params.n])  

v_p0 = 0 
v_fluid0=np.zeros(params.n)
y0 = np.concatenate((x_fluid_eq_75bar_water_05, [v_p0], v_fluid0))

p_0_x_75bar_water_05= np.zeros(params.n)
for i in range (params.n):
    p_0_x_75bar_water_05[i] = params.p_t0 + params.rho_water*params.g*(x_fluid_eq_75bar_water_05[params.n]-x_fluid_eq_75bar_water_05[i])


sol_75bar_water_05 = solve_ivp(ms.system_water, [0, T_75], y0, args=(params.m_piston_05,params.H_05, V_t0_75bar_05, V_p0_75bar_05, params.k_s_water, params.m_e_water, params.n, p_p0), atol = 1e-6, rtol=1e-6, max_step = 1e-6)
sol_75bar_water_05.t=np.array(sol_75bar_water_05.t)

x_p_75bar_water_05 = sol_75bar_water_05.y[0]     
x_fluid_75bar_water_05 = sol_75bar_water_05.y[1:params.n+1]
v_fluid_75bar_water_05 = sol_75bar_water_05.y[params.n+1:2*(params.n+1)] 
a0_values_75bar_water_05 = [ms.piston_acceleration_water(y,params.m_piston_05, params.H_05, V_p0_75bar_05, params.n, params.k_s_water, p_p0) for y in sol_75bar_water_05.y.T]
a0_min_75bar_water_05 = np.min(a0_values_75bar_water_05)
a0_max_75bar_water_05 = np.max(a0_values_75bar_water_05)
v_piston_75bar_water_05 = sol_75bar_water_05.y[params.n+1, :]
v_piston_max_75bar_water_05=np.max(v_piston_75bar_water_05)

all_acc_75bar_water_05 = np.array([
    ms.all_accelerations_water(y, params.m_piston_05, params.H_05, V_t0_75bar_05, V_p0_75bar_05, params.k_s_water, params.m_e_water, params.n, p_p0)
    for y in sol_75bar_water_05.y.T
])

a_piston_75bar_water_05 = all_acc_75bar_water_05[:, 0]      
a_fluid_75bar_water_05  = all_acc_75bar_water_05[:, 1:].T  
V_p_75bar_water_05 = params.A_p * (x_p_75bar_water_05 - params.th_piston_05)
V_t_75bar_water_05 = params.A_p * (params.H_05 - x_fluid_75bar_water_05[params.n-1, :])

p_t_75bar_water_05=[]
for i in range(len(sol_75bar_water_05.t)):
    p_ti=ms.pressure_on_top(sol_75bar_water_05.y[:,i], params.H_05, params.th_piston_05, V_t0_75bar_05, params.n)
    p_t_75bar_water_05.append(p_ti)
p_t_75bar_water_05 = np.array(p_t_75bar_water_05)
p_t_75bar_water_max_05 = np.max(p_t_75bar_water_05)

p_p_75bar_water_05=[]
for i in range(len(sol_75bar_water_05.t)):
    p_pi=ms.pressure_under_piston(sol_75bar_water_05.y[:,i], params.H_05, params.th_piston_05, V_p0_75bar_05, params.n,p_p0)
    p_p_75bar_water_05.append(p_pi)
p_p_75bar_water_05 = np.array(p_p_75bar_water_05)

dx_75bar_water_05 = np.zeros((params.n, len(sol_75bar_water_05.t)))
for j in range(len(sol_75bar_water_05.t)):
    dx_75bar_water_05[0,j]=x_fluid_75bar_water_05[0,j]-x_p_75bar_water_05[j]
    for i in range (1,params.n):
        dx_75bar_water_05[i,j]=x_fluid_75bar_water_05[i,j]-x_fluid_75bar_water_05[i-1,j]
p_75bar_water_05= np.ones((params.n,len(sol_75bar_water_05.t)))*params.p_0
for i in range(params.n):
    for j in range(len(sol_75bar_water_05.t)):
        p_75bar_water_05[i,j]=params.p_0-params.rho_l_water*params.c_water**2*((dx_75bar_water_05[i,j]*params.A_p)-dx_eq_75bar_water_05[i]*params.A_p) /(dx_eq_75bar_water_05[i]*params.A_p)

#M=0.9#
x_fluid_eq_75bar_water_09= ms.equilibrate_fluid_positions(params.n,params.L0,params.k_s_water,params.m_e_water,params.m_piston_09,params.g,params.p_t0,p_p0)
for i in range(len(x_fluid_eq_75bar_water_09)):
    x_fluid_eq_75bar_water_09[i]=x_fluid_eq_75bar_water_09[i]+params.x_air_bottom
x_p0_75bar_water_09=x_fluid_eq_75bar_water_09[0]
offset= x_p0_75bar_water_09 - (params.x_air_bottom+params.th_piston_09)
for i in range(len(x_fluid_eq_75bar_water_09)):
    x_fluid_eq_75bar_water_09[i]=x_fluid_eq_75bar_water_09[i]-offset
x_p0_75bar_water_09=x_fluid_eq_75bar_water_09[0]

dx_eq_75bar_water_09 = np.zeros(params.n)
dx_eq_75bar_water_09[0]=x_fluid_eq_75bar_water_09[1]-x_p0_75bar_water_09
for i in range (1,params.n):
    dx_eq_75bar_water_09[i]=x_fluid_eq_75bar_water_09[i]-x_fluid_eq_75bar_water_09[i-1]

V_p0_75bar_09 = params.A_p * (x_p0_75bar_water_09-params.th_piston_09) 
V_t0_75bar_09 = params.A_p * (params.H_09-x_fluid_eq_75bar_water_09[params.n]) 

v_p0 = 0 
v_fluid0=np.zeros(params.n)
y0 = np.concatenate((x_fluid_eq_75bar_water_09, [v_p0], v_fluid0))

p_0_x_75bar_water_09= np.zeros(params.n)
for i in range (params.n):
    p_0_x_75bar_water_09[i] = params.p_t0 + params.rho_water*params.g*(x_fluid_eq_75bar_water_09[params.n]-x_fluid_eq_75bar_water_09[i])


sol_75bar_water_09 = solve_ivp(ms.system_water, [0, T_75], y0, args=(params.m_piston_09,params.H_09, V_t0_75bar_09, V_p0_75bar_09, params.k_s_water, params.m_e_water, params.n, p_p0), atol = 1e-6, rtol=1e-6, max_step = 1e-6)
sol_75bar_water_09.t=np.array(sol_75bar_water_09.t)

x_p_75bar_water_09 = sol_75bar_water_09.y[0]     
x_fluid_75bar_water_09 = sol_75bar_water_09.y[1:params.n+1] 
v_fluid_75bar_water_09 = sol_75bar_water_09.y[params.n+1:2*(params.n+1)] 
a0_values_75bar_water_09 = [ms.piston_acceleration_water(y,params.m_piston_09, params.H_09, V_p0_75bar_09, params.n, params.k_s_water, p_p0) for y in sol_75bar_water_09.y.T]
a0_min_75bar_water_09 = np.min(a0_values_75bar_water_09)
a0_max_75bar_water_09 = np.max(a0_values_75bar_water_09)
v_piston_75bar_water_09 = sol_75bar_water_09.y[params.n+1, :]
v_piston_max_75bar_water_09=np.max(v_piston_75bar_water_09)

all_acc_75bar_water_09 = np.array([
    ms.all_accelerations_water(y, params.m_piston_09, params.H_09, V_t0_75bar_09, V_p0_75bar_09, params.k_s_water, params.m_e_water, params.n, p_p0)
    for y in sol_75bar_water_09.y.T
])

a_piston_75bar_water_09 = all_acc_75bar_water_09[:, 0]      
a_fluid_75bar_water_09  = all_acc_75bar_water_09[:, 1:].T   
V_p_75bar_water_09 = params.A_p * (x_p_75bar_water_09 - params.th_piston_09)
V_t_75bar_water_09 = params.A_p * (params.H_09 - x_fluid_75bar_water_09[params.n-1, :])

p_t_75bar_water_09=[]
for i in range(len(sol_75bar_water_09.t)):
    p_ti=ms.pressure_on_top(sol_75bar_water_09.y[:,i], params.H_09, params.th_piston_09, V_t0_75bar_09, params.n)
    p_t_75bar_water_09.append(p_ti)
p_t_75bar_water_09 = np.array(p_t_75bar_water_09)
p_t_75bar_water_max_09 = np.max(p_t_75bar_water_09)

p_p_75bar_water_09=[]
for i in range(len(sol_75bar_water_09.t)):
    p_pi=ms.pressure_under_piston(sol_75bar_water_09.y[:,i], params.H_09, params.th_piston_09, V_p0_75bar_09, params.n,p_p0)
    p_p_75bar_water_09.append(p_pi)
p_p_75bar_water_09 = np.array(p_p_75bar_water_09)

dx_75bar_water_09 = np.zeros((params.n, len(sol_75bar_water_09.t)))
for j in range(len(sol_75bar_water_09.t)):
    dx_75bar_water_09[0,j]=x_fluid_75bar_water_09[0,j]-x_p_75bar_water_09[j]
    for i in range (1,params.n):
        dx_75bar_water_09[i,j]=x_fluid_75bar_water_09[i,j]-x_fluid_75bar_water_09[i-1,j]
p_75bar_water_09= np.ones((params.n,len(sol_75bar_water_09.t)))*params.p_0
for i in range(params.n):
    for j in range(len(sol_75bar_water_09.t)):
        p_75bar_water_09[i,j]=params.p_0-params.rho_l_water*params.c_water**2*((dx_75bar_water_09[i,j]*params.A_p)-dx_eq_75bar_water_09[i]*params.A_p) /(dx_eq_75bar_water_09[i]*params.A_p)

#M=1#
x_fluid_eq_75bar_water_1= ms.equilibrate_fluid_positions(params.n,params.L0,params.k_s_water,params.m_e_water,params.m_piston_1,params.g,params.p_t0,p_p0)
for i in range(len(x_fluid_eq_75bar_water_1)):
    x_fluid_eq_75bar_water_1[i]=x_fluid_eq_75bar_water_1[i]+params.x_air_bottom
x_p0_75bar_water_1=x_fluid_eq_75bar_water_1[0]
offset= x_p0_75bar_water_1 - (params.x_air_bottom+params.th_piston_1)
for i in range(len(x_fluid_eq_75bar_water_1)):
    x_fluid_eq_75bar_water_1[i]=x_fluid_eq_75bar_water_1[i]-offset
x_p0_75bar_water_1=x_fluid_eq_75bar_water_1[0]

dx_eq_75bar_water_1 = np.zeros(params.n)
dx_eq_75bar_water_1[0]=x_fluid_eq_75bar_water_1[1]-x_p0_75bar_water_1
for i in range (1,params.n):
    dx_eq_75bar_water_1[i]=x_fluid_eq_75bar_water_1[i]-x_fluid_eq_75bar_water_1[i-1]

V_p0_75bar_1 = params.A_p * (x_p0_75bar_water_1-params.th_piston_1)
V_t0_75bar_1 = params.A_p * (params.H_1-x_fluid_eq_75bar_water_1[params.n])  

v_p0 = 0 
v_fluid0=np.zeros(params.n)
y0 = np.concatenate((x_fluid_eq_75bar_water_1, [v_p0], v_fluid0))

p_0_x_75bar_water_1= np.zeros(params.n)
for i in range (params.n):
    p_0_x_75bar_water_1[i] = params.p_t0 + params.rho_water*params.g*(x_fluid_eq_75bar_water_1[params.n]-x_fluid_eq_75bar_water_1[i])


sol_75bar_water_1 = solve_ivp(ms.system_water, [0, T_75], y0, args=(params.m_piston_1,params.H_1, V_t0_75bar_1, V_p0_75bar_1, params.k_s_water, params.m_e_water, params.n, p_p0), atol = 1e-6, rtol=1e-6, max_step = 1e-6)
sol_75bar_water_1.t=np.array(sol_75bar_water_1.t)

x_p_75bar_water_1 = sol_75bar_water_1.y[0]     
x_fluid_75bar_water_1 = sol_75bar_water_1.y[1:params.n+1] 
v_fluid_75bar_water_1 = sol_75bar_water_1.y[params.n+1:2*(params.n+1)] 
a0_values_75bar_water_1 = [ms.piston_acceleration_water(y,params.m_piston_1, params.H_1, V_p0_75bar_1, params.n, params.k_s_water, p_p0) for y in sol_75bar_water_1.y.T]
a0_min_75bar_water_1 = np.min(a0_values_75bar_water_1)
a0_max_75bar_water_1 = np.max(a0_values_75bar_water_1)
v_piston_75bar_water_1 = sol_75bar_water_1.y[params.n+1, :]
v_piston_max_75bar_water_1=np.max(v_piston_75bar_water_1)

all_acc_75bar_water_1 = np.array([
    ms.all_accelerations_water(y, params.m_piston_1, params.H_1, V_t0_75bar_1, V_p0_75bar_1, params.k_s_water, params.m_e_water, params.n, p_p0)
    for y in sol_75bar_water_1.y.T
])

a_piston_75bar_water_1 = all_acc_75bar_water_1[:, 0]     
a_fluid_75bar_water_1  = all_acc_75bar_water_1[:, 1:].T  
V_p_75bar_water_1 = params.A_p * (x_p_75bar_water_1 - params.th_piston_1)
V_t_75bar_water_1 = params.A_p * (params.H_1 - x_fluid_75bar_water_1[params.n-1, :])

p_t_75bar_water_1=[]
for i in range(len(sol_75bar_water_1.t)):
    p_ti=ms.pressure_on_top(sol_75bar_water_1.y[:,i], params.H_1, params.th_piston_1, V_t0_75bar_1, params.n)
    p_t_75bar_water_1.append(p_ti)
p_t_75bar_water_1 = np.array(p_t_75bar_water_1)
p_t_75bar_water_max_1 = np.max(p_t_75bar_water_1)

p_p_75bar_water_1=[]
for i in range(len(sol_75bar_water_1.t)):
    p_pi=ms.pressure_under_piston(sol_75bar_water_1.y[:,i], params.H_1, params.th_piston_1, V_p0_75bar_1, params.n,p_p0)
    p_p_75bar_water_1.append(p_pi)
p_p_75bar_water_1 = np.array(p_p_75bar_water_1)

dx_75bar_water_1 = np.zeros((params.n, len(sol_75bar_water_1.t)))
for j in range(len(sol_75bar_water_1.t)):
    dx_75bar_water_1[0,j]=x_fluid_75bar_water_1[0,j]-x_p_75bar_water_1[j]
    for i in range (1,params.n):
        dx_75bar_water_1[i,j]=x_fluid_75bar_water_1[i,j]-x_fluid_75bar_water_1[i-1,j]
p_75bar_water_1= np.ones((params.n,len(sol_75bar_water_1.t)))*params.p_0
for i in range(params.n):
    for j in range(len(sol_75bar_water_1.t)):
        p_75bar_water_1[i,j]=params.p_0-params.rho_l_water*params.c_water**2*((dx_75bar_water_1[i,j]*params.A_p)-dx_eq_75bar_water_1[i]*params.A_p) /(dx_eq_75bar_water_1[i]*params.A_p)

#M=1.5#
x_fluid_eq_75bar_water_15= ms.equilibrate_fluid_positions(params.n,params.L0,params.k_s_water,params.m_e_water,params.m_piston_15,params.g,params.p_t0,p_p0)
for i in range(len(x_fluid_eq_75bar_water_15)):
    x_fluid_eq_75bar_water_15[i]=x_fluid_eq_75bar_water_15[i]+params.x_air_bottom
x_p0_75bar_water_15=x_fluid_eq_75bar_water_15[0]
offset= x_p0_75bar_water_15 - (params.x_air_bottom+params.th_piston_15)
for i in range(len(x_fluid_eq_75bar_water_15)):
    x_fluid_eq_75bar_water_15[i]=x_fluid_eq_75bar_water_15[i]-offset
x_p0_75bar_water_15=x_fluid_eq_75bar_water_15[0]

dx_eq_75bar_water_15 = np.zeros(params.n)
dx_eq_75bar_water_15[0]=x_fluid_eq_75bar_water_15[1]-x_p0_75bar_water_15
for i in range (1,params.n):
    dx_eq_75bar_water_15[i]=x_fluid_eq_75bar_water_15[i]-x_fluid_eq_75bar_water_15[i-1]

V_p0_75bar_15 = params.A_p * (x_p0_75bar_water_15-params.th_piston_15) 
V_t0_75bar_15 = params.A_p * (params.H_15-x_fluid_eq_75bar_water_15[params.n])  

v_p0 = 0 
v_fluid0=np.zeros(params.n)
y0 = np.concatenate((x_fluid_eq_75bar_water_15, [v_p0], v_fluid0))

p_0_x_75bar_water_15= np.zeros(params.n)
for i in range (params.n):
    p_0_x_75bar_water_15[i] = params.p_t0 + params.rho_water*params.g*(x_fluid_eq_75bar_water_15[params.n]-x_fluid_eq_75bar_water_15[i])


sol_75bar_water_15 = solve_ivp(ms.system_water, [0, T_75], y0, args=(params.m_piston_15,params.H_15, V_t0_75bar_15, V_p0_75bar_15, params.k_s_water, params.m_e_water, params.n, p_p0), atol = 1e-6, rtol=1e-6, max_step = 1e-6)
sol_75bar_water_15.t=np.array(sol_75bar_water_15.t)

x_p_75bar_water_15 = sol_75bar_water_15.y[0]    
x_fluid_75bar_water_15 = sol_75bar_water_15.y[1:params.n+1] 
v_fluid_75bar_water_15 = sol_75bar_water_15.y[params.n+1:2*(params.n+1)] 
a0_values_75bar_water_15 = [ms.piston_acceleration_water(y,params.m_piston_15, params.H_15, V_p0_75bar_15, params.n, params.k_s_water, p_p0) for y in sol_75bar_water_15.y.T]
a0_min_75bar_water_15 = np.min(a0_values_75bar_water_15)
a0_max_75bar_water_15 = np.max(a0_values_75bar_water_15)
v_piston_75bar_water_15 = sol_75bar_water_15.y[params.n+1, :]
v_piston_max_75bar_water_15=np.max(v_piston_75bar_water_15)

all_acc_75bar_water_15 = np.array([
    ms.all_accelerations_water(y, params.m_piston_15, params.H_15, V_t0_75bar_15, V_p0_75bar_15, params.k_s_water, params.m_e_water, params.n, p_p0)
    for y in sol_75bar_water_15.y.T
])

a_piston_75bar_water_15 = all_acc_75bar_water_15[:, 0]      
a_fluid_75bar_water_15  = all_acc_75bar_water_15[:, 1:].T   
V_p_75bar_water_15 = params.A_p * (x_p_75bar_water_15 - params.th_piston_15)
V_t_75bar_water_15 = params.A_p * (params.H_15 - x_fluid_75bar_water_15[params.n-1, :])

p_t_75bar_water_15=[]
for i in range(len(sol_75bar_water_15.t)):
    p_ti=ms.pressure_on_top(sol_75bar_water_15.y[:,i], params.H_15, params.th_piston_15, V_t0_75bar_15, params.n)
    p_t_75bar_water_15.append(p_ti)
p_t_75bar_water_15 = np.array(p_t_75bar_water_15)
p_t_75bar_water_max_15 = np.max(p_t_75bar_water_15)

p_p_75bar_water_15=[]
for i in range(len(sol_75bar_water_15.t)):
    p_pi=ms.pressure_under_piston(sol_75bar_water_15.y[:,i], params.H_15, params.th_piston_15, V_p0_75bar_15, params.n,p_p0)
    p_p_75bar_water_15.append(p_pi)
p_p_75bar_water_15 = np.array(p_p_75bar_water_15)

dx_75bar_water_15 = np.zeros((params.n, len(sol_75bar_water_15.t)))
for j in range(len(sol_75bar_water_15.t)):
    dx_75bar_water_15[0,j]=x_fluid_75bar_water_15[0,j]-x_p_75bar_water_15[j]
    for i in range (1,params.n):
        dx_75bar_water_15[i,j]=x_fluid_75bar_water_15[i,j]-x_fluid_75bar_water_15[i-1,j]
p_75bar_water_15= np.ones((params.n,len(sol_75bar_water_15.t)))*params.p_0
for i in range(params.n):
    for j in range(len(sol_75bar_water_15.t)):
        p_75bar_water_15[i,j]=params.p_0-params.rho_l_water*params.c_water**2*((dx_75bar_water_15[i,j]*params.A_p)-dx_eq_75bar_water_15[i]*params.A_p) /(dx_eq_75bar_water_15[i]*params.A_p)

#M=2#
x_fluid_eq_75bar_water_2= ms.equilibrate_fluid_positions(params.n,params.L0,params.k_s_water,params.m_e_water,params.m_piston_2,params.g,params.p_t0,p_p0)
for i in range(len(x_fluid_eq_75bar_water_2)):
    x_fluid_eq_75bar_water_2[i]=x_fluid_eq_75bar_water_2[i]+params.x_air_bottom
x_p0_75bar_water_2=x_fluid_eq_75bar_water_2[0]
offset= x_p0_75bar_water_2 - (params.x_air_bottom+params.th_piston_2)
for i in range(len(x_fluid_eq_75bar_water_2)):
    x_fluid_eq_75bar_water_2[i]=x_fluid_eq_75bar_water_2[i]-offset
x_p0_75bar_water_2=x_fluid_eq_75bar_water_2[0]

dx_eq_75bar_water_2 = np.zeros(params.n)
dx_eq_75bar_water_2[0]=x_fluid_eq_75bar_water_2[1]-x_p0_75bar_water_2
for i in range (1,params.n):
    dx_eq_75bar_water_2[i]=x_fluid_eq_75bar_water_2[i]-x_fluid_eq_75bar_water_2[i-1]

V_p0_75bar_2 = params.A_p * (x_p0_75bar_water_2-params.th_piston_2) 
V_t0_75bar_2 = params.A_p * (params.H_2-x_fluid_eq_75bar_water_2[params.n])  

v_p0 = 0 
v_fluid0=np.zeros(params.n)
y0 = np.concatenate((x_fluid_eq_75bar_water_2, [v_p0], v_fluid0))

p_0_x_75bar_water_2= np.zeros(params.n)
for i in range (params.n):
    p_0_x_75bar_water_2[i] = params.p_t0 + params.rho_water*params.g*(x_fluid_eq_75bar_water_2[params.n]-x_fluid_eq_75bar_water_2[i])


sol_75bar_water_2 = solve_ivp(ms.system_water, [0, T_75], y0, args=(params.m_piston_2,params.H_2, V_t0_75bar_2, V_p0_75bar_2, params.k_s_water, params.m_e_water, params.n, p_p0), atol = 1e-6, rtol=1e-6, max_step = 1e-6)
sol_75bar_water_2.t=np.array(sol_75bar_water_2.t)

x_p_75bar_water_2 = sol_75bar_water_2.y[0]      
x_fluid_75bar_water_2 = sol_75bar_water_2.y[1:params.n+1] 
v_fluid_75bar_water_2 = sol_75bar_water_2.y[params.n+1:2*(params.n+1)] 
a0_values_75bar_water_2 = [ms.piston_acceleration_water(y,params.m_piston_2, params.H_2, V_p0_75bar_2, params.n, params.k_s_water, p_p0) for y in sol_75bar_water_2.y.T]
a0_min_75bar_water_2 = np.min(a0_values_75bar_water_2)
a0_max_75bar_water_2 = np.max(a0_values_75bar_water_2)
v_piston_75bar_water_2 = sol_75bar_water_2.y[params.n+1, :]
v_piston_max_75bar_water_2=np.max(v_piston_75bar_water_2)

all_acc_75bar_water_2 = np.array([
    ms.all_accelerations_water(y, params.m_piston_2, params.H_2, V_t0_75bar_2, V_p0_75bar_2, params.k_s_water, params.m_e_water, params.n, p_p0)
    for y in sol_75bar_water_2.y.T
])

a_piston_75bar_water_2 = all_acc_75bar_water_2[:, 0]      
a_fluid_75bar_water_2  = all_acc_75bar_water_2[:, 1:].T  
V_p_75bar_water_2 = params.A_p * (x_p_75bar_water_2 - params.th_piston_2)
V_t_75bar_water_2 = params.A_p * (params.H_2 - x_fluid_75bar_water_2[params.n-1, :])

p_t_75bar_water_2=[]
for i in range(len(sol_75bar_water_2.t)):
    p_ti=ms.pressure_on_top(sol_75bar_water_2.y[:,i], params.H_2, params.th_piston_2, V_t0_75bar_2, params.n)
    p_t_75bar_water_2.append(p_ti)
p_t_75bar_water_2 = np.array(p_t_75bar_water_2)
p_t_75bar_water_max_2 = np.max(p_t_75bar_water_2)

p_p_75bar_water_2=[]
for i in range(len(sol_75bar_water_2.t)):
    p_pi=ms.pressure_under_piston(sol_75bar_water_2.y[:,i], params.H_2, params.th_piston_2, V_p0_75bar_2, params.n,p_p0)
    p_p_75bar_water_2.append(p_pi)
p_p_75bar_water_2 = np.array(p_p_75bar_water_2)

dx_75bar_water_2 = np.zeros((params.n, len(sol_75bar_water_2.t)))
for j in range(len(sol_75bar_water_2.t)):
    dx_75bar_water_2[0,j]=x_fluid_75bar_water_2[0,j]-x_p_75bar_water_2[j]
    for i in range (1,params.n):
        dx_75bar_water_2[i,j]=x_fluid_75bar_water_2[i,j]-x_fluid_75bar_water_2[i-1,j]
p_75bar_water_2= np.ones((params.n,len(sol_75bar_water_2.t)))*params.p_0
for i in range(params.n):
    for j in range(len(sol_75bar_water_2.t)):
        p_75bar_water_2[i,j]=params.p_0-params.rho_l_water*params.c_water**2*((dx_75bar_water_2[i,j]*params.A_p)-dx_eq_75bar_water_2[i]*params.A_p) /(dx_eq_75bar_water_2[i]*params.A_p)

#no piston#
x_fluid_eq_75bar_water_no_piston = ms.equilibrate_fluid_positions_no_piston(params.n, params.L0, params.k_s_water, params.m_e_water, params.g, params.p_t0, p_p0)
x_0_75bar_water_no_piston = x_fluid_eq_75bar_water_no_piston[0]

dx_eq_75bar_water_no_piston = np.zeros(params.n)
dx_eq_75bar_water_no_piston[0] = x_fluid_eq_75bar_water_no_piston[1] - x_0_75bar_water_no_piston
for i in range(1, params.n):
    dx_eq_75bar_water_no_piston[i] = x_fluid_eq_75bar_water_no_piston[i] - x_fluid_eq_75bar_water_no_piston[i - 1]

V_p0 = params.A_p * x_0_75bar_water_no_piston  
V_t0 = params.A_p * (params.H_no_piston - x_fluid_eq_75bar_water_no_piston[params.n])  

v_0 = 0 
v_fluid0 = np.zeros(params.n)
y0 = np.concatenate((x_fluid_eq_75bar_water_no_piston, [v_0], v_fluid0))


sol_75bar_water_no_piston = solve_ivp(ms.system_water_no_piston, [0, T_75], y0, args=(V_t0,V_p0, p_p0), atol=1e-6, rtol=1e-6, max_step=1e-6)
sol_75bar_water_no_piston.t=np.array(sol_75bar_water_no_piston.t)

x_0_75bar_water_no_piston = sol_75bar_water_no_piston.y[0]  
x_fluid_75bar_water_no_piston = sol_75bar_water_no_piston.y[1:params.n + 1] 
v_fluid_75bar_water_no_piston = sol_75bar_water_no_piston.y[params.n + 1:2 * (params.n + 1)] 
a0_values_75bar_water_no_piston = [ms.liquid_acceleration_water(y, params.n, params.k_s_water, params.m_e_water, params.L0, V_p0, p_p0) for y in sol_75bar_water_no_piston.y.T]
a0_min_75bar_water_no_piston = np.min(a0_values_75bar_water_no_piston)
a0_max_75bar_water_no_piston = np.max(a0_values_75bar_water_no_piston)
v_piston_75bar_water_no_piston = sol_75bar_water_no_piston.y[params.n + 1, :]
v_piston_max_75bar_water_no_piston = np.max(v_piston_75bar_water_no_piston)

all_acc_75bar_water_no_piston = np.array([
    ms.all_accelerations_water_no_piston(y, V_t0, V_p0, p_p0)
    for y in sol_75bar_water_no_piston.y.T
])  

a_fluid_75bar_water_no_piston = all_acc_75bar_water_no_piston.T  
V_p_75bar_water_no_piston = params.A_p * x_0_75bar_water_no_piston
V_t_75bar_water_no_piston = params.A_p * (params.H_no_piston - x_fluid_75bar_water_no_piston[params.n-1, :])

p_t_75bar_water_no_piston = []
for i in range(len(sol_75bar_water_no_piston.t)):
    p_ti = ms.pressure_on_top_no_piston(sol_75bar_water_no_piston.y[:, i],V_t0)
    p_t_75bar_water_no_piston.append(p_ti)


p_p_75bar_water_no_piston = []
for i in range(len(sol_75bar_water_no_piston.t)):
    p_pi = ms.pressure_gas_bottom(sol_75bar_water_no_piston.y[:, i], V_p0, p_p0)
    p_p_75bar_water_no_piston.append(p_pi)


dx_75bar_water_no_piston = np.zeros((params.n, len(sol_75bar_water_no_piston.t)))
for j in range(len(sol_75bar_water_no_piston.t)):
    dx_75bar_water_no_piston[0, j] = x_fluid_75bar_water_no_piston[0, j] - x_0_75bar_water_no_piston[j]
    for i in range(1, params.n):
        dx_75bar_water_no_piston[i, j] = x_fluid_75bar_water_no_piston[i, j] - x_fluid_75bar_water_no_piston[i - 1, j]

p_75bar_water_no_piston = np.ones((params.n, len(sol_75bar_water_no_piston.t))) * params.p_0
for i in range(params.n):
    for j in range(len(sol_75bar_water_no_piston.t)):
        p_75bar_water_no_piston[i, j] = params.p_0 - params.rho_l_water * params.c_water**2 * ((dx_75bar_water_no_piston[i, j] * params.A_p) - dx_eq_75bar_water_no_piston[i] * params.A_p) / (dx_eq_75bar_water_no_piston[i] * params.A_p)

#lithium#
#M=1.69#
x_fluid_eq_75bar_lithium= ms.equilibrate_fluid_positions(params.n,params.L0,params.k_s_lithium,params.m_e_lithium,params.m_piston_ref,params.g,params.p_t0,p_p0)
for i in range(len(x_fluid_eq_75bar_lithium)):
    x_fluid_eq_75bar_lithium[i]=x_fluid_eq_75bar_lithium[i]+params.x_air_bottom
x_p0_75bar_lithium=x_fluid_eq_75bar_lithium[0]
offset= x_p0_75bar_lithium - 0.25
for i in range(len(x_fluid_eq_75bar_lithium)):
    x_fluid_eq_75bar_lithium[i]=x_fluid_eq_75bar_lithium[i]-offset
x_p0_75bar_lithium=x_fluid_eq_75bar_lithium[0]

dx_eq_75bar_lithium= np.zeros(params.n)
dx_eq_75bar_lithium[0]=x_fluid_eq_75bar_lithium[1]-x_p0_75bar_lithium
for i in range (1,params.n):
    dx_eq_75bar_lithium[i]=x_fluid_eq_75bar_lithium[i]-x_fluid_eq_75bar_lithium[i-1]

V_p0 = params.A_p * (x_p0_75bar_lithium-params.th_piston_ref)
V_t0 = params.A_p * (params.H_ref-x_fluid_eq_75bar_lithium[params.n])  

v_p0 = 0 
v_fluid0=np.zeros(params.n)
y0 = np.concatenate((x_fluid_eq_75bar_lithium, [v_p0], v_fluid0))


sol_75bar_lithium = solve_ivp(ms.system_lith, [0, T_75], y0, args=(params.m_piston_ref,params.H_ref, V_t0, V_p0, params.n, p_p0), atol = 1e-6, rtol=1e-6, max_step = 1e-6)


x_p_75bar_lithium = sol_75bar_lithium.y[0]      
x_fluid_75bar_lithium = sol_75bar_lithium.y[1:params.n+1]
v_fluid_75bar_lithium = sol_75bar_lithium.y[params.n+1:2*(params.n+1)] 
a0_values_75bar_lithium = [ms.piston_acceleration_lith(y, params.m_piston_ref, V_p0, p_p0) for y in sol_75bar_lithium.y.T]
a0_min_75bar_lithium = np.min(a0_values_75bar_lithium)
a0_max_75bar_lithium = np.max(a0_values_75bar_lithium)
v_piston_75bar_lithium = sol_75bar_lithium.y[params.n+1, :]
v_piston_max_75bar_lithium=np.max(v_piston_75bar_lithium)


p_t_75bar_lithium=[]
for i in range(len(sol_75bar_lithium.t)):
    p_ti=ms.pressure_on_top(sol_75bar_lithium.y[:,i], params.H_ref, params.th_piston_ref, V_t0, params.n)
    p_t_75bar_lithium.append(p_ti)


p_p_75bar_lithium=[]
for i in range(len(sol_75bar_lithium.t)):
    p_pi=ms.pressure_under_piston(sol_75bar_lithium.y[:,i], params.H_ref, params.th_piston_ref, V_p0, params.n,p_p0)
    p_p_75bar_lithium.append(p_pi)


dx_75bar_lithium = np.zeros((params.n, len(sol_75bar_lithium.t)))
for j in range(len(sol_75bar_lithium.t)):
    dx_75bar_lithium[0,j]=x_fluid_75bar_lithium[0,j]-x_p_75bar_lithium[j]
    for i in range (1,params.n):
        dx_75bar_lithium[i,j]=x_fluid_75bar_lithium[i,j]-x_fluid_75bar_lithium[i-1,j]
p_75bar_lithium= np.ones((params.n,len(sol_75bar_lithium.t)))*params.p_0
for i in range(params.n):
    for j in range(len(sol_75bar_lithium.t)):
        p_75bar_lithium[i,j]=params.p_0-params.rho_l_lithium*params.c_lithium**2*((dx_75bar_lithium[i,j]*params.A_p)-dx_eq_75bar_lithium[i]*params.A_p) /(dx_eq_75bar_lithium[i]*params.A_p)


#Driving pressure: 10MPa#
p_p0 = 100e5
T_100 = 0.02 

#water#
#M=0.1#
x_fluid_eq_100bar_water_01= ms.equilibrate_fluid_positions(params.n,params.L0,params.k_s_water,params.m_e_water,params.m_piston_01,params.g,params.p_t0,p_p0)
for i in range(len(x_fluid_eq_100bar_water_01)):
    x_fluid_eq_100bar_water_01[i]=x_fluid_eq_100bar_water_01[i]+params.x_air_bottom
x_p0_100bar_water_01=x_fluid_eq_100bar_water_01[0]
offset= x_p0_100bar_water_01 - (params.x_air_bottom+params.th_piston_01)
for i in range(len(x_fluid_eq_100bar_water_01)):
    x_fluid_eq_100bar_water_01[i]=x_fluid_eq_100bar_water_01[i]-offset
x_p0_100bar_water_01=x_fluid_eq_100bar_water_01[0]

dx_eq_100bar_water_01 = np.zeros(params.n)
dx_eq_100bar_water_01[0]=x_fluid_eq_100bar_water_01[1]-x_p0_100bar_water_01
for i in range (1,params.n):
    dx_eq_100bar_water_01[i]=x_fluid_eq_100bar_water_01[i]-x_fluid_eq_100bar_water_01[i-1]

V_p0_100bar_01 = params.A_p * (x_p0_100bar_water_01-params.th_piston_01)
V_t0_100bar_01 = params.A_p * (params.H_01-x_fluid_eq_100bar_water_01[params.n])  

v_p0 = 0 
v_fluid0=np.zeros(params.n)
y0 = np.concatenate((x_fluid_eq_100bar_water_01, [v_p0], v_fluid0))

p_0_x_100bar_water_01= np.zeros(params.n)
for i in range (params.n):
    p_0_x_100bar_water_01[i] = params.p_t0 + params.rho_water*params.g*(x_fluid_eq_100bar_water_01[params.n]-x_fluid_eq_100bar_water_01[i])


sol_100bar_water_01 = solve_ivp(ms.system_water, [0, T_100], y0, args=(params.m_piston_01,params.H_01, V_t0_100bar_01, V_p0_100bar_01, params.k_s_water, params.m_e_water, params.n, p_p0), atol = 1e-6, rtol=1e-6, max_step = 1e-6)
sol_100bar_water_01.t=np.array(sol_100bar_water_01.t)

x_p_100bar_water_01 = sol_100bar_water_01.y[0]      
x_fluid_100bar_water_01 = sol_100bar_water_01.y[1:params.n+1]
v_fluid_100bar_water_01 = sol_100bar_water_01.y[params.n+1:2*(params.n+1)] 
a0_values_100bar_water_01 = [ms.piston_acceleration_water(y,params.m_piston_01, params.H_01, V_p0_100bar_01, params.n, params.k_s_water, p_p0) for y in sol_100bar_water_01.y.T]
a0_min_100bar_water_01 = np.min(a0_values_100bar_water_01)
a0_max_100bar_water_01 = np.max(a0_values_100bar_water_01)
v_piston_100bar_water_01 = sol_100bar_water_01.y[params.n+1, :]
v_piston_max_100bar_water_01=np.max(v_piston_100bar_water_01)

all_acc_100bar_water_01 = np.array([
    ms.all_accelerations_water(y, params.m_piston_01, params.H_01, V_t0_100bar_01, V_p0_100bar_01, params.k_s_water, params.m_e_water, params.n, p_p0)
    for y in sol_100bar_water_01.y.T
])

a_piston_100bar_water_01 = all_acc_100bar_water_01[:, 0]      
a_fluid_100bar_water_01  = all_acc_100bar_water_01[:, 1:].T   
V_p_100bar_water_01 = params.A_p * (x_p_100bar_water_01 - params.th_piston_01)
V_t_100bar_water_01 = params.A_p * (params.H_01 - x_fluid_100bar_water_01[params.n-1, :])

p_t_100bar_water_01=[]
for i in range(len(sol_100bar_water_01.t)):
    p_ti=ms.pressure_on_top(sol_100bar_water_01.y[:,i], params.H_01, params.th_piston_01, V_t0_100bar_01, params.n)
    p_t_100bar_water_01.append(p_ti)
p_t_100bar_water_01 = np.array(p_t_100bar_water_01)
p_t_100bar_water_max_01 = np.max(p_t_100bar_water_01)

p_p_100bar_water_01=[]
for i in range(len(sol_100bar_water_01.t)):
    p_pi=ms.pressure_under_piston(sol_100bar_water_01.y[:,i], params.H_01, params.th_piston_01, V_p0_100bar_01, params.n,p_p0)
    p_p_100bar_water_01.append(p_pi)
p_p_100bar_water_01 = np.array(p_p_100bar_water_01)

dx_100bar_water_01 = np.zeros((params.n, len(sol_100bar_water_01.t)))
for j in range(len(sol_100bar_water_01.t)):
    dx_100bar_water_01[0,j]=x_fluid_100bar_water_01[0,j]-x_p_100bar_water_01[j]
    for i in range (1,params.n):
        dx_100bar_water_01[i,j]=x_fluid_100bar_water_01[i,j]-x_fluid_100bar_water_01[i-1,j]
p_100bar_water_01= np.ones((params.n,len(sol_100bar_water_01.t)))*params.p_0
for i in range(params.n):
    for j in range(len(sol_100bar_water_01.t)):
        p_100bar_water_01[i,j]=params.p_0-params.rho_l_water*params.c_water**2*((dx_100bar_water_01[i,j]*params.A_p)-dx_eq_100bar_water_01[i]*params.A_p) /(dx_eq_100bar_water_01[i]*params.A_p)

#M=0.5#
x_fluid_eq_100bar_water_05= ms.equilibrate_fluid_positions(params.n,params.L0,params.k_s_water,params.m_e_water,params.m_piston_05,params.g,params.p_t0,p_p0)
for i in range(len(x_fluid_eq_100bar_water_05)):
    x_fluid_eq_100bar_water_05[i]=x_fluid_eq_100bar_water_05[i]+params.x_air_bottom
x_p0_100bar_water_05=x_fluid_eq_100bar_water_05[0]
offset= x_p0_100bar_water_05 - (params.x_air_bottom+params.th_piston_05)
for i in range(len(x_fluid_eq_100bar_water_05)):
    x_fluid_eq_100bar_water_05[i]=x_fluid_eq_100bar_water_05[i]-offset
x_p0_100bar_water_05=x_fluid_eq_100bar_water_05[0]

dx_eq_100bar_water_05 = np.zeros(params.n)
dx_eq_100bar_water_05[0]=x_fluid_eq_100bar_water_05[1]-x_p0_100bar_water_05
for i in range (1,params.n):
    dx_eq_100bar_water_05[i]=x_fluid_eq_100bar_water_05[i]-x_fluid_eq_100bar_water_05[i-1]

V_p0_100bar_05 = params.A_p * (x_p0_100bar_water_05-params.th_piston_05) 
V_t0_100bar_05 = params.A_p * (params.H_05-x_fluid_eq_100bar_water_05[params.n])  

v_p0 = 0 
v_fluid0=np.zeros(params.n)
y0 = np.concatenate((x_fluid_eq_100bar_water_05, [v_p0], v_fluid0))

p_0_x_100bar_water_05= np.zeros(params.n)
for i in range (params.n):
    p_0_x_100bar_water_05[i] = params.p_t0 + params.rho_water*params.g*(x_fluid_eq_100bar_water_05[params.n]-x_fluid_eq_100bar_water_05[i])


sol_100bar_water_05 = solve_ivp(ms.system_water, [0, T_100], y0, args=(params.m_piston_05,params.H_05, V_t0_100bar_05, V_p0_100bar_05, params.k_s_water, params.m_e_water, params.n, p_p0), atol = 1e-6, rtol=1e-6, max_step = 1e-6)
sol_100bar_water_05.t=np.array(sol_100bar_water_05.t)

x_p_100bar_water_05 = sol_100bar_water_05.y[0]      
x_fluid_100bar_water_05 = sol_100bar_water_05.y[1:params.n+1] 
v_fluid_100bar_water_05 = sol_100bar_water_05.y[params.n+1:2*(params.n+1)] 
a0_values_100bar_water_05 = [ms.piston_acceleration_water(y,params.m_piston_05, params.H_05, V_p0_100bar_05, params.n, params.k_s_water, p_p0) for y in sol_100bar_water_05.y.T]
a0_min_100bar_water_05 = np.min(a0_values_100bar_water_05)
a0_max_100bar_water_05 = np.max(a0_values_100bar_water_05)
v_piston_100bar_water_05 = sol_100bar_water_05.y[params.n+1, :]
v_piston_max_100bar_water_05=np.max(v_piston_100bar_water_05)

all_acc_100bar_water_05 = np.array([
    ms.all_accelerations_water(y, params.m_piston_05, params.H_05, V_t0_100bar_05, V_p0_100bar_05, params.k_s_water, params.m_e_water, params.n, p_p0)
    for y in sol_100bar_water_05.y.T
])

a_piston_100bar_water_05 = all_acc_100bar_water_05[:, 0]     
a_fluid_100bar_water_05  = all_acc_100bar_water_05[:, 1:].T  
V_p_100bar_water_05 = params.A_p * (x_p_100bar_water_05 - params.th_piston_05)
V_t_100bar_water_05 = params.A_p * (params.H_05 - x_fluid_100bar_water_05[params.n-1, :])

p_t_100bar_water_05=[]
for i in range(len(sol_100bar_water_05.t)):
    p_ti=ms.pressure_on_top(sol_100bar_water_05.y[:,i], params.H_05, params.th_piston_05, V_t0_100bar_05, params.n)
    p_t_100bar_water_05.append(p_ti)
p_t_100bar_water_05 = np.array(p_t_100bar_water_05)
p_t_100bar_water_max_05 = np.max(p_t_100bar_water_05)

p_p_100bar_water_05=[]
for i in range(len(sol_100bar_water_05.t)):
    p_pi=ms.pressure_under_piston(sol_100bar_water_05.y[:,i], params.H_05, params.th_piston_05, V_p0_100bar_05, params.n,p_p0)
    p_p_100bar_water_05.append(p_pi)
p_p_100bar_water_05 = np.array(p_p_100bar_water_05)

dx_100bar_water_05 = np.zeros((params.n, len(sol_100bar_water_05.t)))
for j in range(len(sol_100bar_water_05.t)):
    dx_100bar_water_05[0,j]=x_fluid_100bar_water_05[0,j]-x_p_100bar_water_05[j]
    for i in range (1,params.n):
        dx_100bar_water_05[i,j]=x_fluid_100bar_water_05[i,j]-x_fluid_100bar_water_05[i-1,j]
p_100bar_water_05= np.ones((params.n,len(sol_100bar_water_05.t)))*params.p_0
for i in range(params.n):
    for j in range(len(sol_100bar_water_05.t)):
        p_100bar_water_05[i,j]=params.p_0-params.rho_l_water*params.c_water**2*((dx_100bar_water_05[i,j]*params.A_p)-dx_eq_100bar_water_05[i]*params.A_p) /(dx_eq_100bar_water_05[i]*params.A_p)

#M=0.9#
x_fluid_eq_100bar_water_09= ms.equilibrate_fluid_positions(params.n,params.L0,params.k_s_water,params.m_e_water,params.m_piston_09,params.g,params.p_t0,p_p0)
for i in range(len(x_fluid_eq_100bar_water_09)):
    x_fluid_eq_100bar_water_09[i]=x_fluid_eq_100bar_water_09[i]+params.x_air_bottom
x_p0_100bar_water_09=x_fluid_eq_100bar_water_09[0]
offset= x_p0_100bar_water_09 - (params.x_air_bottom+params.th_piston_09)
for i in range(len(x_fluid_eq_100bar_water_09)):
    x_fluid_eq_100bar_water_09[i]=x_fluid_eq_100bar_water_09[i]-offset
x_p0_100bar_water_09=x_fluid_eq_100bar_water_09[0]

dx_eq_100bar_water_09 = np.zeros(params.n)
dx_eq_100bar_water_09[0]=x_fluid_eq_100bar_water_09[1]-x_p0_100bar_water_09
for i in range (1,params.n):
    dx_eq_100bar_water_09[i]=x_fluid_eq_100bar_water_09[i]-x_fluid_eq_100bar_water_09[i-1]

V_p0_100bar_09 = params.A_p * (x_p0_100bar_water_09-params.th_piston_09) 
V_t0_100bar_09 = params.A_p * (params.H_09-x_fluid_eq_100bar_water_09[params.n])  

v_p0 = 0 
v_fluid0=np.zeros(params.n)
y0 = np.concatenate((x_fluid_eq_100bar_water_09, [v_p0], v_fluid0))

p_0_x_100bar_water_09= np.zeros(params.n)
for i in range (params.n):
    p_0_x_100bar_water_09[i] = params.p_t0 + params.rho_water*params.g*(x_fluid_eq_100bar_water_09[params.n]-x_fluid_eq_100bar_water_09[i])


sol_100bar_water_09 = solve_ivp(ms.system_water, [0, T_100], y0, args=(params.m_piston_09,params.H_09, V_t0_100bar_09, V_p0_100bar_09, params.k_s_water, params.m_e_water, params.n, p_p0), atol = 1e-6, rtol=1e-6, max_step = 1e-6)
sol_100bar_water_09.t=np.array(sol_100bar_water_09.t)

x_p_100bar_water_09 = sol_100bar_water_09.y[0]     
x_fluid_100bar_water_09 = sol_100bar_water_09.y[1:params.n+1] 
v_fluid_100bar_water_09 = sol_100bar_water_09.y[params.n+1:2*(params.n+1)] 
a0_values_100bar_water_09 = [ms.piston_acceleration_water(y,params.m_piston_09, params.H_09, V_p0_100bar_09, params.n, params.k_s_water, p_p0) for y in sol_100bar_water_09.y.T]
a0_min_100bar_water_09 = np.min(a0_values_100bar_water_09)
a0_max_100bar_water_09 = np.max(a0_values_100bar_water_09)
v_piston_100bar_water_09 = sol_100bar_water_09.y[params.n+1, :]
v_piston_max_100bar_water_09=np.max(v_piston_100bar_water_09)

all_acc_100bar_water_09 = np.array([
    ms.all_accelerations_water(y, params.m_piston_09, params.H_09, V_t0_100bar_09, V_p0_100bar_09, params.k_s_water, params.m_e_water, params.n, p_p0)
    for y in sol_100bar_water_09.y.T
])

a_piston_100bar_water_09 = all_acc_100bar_water_09[:, 0]     
a_fluid_100bar_water_09  = all_acc_100bar_water_09[:, 1:].T   
V_p_100bar_water_09 = params.A_p * (x_p_100bar_water_09 - params.th_piston_09)
V_t_100bar_water_09 = params.A_p * (params.H_09 - x_fluid_100bar_water_09[params.n-1, :])

p_t_100bar_water_09=[]
for i in range(len(sol_100bar_water_09.t)):
    p_ti=ms.pressure_on_top(sol_100bar_water_09.y[:,i], params.H_09, params.th_piston_09, V_t0_100bar_09, params.n)
    p_t_100bar_water_09.append(p_ti)
p_t_100bar_water_09 = np.array(p_t_100bar_water_09)
p_t_100bar_water_max_09 = np.max(p_t_100bar_water_09)

p_p_100bar_water_09=[]
for i in range(len(sol_100bar_water_09.t)):
    p_pi=ms.pressure_under_piston(sol_100bar_water_09.y[:,i], params.H_09, params.th_piston_09, V_p0_100bar_09, params.n,p_p0)
    p_p_100bar_water_09.append(p_pi)
p_p_100bar_water_09 = np.array(p_p_100bar_water_09)

dx_100bar_water_09 = np.zeros((params.n, len(sol_100bar_water_09.t)))
for j in range(len(sol_100bar_water_09.t)):
    dx_100bar_water_09[0,j]=x_fluid_100bar_water_09[0,j]-x_p_100bar_water_09[j]
    for i in range (1,params.n):
        dx_100bar_water_09[i,j]=x_fluid_100bar_water_09[i,j]-x_fluid_100bar_water_09[i-1,j]
p_100bar_water_09= np.ones((params.n,len(sol_100bar_water_09.t)))*params.p_0
for i in range(params.n):
    for j in range(len(sol_100bar_water_09.t)):
        p_100bar_water_09[i,j]=params.p_0-params.rho_l_water*params.c_water**2*((dx_100bar_water_09[i,j]*params.A_p)-dx_eq_100bar_water_09[i]*params.A_p) /(dx_eq_100bar_water_09[i]*params.A_p)

#M=1#
x_fluid_eq_100bar_water_1= ms.equilibrate_fluid_positions(params.n,params.L0,params.k_s_water,params.m_e_water,params.m_piston_1,params.g,params.p_t0,p_p0)
for i in range(len(x_fluid_eq_100bar_water_1)):
    x_fluid_eq_100bar_water_1[i]=x_fluid_eq_100bar_water_1[i]+params.x_air_bottom
x_p0_100bar_water_1=x_fluid_eq_100bar_water_1[0]
offset= x_p0_100bar_water_1 - (params.x_air_bottom+params.th_piston_1)
for i in range(len(x_fluid_eq_100bar_water_1)):
    x_fluid_eq_100bar_water_1[i]=x_fluid_eq_100bar_water_1[i]-offset
x_p0_100bar_water_1=x_fluid_eq_100bar_water_1[0]

dx_eq_100bar_water_1 = np.zeros(params.n)
dx_eq_100bar_water_1[0]=x_fluid_eq_100bar_water_1[1]-x_p0_100bar_water_1
for i in range (1,params.n):
    dx_eq_100bar_water_1[i]=x_fluid_eq_100bar_water_1[i]-x_fluid_eq_100bar_water_1[i-1]

V_p0_100bar_1 = params.A_p * (x_p0_100bar_water_1-params.th_piston_1) 
V_t0_100bar_1 = params.A_p * (params.H_1-x_fluid_eq_100bar_water_1[params.n])  

v_p0 = 0 
v_fluid0=np.zeros(params.n)
y0 = np.concatenate((x_fluid_eq_100bar_water_1, [v_p0], v_fluid0))

p_0_x_100bar_water_1= np.zeros(params.n)
for i in range (params.n):
    p_0_x_100bar_water_1[i] = params.p_t0 + params.rho_water*params.g*(x_fluid_eq_100bar_water_1[params.n]-x_fluid_eq_100bar_water_1[i])


sol_100bar_water_1 = solve_ivp(ms.system_water, [0, T_100], y0, args=(params.m_piston_1,params.H_1, V_t0_100bar_1, V_p0_100bar_1, params.k_s_water, params.m_e_water, params.n, p_p0), atol = 1e-6, rtol=1e-6, max_step = 1e-6)
sol_100bar_water_1.t=np.array(sol_100bar_water_1.t)

x_p_100bar_water_1 = sol_100bar_water_1.y[0]    
x_fluid_100bar_water_1 = sol_100bar_water_1.y[1:params.n+1] 
v_fluid_100bar_water_1 = sol_100bar_water_1.y[params.n+1:2*(params.n+1)] 
a0_values_100bar_water_1 = [ms.piston_acceleration_water(y,params.m_piston_1, params.H_1, V_p0_100bar_1, params.n, params.k_s_water, p_p0) for y in sol_100bar_water_1.y.T]
a0_min_100bar_water_1 = np.min(a0_values_100bar_water_1)
a0_max_100bar_water_1 = np.max(a0_values_100bar_water_1)
v_piston_100bar_water_1 = sol_100bar_water_1.y[params.n+1, :]
v_piston_max_100bar_water_1=np.max(v_piston_100bar_water_1)

all_acc_100bar_water_1 = np.array([
    ms.all_accelerations_water(y, params.m_piston_1, params.H_1, V_t0_100bar_1, V_p0_100bar_1, params.k_s_water, params.m_e_water, params.n, p_p0)
    for y in sol_100bar_water_1.y.T
])

a_piston_100bar_water_1 = all_acc_100bar_water_1[:, 0]      
a_fluid_100bar_water_1  = all_acc_100bar_water_1[:, 1:].T   
V_p_100bar_water_1 = params.A_p * (x_p_100bar_water_1 - params.th_piston_1)
V_t_100bar_water_1 = params.A_p * (params.H_1 - x_fluid_100bar_water_1[params.n-1, :])

p_t_100bar_water_1=[]
for i in range(len(sol_100bar_water_1.t)):
    p_ti=ms.pressure_on_top(sol_100bar_water_1.y[:,i], params.H_1, params.th_piston_1, V_t0_100bar_1, params.n)
    p_t_100bar_water_1.append(p_ti)
p_t_100bar_water_1 = np.array(p_t_100bar_water_1)
p_t_100bar_water_max_1 = np.max(p_t_100bar_water_1)

p_p_100bar_water_1=[]
for i in range(len(sol_100bar_water_1.t)):
    p_pi=ms.pressure_under_piston(sol_100bar_water_1.y[:,i], params.H_1, params.th_piston_1, V_p0_100bar_1, params.n,p_p0)
    p_p_100bar_water_1.append(p_pi)
p_p_100bar_water_1 = np.array(p_p_100bar_water_1)

dx_100bar_water_1 = np.zeros((params.n, len(sol_100bar_water_1.t)))
for j in range(len(sol_100bar_water_1.t)):
    dx_100bar_water_1[0,j]=x_fluid_100bar_water_1[0,j]-x_p_100bar_water_1[j]
    for i in range (1,params.n):
        dx_100bar_water_1[i,j]=x_fluid_100bar_water_1[i,j]-x_fluid_100bar_water_1[i-1,j]
p_100bar_water_1= np.ones((params.n,len(sol_100bar_water_1.t)))*params.p_0
for i in range(params.n):
    for j in range(len(sol_100bar_water_1.t)):
        p_100bar_water_1[i,j]=params.p_0-params.rho_l_water*params.c_water**2*((dx_100bar_water_1[i,j]*params.A_p)-dx_eq_100bar_water_1[i]*params.A_p) /(dx_eq_100bar_water_1[i]*params.A_p)

#M=1.5#
x_fluid_eq_100bar_water_15= ms.equilibrate_fluid_positions(params.n,params.L0,params.k_s_water,params.m_e_water,params.m_piston_15,params.g,params.p_t0,p_p0)
for i in range(len(x_fluid_eq_100bar_water_15)):
    x_fluid_eq_100bar_water_15[i]=x_fluid_eq_100bar_water_15[i]+params.x_air_bottom
x_p0_100bar_water_15=x_fluid_eq_100bar_water_15[0]
offset= x_p0_100bar_water_15 - (params.x_air_bottom+params.th_piston_15)
for i in range(len(x_fluid_eq_100bar_water_15)):
    x_fluid_eq_100bar_water_15[i]=x_fluid_eq_100bar_water_15[i]-offset
x_p0_100bar_water_15=x_fluid_eq_100bar_water_15[0]

dx_eq_100bar_water_15 = np.zeros(params.n)
dx_eq_100bar_water_15[0]=x_fluid_eq_100bar_water_15[1]-x_p0_100bar_water_15
for i in range (1,params.n):
    dx_eq_100bar_water_15[i]=x_fluid_eq_100bar_water_15[i]-x_fluid_eq_100bar_water_15[i-1]

V_p0_100bar_15 = params.A_p * (x_p0_100bar_water_15-params.th_piston_15) 
V_t0_100bar_15 = params.A_p * (params.H_15-x_fluid_eq_100bar_water_15[params.n]) 

v_p0 = 0 
v_fluid0=np.zeros(params.n)
y0 = np.concatenate((x_fluid_eq_100bar_water_15, [v_p0], v_fluid0))

p_0_x_100bar_water_15= np.zeros(params.n)
for i in range (params.n):
    p_0_x_100bar_water_15[i] = params.p_t0 + params.rho_water*params.g*(x_fluid_eq_100bar_water_15[params.n]-x_fluid_eq_100bar_water_15[i])


sol_100bar_water_15 = solve_ivp(ms.system_water, [0, T_100], y0, args=(params.m_piston_15,params.H_15, V_t0_100bar_15, V_p0_100bar_15, params.k_s_water, params.m_e_water, params.n, p_p0), atol = 1e-6, rtol=1e-6, max_step = 1e-6)
sol_100bar_water_15.t=np.array(sol_100bar_water_15.t)

x_p_100bar_water_15 = sol_100bar_water_15.y[0]      
x_fluid_100bar_water_15 = sol_100bar_water_15.y[1:params.n+1] 
v_fluid_100bar_water_15 = sol_100bar_water_15.y[params.n+1:2*(params.n+1)] 
a0_values_100bar_water_15 = [ms.piston_acceleration_water(y,params.m_piston_15, params.H_15, V_p0_100bar_15, params.n, params.k_s_water, p_p0) for y in sol_100bar_water_15.y.T]
a0_min_100bar_water_15 = np.min(a0_values_100bar_water_15)
a0_max_100bar_water_15 = np.max(a0_values_100bar_water_15)
v_piston_100bar_water_15 = sol_100bar_water_15.y[params.n+1, :]
v_piston_max_100bar_water_15=np.max(v_piston_100bar_water_15)

all_acc_100bar_water_15 = np.array([
    ms.all_accelerations_water(y, params.m_piston_15, params.H_15, V_t0_100bar_15, V_p0_100bar_15, params.k_s_water, params.m_e_water, params.n, p_p0)
    for y in sol_100bar_water_15.y.T
])

a_piston_100bar_water_15 = all_acc_100bar_water_15[:, 0]     
a_fluid_100bar_water_15  = all_acc_100bar_water_15[:, 1:].T   
V_p_100bar_water_15 = params.A_p * (x_p_100bar_water_15 - params.th_piston_15)
V_t_100bar_water_15 = params.A_p * (params.H_15 - x_fluid_100bar_water_15[params.n-1, :])

p_t_100bar_water_15=[]
for i in range(len(sol_100bar_water_15.t)):
    p_ti=ms.pressure_on_top(sol_100bar_water_15.y[:,i], params.H_15, params.th_piston_15, V_t0_100bar_15, params.n)
    p_t_100bar_water_15.append(p_ti)
p_t_100bar_water_15 = np.array(p_t_100bar_water_15)
p_t_100bar_water_max_15 = np.max(p_t_100bar_water_15)

p_p_100bar_water_15=[]
for i in range(len(sol_100bar_water_15.t)):
    p_pi=ms.pressure_under_piston(sol_100bar_water_15.y[:,i], params.H_15, params.th_piston_15, V_p0_100bar_15, params.n,p_p0)
    p_p_100bar_water_15.append(p_pi)
p_p_100bar_water_15 = np.array(p_p_100bar_water_15)

dx_100bar_water_15 = np.zeros((params.n, len(sol_100bar_water_15.t)))
for j in range(len(sol_100bar_water_15.t)):
    dx_100bar_water_15[0,j]=x_fluid_100bar_water_15[0,j]-x_p_100bar_water_15[j]
    for i in range (1,params.n):
        dx_100bar_water_15[i,j]=x_fluid_100bar_water_15[i,j]-x_fluid_100bar_water_15[i-1,j]
p_100bar_water_15= np.ones((params.n,len(sol_100bar_water_15.t)))*params.p_0
for i in range(params.n):
    for j in range(len(sol_100bar_water_15.t)):
        p_100bar_water_15[i,j]=params.p_0-params.rho_l_water*params.c_water**2*((dx_100bar_water_15[i,j]*params.A_p)-dx_eq_100bar_water_15[i]*params.A_p) /(dx_eq_100bar_water_15[i]*params.A_p)

#M=2#
x_fluid_eq_100bar_water_2= ms.equilibrate_fluid_positions(params.n,params.L0,params.k_s_water,params.m_e_water,params.m_piston_2,params.g,params.p_t0,p_p0)
for i in range(len(x_fluid_eq_100bar_water_2)):
    x_fluid_eq_100bar_water_2[i]=x_fluid_eq_100bar_water_2[i]+params.x_air_bottom
x_p0_100bar_water_2=x_fluid_eq_100bar_water_2[0]
offset= x_p0_100bar_water_2 - (params.x_air_bottom+params.th_piston_2)
for i in range(len(x_fluid_eq_100bar_water_2)):
    x_fluid_eq_100bar_water_2[i]=x_fluid_eq_100bar_water_2[i]-offset
x_p0_100bar_water_2=x_fluid_eq_100bar_water_2[0]

dx_eq_100bar_water_2 = np.zeros(params.n)
dx_eq_100bar_water_2[0]=x_fluid_eq_100bar_water_2[1]-x_p0_100bar_water_2
for i in range (1,params.n):
    dx_eq_100bar_water_2[i]=x_fluid_eq_100bar_water_2[i]-x_fluid_eq_100bar_water_2[i-1]

V_p0_100bar_2 = params.A_p * (x_p0_100bar_water_2-params.th_piston_2)
V_t0_100bar_2 = params.A_p * (params.H_2-x_fluid_eq_100bar_water_2[params.n])  

v_p0 = 0 
v_fluid0=np.zeros(params.n)
y0 = np.concatenate((x_fluid_eq_100bar_water_2, [v_p0], v_fluid0))

p_0_x_100bar_water_2= np.zeros(params.n)
for i in range (params.n):
    p_0_x_100bar_water_2[i] = params.p_t0 + params.rho_water*params.g*(x_fluid_eq_100bar_water_2[params.n]-x_fluid_eq_100bar_water_2[i])


sol_100bar_water_2 = solve_ivp(ms.system_water, [0, T_100], y0, args=(params.m_piston_2,params.H_2, V_t0_100bar_2, V_p0_100bar_2, params.k_s_water, params.m_e_water, params.n, p_p0), atol = 1e-6, rtol=1e-6, max_step = 1e-6)
sol_100bar_water_2.t=np.array(sol_100bar_water_2.t)

x_p_100bar_water_2 = sol_100bar_water_2.y[0]      
x_fluid_100bar_water_2 = sol_100bar_water_2.y[1:params.n+1] 
v_fluid_100bar_water_2 = sol_100bar_water_2.y[params.n+1:2*(params.n+1)]
a0_values_100bar_water_2 = [ms.piston_acceleration_water(y,params.m_piston_2, params.H_2, V_p0_100bar_2, params.n, params.k_s_water, p_p0) for y in sol_100bar_water_2.y.T]
a0_min_100bar_water_2 = np.min(a0_values_100bar_water_2)
a0_max_100bar_water_2 = np.max(a0_values_100bar_water_2)
v_piston_100bar_water_2 = sol_100bar_water_2.y[params.n+1, :]
v_piston_max_100bar_water_2=np.max(v_piston_100bar_water_2)

all_acc_100bar_water_2 = np.array([
    ms.all_accelerations_water(y, params.m_piston_2, params.H_2, V_t0_100bar_2, V_p0_100bar_2, params.k_s_water, params.m_e_water, params.n, p_p0)
    for y in sol_100bar_water_2.y.T
])

a_piston_100bar_water_2 = all_acc_100bar_water_2[:, 0]     
a_fluid_100bar_water_2  = all_acc_100bar_water_2[:, 1:].T 
V_p_100bar_water_2 = params.A_p * (x_p_100bar_water_2 - params.th_piston_2)
V_t_100bar_water_2 = params.A_p * (params.H_2 - x_fluid_100bar_water_2[params.n-1, :])

p_t_100bar_water_2=[]
for i in range(len(sol_100bar_water_2.t)):
    p_ti=ms.pressure_on_top(sol_100bar_water_2.y[:,i], params.H_2, params.th_piston_2, V_t0_100bar_2, params.n)
    p_t_100bar_water_2.append(p_ti)
p_t_100bar_water_2 = np.array(p_t_100bar_water_2)
p_t_100bar_water_max_2 = np.max(p_t_100bar_water_2)

p_p_100bar_water_2=[]
for i in range(len(sol_100bar_water_2.t)):
    p_pi=ms.pressure_under_piston(sol_100bar_water_2.y[:,i], params.H_2, params.th_piston_2, V_p0_100bar_2, params.n,p_p0)
    p_p_100bar_water_2.append(p_pi)
p_p_100bar_water_2 = np.array(p_p_100bar_water_2)

dx_100bar_water_2 = np.zeros((params.n, len(sol_100bar_water_2.t)))
for j in range(len(sol_100bar_water_2.t)):
    dx_100bar_water_2[0,j]=x_fluid_100bar_water_2[0,j]-x_p_100bar_water_2[j]
    for i in range (1,params.n):
        dx_100bar_water_2[i,j]=x_fluid_100bar_water_2[i,j]-x_fluid_100bar_water_2[i-1,j]
p_100bar_water_2= np.ones((params.n,len(sol_100bar_water_2.t)))*params.p_0
for i in range(params.n):
    for j in range(len(sol_100bar_water_2.t)):
        p_100bar_water_2[i,j]=params.p_0-params.rho_l_water*params.c_water**2*((dx_100bar_water_2[i,j]*params.A_p)-dx_eq_100bar_water_2[i]*params.A_p) /(dx_eq_100bar_water_2[i]*params.A_p)

#no piston#
x_fluid_eq_100bar_water_no_piston = ms.equilibrate_fluid_positions_no_piston(params.n, params.L0, params.k_s_water, params.m_e_water, params.g, params.p_t0, p_p0)
x_0_100bar_water_no_piston = x_fluid_eq_100bar_water_no_piston[0]

dx_eq_100bar_water_no_piston = np.zeros(params.n)
dx_eq_100bar_water_no_piston[0] = x_fluid_eq_100bar_water_no_piston[1] - x_0_100bar_water_no_piston
for i in range(1, params.n):
    dx_eq_100bar_water_no_piston[i] = x_fluid_eq_100bar_water_no_piston[i] - x_fluid_eq_100bar_water_no_piston[i - 1]

V_p0 = params.A_p * x_0_100bar_water_no_piston  
V_t0 = params.A_p * (params.H_no_piston - x_fluid_eq_100bar_water_no_piston[params.n]) 

v_0 = 0  
v_fluid0 = np.zeros(params.n)
y0 = np.concatenate((x_fluid_eq_100bar_water_no_piston, [v_0], v_fluid0))


sol_100bar_water_no_piston = solve_ivp(ms.system_water_no_piston, [0, T_100], y0, args=(V_t0,V_p0, p_p0), atol=1e-6, rtol=1e-6, max_step=1e-6)
sol_100bar_water_no_piston.t=np.array(sol_100bar_water_no_piston.t)

x_0_100bar_water_no_piston = sol_100bar_water_no_piston.y[0]  
x_fluid_100bar_water_no_piston = sol_100bar_water_no_piston.y[1:params.n + 1] 
v_fluid_100bar_water_no_piston = sol_100bar_water_no_piston.y[params.n + 1:2 * (params.n + 1)]  
a0_values_100bar_water_no_piston = [ms.liquid_acceleration_water(y, params.n, params.k_s_water, params.m_e_water, params.L0, V_p0, p_p0) for y in sol_100bar_water_no_piston.y.T]
a0_min_100bar_water_no_piston = np.min(a0_values_100bar_water_no_piston)
a0_max_100bar_water_no_piston = np.max(a0_values_100bar_water_no_piston)
v_piston_100bar_water_no_piston = sol_100bar_water_no_piston.y[params.n + 1, :]
v_piston_max_100bar_water_no_piston = np.max(v_piston_100bar_water_no_piston)

all_acc_100bar_water_no_piston = np.array([
    ms.all_accelerations_water_no_piston(y, V_t0, V_p0, p_p0)
    for y in sol_100bar_water_no_piston.y.T
])  

a_fluid_100bar_water_no_piston = all_acc_100bar_water_no_piston.T  
V_p_100bar_water_no_piston = params.A_p * x_0_100bar_water_no_piston
V_t_100bar_water_no_piston = params.A_p * (params.H_no_piston - x_fluid_100bar_water_no_piston[params.n-1, :])

p_t_100bar_water_no_piston = []
for i in range(len(sol_100bar_water_no_piston.t)):
    p_ti = ms.pressure_on_top_no_piston(sol_100bar_water_no_piston.y[:, i], V_t0)
    p_t_100bar_water_no_piston.append(p_ti)


p_p_100bar_water_no_piston = []
for i in range(len(sol_100bar_water_no_piston.t)):
    p_pi = ms.pressure_gas_bottom(sol_100bar_water_no_piston.y[:, i], V_p0, p_p0)
    p_p_100bar_water_no_piston.append(p_pi)


dx_100bar_water_no_piston = np.zeros((params.n, len(sol_100bar_water_no_piston.t)))
for j in range(len(sol_100bar_water_no_piston.t)):
    dx_100bar_water_no_piston[0, j] = x_fluid_100bar_water_no_piston[0, j] - x_0_100bar_water_no_piston[j]
    for i in range(1, params.n):
        dx_100bar_water_no_piston[i, j] = x_fluid_100bar_water_no_piston[i, j] - x_fluid_100bar_water_no_piston[i - 1, j]

p_100bar_water_no_piston = np.ones((params.n, len(sol_100bar_water_no_piston.t))) * params.p_0
for i in range(params.n):
    for j in range(len(sol_100bar_water_no_piston.t)):
        p_100bar_water_no_piston[i, j] = params.p_0 - params.rho_l_water * params.c_water**2 * ((dx_100bar_water_no_piston[i, j] * params.A_p) - dx_eq_100bar_water_no_piston[i] * params.A_p) / (dx_eq_100bar_water_no_piston[i] * params.A_p)

#lithium#
#M=1.69#
x_fluid_eq_100bar_lithium= ms.equilibrate_fluid_positions(params.n,params.L0,params.k_s_lithium,params.m_e_lithium,params.m_piston_ref,params.g,params.p_t0,p_p0)
for i in range(len(x_fluid_eq_100bar_lithium)):
    x_fluid_eq_100bar_lithium[i]=x_fluid_eq_100bar_lithium[i]+params.x_air_bottom
x_p0_100bar_lithium=x_fluid_eq_100bar_lithium[0]
offset= x_p0_100bar_lithium - 0.25
for i in range(len(x_fluid_eq_100bar_lithium)):
    x_fluid_eq_100bar_lithium[i]=x_fluid_eq_100bar_lithium[i]-offset
x_p0_100bar_lithium=x_fluid_eq_100bar_lithium[0]

dx_eq_100bar_lithium= np.zeros(params.n)
dx_eq_100bar_lithium[0]=x_fluid_eq_100bar_lithium[1]-x_p0_100bar_lithium
for i in range (1,params.n):
    dx_eq_100bar_lithium[i]=x_fluid_eq_100bar_lithium[i]-x_fluid_eq_100bar_lithium[i-1]

V_p0 = params.A_p * (x_p0_100bar_lithium-params.th_piston_ref) 
V_t0 = params.A_p * (params.H_ref-x_fluid_eq_100bar_lithium[params.n])  

v_p0 = 0 
v_fluid0=np.zeros(params.n)
y0 = np.concatenate((x_fluid_eq_100bar_lithium, [v_p0], v_fluid0))


sol_100bar_lithium = solve_ivp(ms.system_lith, [0, T_100], y0, args=(params.m_piston_ref,params.H_ref, V_t0, V_p0, params.n, p_p0), atol = 1e-6, rtol=1e-6, max_step = 1e-6)


x_p_100bar_lithium = sol_100bar_lithium.y[0]      
x_fluid_100bar_lithium = sol_100bar_lithium.y[1:params.n+1]
v_fluid_100bar_lithium = sol_100bar_lithium.y[params.n+1:2*(params.n+1)] 
a0_values_100bar_lithium = [ms.piston_acceleration_lith(y, params.m_piston_ref, V_p0, p_p0) for y in sol_100bar_lithium.y.T]
a0_min_100bar_lithium = np.min(a0_values_100bar_lithium)
a0_max_100bar_lithium = np.max(a0_values_100bar_lithium)
v_piston_100bar_lithium = sol_100bar_lithium.y[params.n+1, :]
v_piston_max_100bar_lithium=np.max(v_piston_100bar_lithium)


p_t_100bar_lithium=[]
for i in range(len(sol_100bar_lithium.t)):
    p_ti=ms.pressure_on_top(sol_100bar_lithium.y[:,i], params.H_ref, params.th_piston_ref, V_t0, params.n)
    p_t_100bar_lithium.append(p_ti)


p_p_100bar_lithium=[]
for i in range(len(sol_100bar_lithium.t)):
    p_pi=ms.pressure_under_piston(sol_100bar_lithium.y[:,i], params.H_ref, params.th_piston_ref, V_p0, params.n,p_p0)
    p_p_100bar_lithium.append(p_pi)


dx_100bar_lithium = np.zeros((params.n, len(sol_100bar_lithium.t)))
for j in range(len(sol_100bar_lithium.t)):
    dx_100bar_lithium[0,j]=x_fluid_100bar_lithium[0,j]-x_p_100bar_lithium[j]
    for i in range (1,params.n):
        dx_100bar_lithium[i,j]=x_fluid_100bar_lithium[i,j]-x_fluid_100bar_lithium[i-1,j]
p_100bar_lithium= np.ones((params.n,len(sol_100bar_lithium.t)))*params.p_0
for i in range(params.n):
    for j in range(len(sol_100bar_lithium.t)):
        p_100bar_lithium[i,j]=params.p_0-params.rho_l_lithium*params.c_lithium**2*((dx_100bar_lithium[i,j]*params.A_p)-dx_eq_100bar_lithium[i]*params.A_p) /(dx_eq_100bar_lithium[i]*params.A_p)