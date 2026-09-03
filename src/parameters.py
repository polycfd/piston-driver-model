import numpy as np
from math import pi
import matplotlib.cm as cm

# ============================================================
# SYSTEM PARAMETERS
# ============================================================

rho_water = 1000 #[kg/m^3]
rho_lithium = 531 #[kg/m^3]
rho_aluminium = 2698 #[kg/m^3]
g=9.81
c_water=1500 #[m/s]
c_lithium = 4544 #[m/s]
gamma=1.4
R=286.984 #[J/kg.K]
cv=717.46 #[J/kg.K] 
cp = R + cv #[J/kg.K]

#DIMENSIONS#

#n=100
h=0.15 #[m] - length of the water column
dx_target=0.0015 # [m] - lenght of one element
n = max(1, int(round(h / dx_target)))
L0 = h / n  
spatial_coordinates = [i * h / (n - 1) for i in range(n)]
spatial_coordinates=np.array(spatial_coordinates)

mass_ratio_target_01 = 0.1
th_piston_01 = mass_ratio_target_01* h * rho_water/rho_aluminium # [m] - piston thickness
mass_ratio_target_05 = 0.5
th_piston_05 = mass_ratio_target_05* h * rho_water/rho_aluminium
mass_ratio_target_09 =0.9
th_piston_09=mass_ratio_target_09 * h * rho_water/rho_aluminium
mass_ratio_target_1 = 1
th_piston_1 = mass_ratio_target_1* h * rho_water/rho_aluminium
mass_ratio_target_15 = 1.5
th_piston_15 = mass_ratio_target_15* h * rho_water/rho_aluminium
mass_ratio_target_2 = 2
th_piston_2 = mass_ratio_target_2* h * rho_water/rho_aluminium

x_air_bottom = 0.2 # [m] - lower gas volume initial thickness
h_gt = 0.1 # [m] - upper gas volume initial thickness

H_01=h+th_piston_01+h_gt+x_air_bottom #[m] - cylinder length
H_05=h+th_piston_05+h_gt+x_air_bottom
H_09=h+th_piston_09+h_gt+x_air_bottom
H_1=h+th_piston_1+h_gt+x_air_bottom 
H_15=h+th_piston_15+h_gt+x_air_bottom 
H_2=h+th_piston_2+h_gt+x_air_bottom 
H_no_piston=h+h_gt+x_air_bottom
d_cyl = 0.0381 # [m] cylinder diameter
A_p=pi*(d_cyl/2)**2 # [m^2] piston area
V_0 = A_p * L0 

m_w_water=h*A_p*rho_water #[kg] - water column mass
m_e_water=m_w_water/n #[kg] - mass of one element (water)
m_w_lithium=h*A_p*rho_lithium #[kg] - lithium column mass
m_e_lithium=m_w_lithium/n #[kg] - mass of one element (lithium)
m_piston_05=th_piston_05*A_p*rho_aluminium #[kg] - piston mass
m_piston_01=th_piston_01*A_p*rho_aluminium 
m_piston_09=th_piston_09*A_p*rho_aluminium
m_piston_1=th_piston_1*A_p*rho_aluminium
m_piston_15=th_piston_15*A_p*rho_aluminium 
m_piston_2=th_piston_2*A_p*rho_aluminium 


k_s_water=(m_e_water*(c_water**2))/(L0**2) #water spring stiffness
rho_l_water = m_e_water / V_0
k_s_lithium=(m_e_lithium*(c_lithium**2))/(L0**2) #lithium spring stiffness
rho_l_lithium = m_e_lithium / V_0

#reference configuration#
th_piston_ref = 0.05
m_piston_ref = th_piston_ref * A_p * rho_aluminium
H_ref= th_piston_ref+h+h_gt+x_air_bottom

#Pressures
p_t0=1e5 #[Pa] - upper gas initial pressure
p_p0=p_t0 #[Pa] - lower gas initial pressure
p_0=p_t0 #[Pa] - liquid initial pressure

pressures = [1, 2, 3, 4, 5, 6, 7.5, 10] #[MPa]
cmap = cm.coolwarm
colors = [cmap(i / (len(pressures) - 1)) for i in range(len(pressures))]
M_list=[0, 0.1, 0.5, 1, 1.5, 2]