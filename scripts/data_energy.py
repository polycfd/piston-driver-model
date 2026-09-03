import numpy as np
from src.parameters import A_p
from src.parameters import n
from src import parameters as param
from src import energy_balance as en
from scripts import run_parametric_study as param_study

# ==============================================================================================================================
# CONTROL-VOLUME ENERGY AND ITS 3 CONTRIBUTIONS (E_piston, E_in and E_out) COMPUTED FOR THE run_parametric_study SIMULATIONS
# ==============================================================================================================================

#time for the two first compression peaks#
#1 MPa driving pressure#
t_first_peak_10bar_water_01=0.0068766747
t_second_peak_10bar_water_01=0.0206892159
t_first_peak_10bar_water_05=0.00807557
t_second_peak_10bar_water_05=0.02406741
t_first_peak_10bar_water_1=0.0092641
t_second_peak_10bar_water_1=0.02786797
t_first_peak_10bar_water_15=0.01040442
t_second_peak_10bar_water_15=0.03121065
t_first_peak_10bar_water_2=0.01142168
t_second_peak_10bar_water_2=0.03419141
t_first_peak_10bar_water_no_piston=0.006536362
t_second_peak_10bar_water_no_piston=0.0198658811

#2 MPa driving pressure#
t_first_peak_20bar_water_01=0.00463716
t_second_peak_20bar_water_01=0.01386422
t_first_peak_20bar_water_05=0.00541723
t_second_peak_20bar_water_05=0.01623299
t_first_peak_20bar_water_1=0.00626245
t_second_peak_20bar_water_1=0.01878372
t_first_peak_20bar_water_15=0.00700542
t_second_peak_20bar_water_15=0.02102342
t_first_peak_20bar_water_2=0.00767222
t_second_peak_20bar_water_2=0.02302085
t_first_peak_20bar_water_no_piston=0.00443075
t_second_peak_20bar_water_no_piston=0.01330120

#3 MPa driving pressure#
t_first_peak_30bar_water_01=0.00373128
t_second_peak_30bar_water_01=0.01117822
t_first_peak_30bar_water_05=0.00436836
t_second_peak_30bar_water_05=0.01311233
t_first_peak_30bar_water_1=0.00505642
t_second_peak_30bar_water_1=0.01515195
t_first_peak_30bar_water_15=0.00565026
t_second_peak_30bar_water_15=0.01696944
t_first_peak_30bar_water_2=0.0061885
t_second_peak_30bar_water_2=0.01860721
t_first_peak_30bar_water_no_piston=0.00356234
t_second_peak_30bar_water_no_piston=0.01069134

#4 MPa driving pressure#
t_first_peak_40bar_water_01=0.00320913
t_second_peak_40bar_water_01=0.009619958
t_first_peak_40bar_water_05=0.00379031
t_second_peak_40bar_water_05=0.01132087
t_first_peak_40bar_water_1=0.00436821
t_second_peak_40bar_water_1=0.01310301
t_first_peak_40bar_water_15=0.00487541
t_second_peak_40bar_water_15=0.0145975
t_first_peak_40bar_water_2=0.00533241
t_second_peak_40bar_water_2=0.01604724
t_first_peak_40bar_water_no_piston=0.00308147
t_second_peak_40bar_water_no_piston=0.00924162

#5 MPa driving pressure#
t_first_peak_50bar_water_01=0.00286551
t_second_peak_50bar_water_01=0.00859142
t_first_peak_50bar_water_05=0.00339641
t_second_peak_50bar_water_05=0.01013135
t_first_peak_50bar_water_1=0.0039035
t_second_peak_50bar_water_1=0.01169147
t_first_peak_50bar_water_15=0.00435864
t_second_peak_50bar_water_15=0.01304859
t_first_peak_50bar_water_2=0.00476798
t_second_peak_50bar_water_2=0.01438095
t_first_peak_50bar_water_no_piston=0.00274631
t_second_peak_50bar_water_no_piston=0.008239575

#6 MPa driving pressure#
t_first_peak_60bar_water_01=0.00260603
t_second_peak_60bar_water_01=0.00789892
t_first_peak_60bar_water_05=0.00310625
t_second_peak_60bar_water_05=0.00924325
t_first_peak_60bar_water_1=0.00356649
t_second_peak_60bar_water_1=0.01068415
t_first_peak_60bar_water_15=0.00397978
t_second_peak_60bar_water_15=0.01195187
t_first_peak_60bar_water_2=0.00434905
t_second_peak_60bar_water_2=0.01321142
t_first_peak_60bar_water_no_piston=0.002491930
t_second_peak_60bar_water_no_piston=0.00751545

#7.5 MPa driving pressure#
t_first_peak_75bar_water_01=0.00233927
t_second_peak_75bar_water_01=0.00708179
t_first_peak_75bar_water_05=0.00278884
t_second_peak_75bar_water_05=0.008296224
t_first_peak_75bar_water_1=0.003199194
t_second_peak_75bar_water_1=0.009505402
t_first_peak_75bar_water_15=0.00356605
t_second_peak_75bar_water_15=0.01076053
t_first_peak_75bar_water_2=0.0038909
t_second_peak_75bar_water_2=0.01178046
t_first_peak_75bar_water_no_piston=0.00222732
t_second_peak_75bar_water_no_piston=0.00673320

#10 MPa driving pressure#
t_first_peak_100bar_water_01=0.00205366
t_second_peak_100bar_water_01=0.00612969
t_first_peak_100bar_water_05=0.00242707
t_second_peak_100bar_water_05=0.007189384
t_first_peak_100bar_water_1=0.00278547
t_second_peak_100bar_water_1=0.00834022
t_first_peak_100bar_water_15=0.00309475
t_second_peak_100bar_water_15=0.00937058
t_first_peak_100bar_water_2= 0.00348000
t_second_peak_100bar_water_2=0.01029514
t_first_peak_100bar_water_no_piston=0.00193083
t_second_peak_100bar_water_no_piston=0.00584614

#total energy, M=0#
t_10bar_no_piston, E_total_10bar_no_piston = en.compute_energy_balance_no_piston(param_study.sol_10bar_water_no_piston.t, param_study.p_t_10bar_water_no_piston, param_study.V_t_10bar_water_no_piston, param_study.p_p_10bar_water_no_piston, param_study.V_p_10bar_water_no_piston, param_study.v_fluid_10bar_water_no_piston, param_study.x_0_10bar_water_no_piston, param_study.x_fluid_10bar_water_no_piston)
E_ref_10bar_no_piston = E_total_10bar_no_piston[0]
t_20bar_no_piston, E_total_20bar_no_piston = en.compute_energy_balance_no_piston(param_study.sol_20bar_water_no_piston.t, param_study.p_t_20bar_water_no_piston, param_study.V_t_20bar_water_no_piston, param_study.p_p_20bar_water_no_piston, param_study.V_p_20bar_water_no_piston, param_study.v_fluid_20bar_water_no_piston, param_study.x_0_20bar_water_no_piston, param_study.x_fluid_20bar_water_no_piston)
E_ref_20bar_no_piston = E_total_20bar_no_piston[0]
t_30bar_no_piston, E_total_30bar_no_piston = en.compute_energy_balance_no_piston(param_study.sol_30bar_water_no_piston.t, param_study.p_t_30bar_water_no_piston, param_study.V_t_30bar_water_no_piston, param_study.p_p_30bar_water_no_piston, param_study.V_p_30bar_water_no_piston, param_study.v_fluid_30bar_water_no_piston, param_study.x_0_30bar_water_no_piston, param_study.x_fluid_30bar_water_no_piston)
E_ref_30bar_no_piston = E_total_30bar_no_piston[0]
t_40bar_no_piston, E_total_40bar_no_piston = en.compute_energy_balance_no_piston(param_study.sol_40bar_water_no_piston.t, param_study.p_t_40bar_water_no_piston, param_study.V_t_40bar_water_no_piston, param_study.p_p_40bar_water_no_piston, param_study.V_p_40bar_water_no_piston, param_study.v_fluid_40bar_water_no_piston, param_study.x_0_40bar_water_no_piston, param_study.x_fluid_40bar_water_no_piston)
E_ref_40bar_no_piston = E_total_40bar_no_piston[0]
t_50bar_no_piston, E_total_50bar_no_piston = en.compute_energy_balance_no_piston(param_study.sol_50bar_water_no_piston.t, param_study.p_t_50bar_water_no_piston, param_study.V_t_50bar_water_no_piston, param_study.p_p_50bar_water_no_piston, param_study.V_p_50bar_water_no_piston, param_study.v_fluid_50bar_water_no_piston, param_study.x_0_50bar_water_no_piston, param_study.x_fluid_50bar_water_no_piston)
E_ref_50bar_no_piston = E_total_50bar_no_piston[0]
t_60bar_no_piston, E_total_60bar_no_piston = en.compute_energy_balance_no_piston(param_study.sol_60bar_water_no_piston.t, param_study.p_t_60bar_water_no_piston, param_study.V_t_60bar_water_no_piston, param_study.p_p_60bar_water_no_piston, param_study.V_p_60bar_water_no_piston, param_study.v_fluid_60bar_water_no_piston, param_study.x_0_60bar_water_no_piston, param_study.x_fluid_60bar_water_no_piston)
E_ref_60bar_no_piston = E_total_60bar_no_piston[0]
t_75bar_no_piston, E_total_75bar_no_piston = en.compute_energy_balance_no_piston(param_study.sol_75bar_water_no_piston.t, param_study.p_t_75bar_water_no_piston, param_study.V_t_75bar_water_no_piston, param_study.p_p_75bar_water_no_piston, param_study.V_p_75bar_water_no_piston, param_study.v_fluid_75bar_water_no_piston, param_study.x_0_75bar_water_no_piston, param_study.x_fluid_75bar_water_no_piston)
E_ref_75bar_no_piston = E_total_75bar_no_piston[0]
t_100bar_no_piston, E_total_100bar_no_piston = en.compute_energy_balance_no_piston(param_study.sol_100bar_water_no_piston.t, param_study.p_t_100bar_water_no_piston, param_study.V_t_100bar_water_no_piston, param_study.p_p_100bar_water_no_piston, param_study.V_p_100bar_water_no_piston, param_study.v_fluid_100bar_water_no_piston, param_study.x_0_100bar_water_no_piston, param_study.x_fluid_100bar_water_no_piston)
E_ref_100bar_no_piston = E_total_100bar_no_piston[0]

#total energy, M=0.1#
t_10bar_M01, E_total_10bar_M01 = en.compute_energy_balance(param_study.sol_10bar_water_01.t, param_study.p_t_10bar_water_01, param_study.V_t_10bar_water_01, param_study.p_p_10bar_water_01, param_study.V_p_10bar_water_01, param.m_piston_01, param_study.v_piston_10bar_water_01, param_study.v_fluid_10bar_water_01, param_study.x_p_10bar_water_01, param_study.x_fluid_10bar_water_01)
E_ref_10bar_M01 = E_total_10bar_M01[0]
t_20bar_M01, E_total_20bar_M01 = en.compute_energy_balance(param_study.sol_20bar_water_01.t, param_study.p_t_20bar_water_01, param_study.V_t_20bar_water_01, param_study.p_p_20bar_water_01, param_study.V_p_20bar_water_01, param.m_piston_01, param_study.v_piston_20bar_water_01, param_study.v_fluid_20bar_water_01, param_study.x_p_20bar_water_01, param_study.x_fluid_20bar_water_01)
E_ref_20bar_M01 = E_total_20bar_M01[0]
t_30bar_M01, E_total_30bar_M01 = en.compute_energy_balance(param_study.sol_30bar_water_01.t, param_study.p_t_30bar_water_01, param_study.V_t_30bar_water_01, param_study.p_p_30bar_water_01, param_study.V_p_30bar_water_01, param.m_piston_01, param_study.v_piston_30bar_water_01, param_study.v_fluid_30bar_water_01, param_study.x_p_30bar_water_01, param_study.x_fluid_30bar_water_01)
E_ref_30bar_M01 = E_total_30bar_M01[0]
t_40bar_M01, E_total_40bar_M01 = en.compute_energy_balance(param_study.sol_40bar_water_01.t, param_study.p_t_40bar_water_01, param_study.V_t_40bar_water_01, param_study.p_p_40bar_water_01, param_study.V_p_40bar_water_01, param.m_piston_01, param_study.v_piston_40bar_water_01, param_study.v_fluid_40bar_water_01, param_study.x_p_40bar_water_01, param_study.x_fluid_40bar_water_01)
E_ref_40bar_M01 = E_total_40bar_M01[0]
t_50bar_M01, E_total_50bar_M01 = en.compute_energy_balance(param_study.sol_50bar_water_01.t, param_study.p_t_50bar_water_01, param_study.V_t_50bar_water_01, param_study.p_p_50bar_water_01, param_study.V_p_50bar_water_01, param.m_piston_01, param_study.v_piston_50bar_water_01, param_study.v_fluid_50bar_water_01, param_study.x_p_50bar_water_01, param_study.x_fluid_50bar_water_01)
E_ref_50bar_M01 = E_total_50bar_M01[0]
t_60bar_M01, E_total_60bar_M01 = en.compute_energy_balance(param_study.sol_60bar_water_01.t, param_study.p_t_60bar_water_01, param_study.V_t_60bar_water_01, param_study.p_p_60bar_water_01, param_study.V_p_60bar_water_01, param.m_piston_01, param_study.v_piston_60bar_water_01, param_study.v_fluid_60bar_water_01, param_study.x_p_60bar_water_01, param_study.x_fluid_60bar_water_01)
E_ref_60bar_M01 = E_total_60bar_M01[0]
t_75bar_M01, E_total_75bar_M01 = en.compute_energy_balance(param_study.sol_75bar_water_01.t, param_study.p_t_75bar_water_01, param_study.V_t_75bar_water_01, param_study.p_p_75bar_water_01, param_study.V_p_75bar_water_01, param.m_piston_01, param_study.v_piston_75bar_water_01, param_study.v_fluid_75bar_water_01, param_study.x_p_75bar_water_01, param_study.x_fluid_75bar_water_01)
E_ref_75bar_M01 = E_total_75bar_M01[0]
t_100bar_M01, E_total_100bar_M01 = en.compute_energy_balance(param_study.sol_100bar_water_01.t, param_study.p_t_100bar_water_01, param_study.V_t_100bar_water_01, param_study.p_p_100bar_water_01, param_study.V_p_100bar_water_01, param.m_piston_01, param_study.v_piston_100bar_water_01, param_study.v_fluid_100bar_water_01, param_study.x_p_100bar_water_01, param_study.x_fluid_100bar_water_01)
E_ref_100bar_M01 = E_total_100bar_M01[0]

#total energy, M=0.5#
t_10bar_M05, E_total_10bar_M05 = en.compute_energy_balance(param_study.sol_10bar_water_05.t, param_study.p_t_10bar_water_05, param_study.V_t_10bar_water_05, param_study.p_p_10bar_water_05, param_study.V_p_10bar_water_05, param.m_piston_05, param_study.v_piston_10bar_water_05, param_study.v_fluid_10bar_water_05, param_study.x_p_10bar_water_05, param_study.x_fluid_10bar_water_05)
E_ref_10bar_M05 = E_total_10bar_M05[0]
t_20bar_M05, E_total_20bar_M05 = en.compute_energy_balance(param_study.sol_20bar_water_05.t, param_study.p_t_20bar_water_05, param_study.V_t_20bar_water_05, param_study.p_p_20bar_water_05, param_study.V_p_20bar_water_05, param.m_piston_05, param_study.v_piston_20bar_water_05, param_study.v_fluid_20bar_water_05, param_study.x_p_20bar_water_05, param_study.x_fluid_20bar_water_05)
E_ref_20bar_M05 = E_total_20bar_M05[0]
t_30bar_M05, E_total_30bar_M05 = en.compute_energy_balance(param_study.sol_30bar_water_05.t, param_study.p_t_30bar_water_05, param_study.V_t_30bar_water_05, param_study.p_p_30bar_water_05, param_study.V_p_30bar_water_05, param.m_piston_05, param_study.v_piston_30bar_water_05, param_study.v_fluid_30bar_water_05, param_study.x_p_30bar_water_05, param_study.x_fluid_30bar_water_05)
E_ref_30bar_M05 = E_total_30bar_M05[0]
t_40bar_M05, E_total_40bar_M05 = en.compute_energy_balance(param_study.sol_40bar_water_05.t, param_study.p_t_40bar_water_05, param_study.V_t_40bar_water_05, param_study.p_p_40bar_water_05, param_study.V_p_40bar_water_05, param.m_piston_05, param_study.v_piston_40bar_water_05, param_study.v_fluid_40bar_water_05, param_study.x_p_40bar_water_05, param_study.x_fluid_40bar_water_05)
E_ref_40bar_M05 = E_total_40bar_M05[0]
t_50bar_M05, E_total_50bar_M05 = en.compute_energy_balance(param_study.sol_50bar_water_05.t, param_study.p_t_50bar_water_05, param_study.V_t_50bar_water_05, param_study.p_p_50bar_water_05, param_study.V_p_50bar_water_05, param.m_piston_05, param_study.v_piston_50bar_water_05, param_study.v_fluid_50bar_water_05, param_study.x_p_50bar_water_05, param_study.x_fluid_50bar_water_05)
E_ref_50bar_M05 = E_total_50bar_M05[0]
t_60bar_M05, E_total_60bar_M05 = en.compute_energy_balance(param_study.sol_60bar_water_05.t, param_study.p_t_60bar_water_05, param_study.V_t_60bar_water_05, param_study.p_p_60bar_water_05, param_study.V_p_60bar_water_05, param.m_piston_05, param_study.v_piston_60bar_water_05, param_study.v_fluid_60bar_water_05, param_study.x_p_60bar_water_05, param_study.x_fluid_60bar_water_05)
E_ref_60bar_M05 = E_total_60bar_M05[0]
t_75bar_M05, E_total_75bar_M05 = en.compute_energy_balance(param_study.sol_75bar_water_05.t, param_study.p_t_75bar_water_05, param_study.V_t_75bar_water_05, param_study.p_p_75bar_water_05, param_study.V_p_75bar_water_05, param.m_piston_05, param_study.v_piston_75bar_water_05, param_study.v_fluid_75bar_water_05, param_study.x_p_75bar_water_05, param_study.x_fluid_75bar_water_05)
E_ref_75bar_M05 = E_total_75bar_M05[0]
t_100bar_M05, E_total_100bar_M05 = en.compute_energy_balance(param_study.sol_100bar_water_05.t, param_study.p_t_100bar_water_05, param_study.V_t_100bar_water_05, param_study.p_p_100bar_water_05, param_study.V_p_100bar_water_05, param.m_piston_05, param_study.v_piston_100bar_water_05, param_study.v_fluid_100bar_water_05, param_study.x_p_100bar_water_05, param_study.x_fluid_100bar_water_05)
E_ref_100bar_M05 = E_total_100bar_M05[0]

#total energy, M=0.9#
t_10bar_M09, E_total_10bar_M09 = en.compute_energy_balance(param_study.sol_10bar_water_09.t, param_study.p_t_10bar_water_09, param_study.V_t_10bar_water_09, param_study.p_p_10bar_water_09, param_study.V_p_10bar_water_09, param.m_piston_09, param_study.v_piston_10bar_water_09, param_study.v_fluid_10bar_water_09, param_study.x_p_10bar_water_09, param_study.x_fluid_10bar_water_09)
E_ref_10bar_M09 = E_total_10bar_M09[0]
t_20bar_M09, E_total_20bar_M09 = en.compute_energy_balance(param_study.sol_20bar_water_09.t, param_study.p_t_20bar_water_09, param_study.V_t_20bar_water_09, param_study.p_p_20bar_water_09, param_study.V_p_20bar_water_09, param.m_piston_09, param_study.v_piston_20bar_water_09, param_study.v_fluid_20bar_water_09, param_study.x_p_20bar_water_09, param_study.x_fluid_20bar_water_09)
E_ref_20bar_M09 = E_total_20bar_M09[0]
t_30bar_M09, E_total_30bar_M09 = en.compute_energy_balance(param_study.sol_30bar_water_09.t, param_study.p_t_30bar_water_09, param_study.V_t_30bar_water_09, param_study.p_p_30bar_water_09, param_study.V_p_30bar_water_09, param.m_piston_09, param_study.v_piston_30bar_water_09, param_study.v_fluid_30bar_water_09, param_study.x_p_30bar_water_09, param_study.x_fluid_30bar_water_09)
E_ref_30bar_M09 = E_total_30bar_M09[0]
t_40bar_M09, E_total_40bar_M09 = en.compute_energy_balance(param_study.sol_40bar_water_09.t, param_study.p_t_40bar_water_09, param_study.V_t_40bar_water_09, param_study.p_p_40bar_water_09, param_study.V_p_40bar_water_09, param.m_piston_09, param_study.v_piston_40bar_water_09, param_study.v_fluid_40bar_water_09, param_study.x_p_40bar_water_09, param_study.x_fluid_40bar_water_09)
E_ref_40bar_M09 = E_total_40bar_M09[0]
t_50bar_M09, E_total_50bar_M09 = en.compute_energy_balance(param_study.sol_50bar_water_09.t, param_study.p_t_50bar_water_09, param_study.V_t_50bar_water_09, param_study.p_p_50bar_water_09, param_study.V_p_50bar_water_09, param.m_piston_09, param_study.v_piston_50bar_water_09, param_study.v_fluid_50bar_water_09, param_study.x_p_50bar_water_09, param_study.x_fluid_50bar_water_09)
E_ref_50bar_M09 = E_total_50bar_M09[0]
t_60bar_M09, E_total_60bar_M09 = en.compute_energy_balance(param_study.sol_60bar_water_09.t, param_study.p_t_60bar_water_09, param_study.V_t_60bar_water_09, param_study.p_p_60bar_water_09, param_study.V_p_60bar_water_09, param.m_piston_09, param_study.v_piston_60bar_water_09, param_study.v_fluid_60bar_water_09, param_study.x_p_60bar_water_09, param_study.x_fluid_60bar_water_09)
E_ref_60bar_M09 = E_total_60bar_M09[0]
t_75bar_M09, E_total_75bar_M09 = en.compute_energy_balance(param_study.sol_75bar_water_09.t, param_study.p_t_75bar_water_09, param_study.V_t_75bar_water_09, param_study.p_p_75bar_water_09, param_study.V_p_75bar_water_09, param.m_piston_09, param_study.v_piston_75bar_water_09, param_study.v_fluid_75bar_water_09, param_study.x_p_75bar_water_09, param_study.x_fluid_75bar_water_09)
E_ref_75bar_M09 = E_total_75bar_M09[0]
t_100bar_M09, E_total_100bar_M09 = en.compute_energy_balance(param_study.sol_100bar_water_09.t, param_study.p_t_100bar_water_09, param_study.V_t_100bar_water_09, param_study.p_p_100bar_water_09, param_study.V_p_100bar_water_09, param.m_piston_09, param_study.v_piston_100bar_water_09, param_study.v_fluid_100bar_water_09, param_study.x_p_100bar_water_09, param_study.x_fluid_100bar_water_09)
E_ref_100bar_M09 = E_total_100bar_M09[0]

#total energy, M=1#
t_10bar_M1, E_total_10bar_M1 = en.compute_energy_balance(param_study.sol_10bar_water_1.t, param_study.p_t_10bar_water_1, param_study.V_t_10bar_water_1, param_study.p_p_10bar_water_1, param_study.V_p_10bar_water_1, param.m_piston_1, param_study.v_piston_10bar_water_1, param_study.v_fluid_10bar_water_1, param_study.x_p_10bar_water_1, param_study.x_fluid_10bar_water_1)
E_ref_10bar_M1 = E_total_10bar_M1[0]
t_20bar_M1, E_total_20bar_M1 = en.compute_energy_balance(param_study.sol_20bar_water_1.t, param_study.p_t_20bar_water_1, param_study.V_t_20bar_water_1, param_study.p_p_20bar_water_1, param_study.V_p_20bar_water_1, param.m_piston_1, param_study.v_piston_20bar_water_1, param_study.v_fluid_20bar_water_1, param_study.x_p_20bar_water_1, param_study.x_fluid_20bar_water_1)
E_ref_20bar_M1 = E_total_20bar_M1[0]
t_30bar_M1, E_total_30bar_M1 = en.compute_energy_balance(param_study.sol_30bar_water_1.t, param_study.p_t_30bar_water_1, param_study.V_t_30bar_water_1, param_study.p_p_30bar_water_1, param_study.V_p_30bar_water_1, param.m_piston_1, param_study.v_piston_30bar_water_1, param_study.v_fluid_30bar_water_1, param_study.x_p_30bar_water_1, param_study.x_fluid_30bar_water_1)
E_ref_30bar_M1 = E_total_30bar_M1[0]
t_40bar_M1, E_total_40bar_M1 = en.compute_energy_balance(param_study.sol_40bar_water_1.t, param_study.p_t_40bar_water_1, param_study.V_t_40bar_water_1, param_study.p_p_40bar_water_1, param_study.V_p_40bar_water_1, param.m_piston_1, param_study.v_piston_40bar_water_1, param_study.v_fluid_40bar_water_1, param_study.x_p_40bar_water_1, param_study.x_fluid_40bar_water_1)
E_ref_40bar_M1 = E_total_40bar_M1[0]
t_50bar_M1, E_total_50bar_M1 = en.compute_energy_balance(param_study.sol_50bar_water_1.t, param_study.p_t_50bar_water_1, param_study.V_t_50bar_water_1, param_study.p_p_50bar_water_1, param_study.V_p_50bar_water_1, param.m_piston_1, param_study.v_piston_50bar_water_1, param_study.v_fluid_50bar_water_1, param_study.x_p_50bar_water_1, param_study.x_fluid_50bar_water_1)
E_ref_50bar_M1 = E_total_50bar_M1[0]
t_60bar_M1, E_total_60bar_M1 = en.compute_energy_balance(param_study.sol_60bar_water_1.t, param_study.p_t_60bar_water_1, param_study.V_t_60bar_water_1, param_study.p_p_60bar_water_1, param_study.V_p_60bar_water_1, param.m_piston_1, param_study.v_piston_60bar_water_1, param_study.v_fluid_60bar_water_1, param_study.x_p_60bar_water_1, param_study.x_fluid_60bar_water_1)
E_ref_60bar_M1 = E_total_60bar_M1[0]
t_75bar_M1, E_total_75bar_M1 = en.compute_energy_balance(param_study.sol_75bar_water_1.t, param_study.p_t_75bar_water_1, param_study.V_t_75bar_water_1, param_study.p_p_75bar_water_1, param_study.V_p_75bar_water_1, param.m_piston_1, param_study.v_piston_75bar_water_1, param_study.v_fluid_75bar_water_1, param_study.x_p_75bar_water_1, param_study.x_fluid_75bar_water_1)
E_ref_75bar_M1 = E_total_75bar_M1[0]
t_100bar_M1, E_total_100bar_M1 = en.compute_energy_balance(param_study.sol_100bar_water_1.t, param_study.p_t_100bar_water_1, param_study.V_t_100bar_water_1, param_study.p_p_100bar_water_1, param_study.V_p_100bar_water_1, param.m_piston_1, param_study.v_piston_100bar_water_1, param_study.v_fluid_100bar_water_1, param_study.x_p_100bar_water_1, param_study.x_fluid_100bar_water_1)
E_ref_100bar_M1 = E_total_100bar_M1[0]

#total energy, M=1.5#
t_10bar_M15, E_total_10bar_M15 = en.compute_energy_balance(param_study.sol_10bar_water_15.t, param_study.p_t_10bar_water_15, param_study.V_t_10bar_water_15, param_study.p_p_10bar_water_15, param_study.V_p_10bar_water_15, param.m_piston_15, param_study.v_piston_10bar_water_15, param_study.v_fluid_10bar_water_15, param_study.x_p_10bar_water_15, param_study.x_fluid_10bar_water_15)
E_ref_10bar_M15 = E_total_10bar_M15[0]
t_20bar_M15, E_total_20bar_M15 = en.compute_energy_balance(param_study.sol_20bar_water_15.t, param_study.p_t_20bar_water_15, param_study.V_t_20bar_water_15, param_study.p_p_20bar_water_15, param_study.V_p_20bar_water_15, param.m_piston_15, param_study.v_piston_20bar_water_15, param_study.v_fluid_20bar_water_15, param_study.x_p_20bar_water_15, param_study.x_fluid_20bar_water_15)
E_ref_20bar_M15 = E_total_20bar_M15[0]
t_30bar_M15, E_total_30bar_M15 = en.compute_energy_balance(param_study.sol_30bar_water_15.t, param_study.p_t_30bar_water_15, param_study.V_t_30bar_water_15, param_study.p_p_30bar_water_15, param_study.V_p_30bar_water_15, param.m_piston_15, param_study.v_piston_30bar_water_15, param_study.v_fluid_30bar_water_15, param_study.x_p_30bar_water_15, param_study.x_fluid_30bar_water_15)
E_ref_30bar_M15 = E_total_30bar_M15[0]
t_40bar_M15, E_total_40bar_M15 = en.compute_energy_balance(param_study.sol_40bar_water_15.t, param_study.p_t_40bar_water_15, param_study.V_t_40bar_water_15, param_study.p_p_40bar_water_15, param_study.V_p_40bar_water_15, param.m_piston_15, param_study.v_piston_40bar_water_15, param_study.v_fluid_40bar_water_15, param_study.x_p_40bar_water_15, param_study.x_fluid_40bar_water_15)
E_ref_40bar_M15 = E_total_40bar_M15[0]
t_50bar_M15, E_total_50bar_M15 = en.compute_energy_balance(param_study.sol_50bar_water_15.t, param_study.p_t_50bar_water_15, param_study.V_t_50bar_water_15, param_study.p_p_50bar_water_15, param_study.V_p_50bar_water_15, param.m_piston_15, param_study.v_piston_50bar_water_15, param_study.v_fluid_50bar_water_15, param_study.x_p_50bar_water_15, param_study.x_fluid_50bar_water_15)
E_ref_50bar_M15 = E_total_50bar_M15[0]
t_60bar_M15, E_total_60bar_M15 = en.compute_energy_balance(param_study.sol_60bar_water_15.t, param_study.p_t_60bar_water_15, param_study.V_t_60bar_water_15, param_study.p_p_60bar_water_15, param_study.V_p_60bar_water_15, param.m_piston_15, param_study.v_piston_60bar_water_15, param_study.v_fluid_60bar_water_15, param_study.x_p_60bar_water_15, param_study.x_fluid_60bar_water_15)
E_ref_60bar_M15 = E_total_60bar_M15[0]
t_75bar_M15, E_total_75bar_M15 = en.compute_energy_balance(param_study.sol_75bar_water_15.t, param_study.p_t_75bar_water_15, param_study.V_t_75bar_water_15, param_study.p_p_75bar_water_15, param_study.V_p_75bar_water_15, param.m_piston_15, param_study.v_piston_75bar_water_15, param_study.v_fluid_75bar_water_15, param_study.x_p_75bar_water_15, param_study.x_fluid_75bar_water_15)
E_ref_75bar_M15 = E_total_75bar_M15[0]
t_100bar_M15, E_total_100bar_M15 = en.compute_energy_balance(param_study.sol_100bar_water_15.t, param_study.p_t_100bar_water_15, param_study.V_t_100bar_water_15, param_study.p_p_100bar_water_15, param_study.V_p_100bar_water_15, param.m_piston_15, param_study.v_piston_100bar_water_15, param_study.v_fluid_100bar_water_15, param_study.x_p_100bar_water_15, param_study.x_fluid_100bar_water_15)
E_ref_100bar_M15 = E_total_100bar_M15[0]

#total energy, M=2#
t_10bar_M2, E_total_10bar_M2 = en.compute_energy_balance(param_study.sol_10bar_water_2.t, param_study.p_t_10bar_water_2, param_study.V_t_10bar_water_2, param_study.p_p_10bar_water_2, param_study.V_p_10bar_water_2, param.m_piston_2, param_study.v_piston_10bar_water_2, param_study.v_fluid_10bar_water_2, param_study.x_p_10bar_water_2, param_study.x_fluid_10bar_water_2)
E_ref_10bar_M2 = E_total_10bar_M2[0]
t_20bar_M2, E_total_20bar_M2 = en.compute_energy_balance(param_study.sol_20bar_water_2.t,param_study. p_t_20bar_water_2, param_study.V_t_20bar_water_2, param_study.p_p_20bar_water_2, param_study.V_p_20bar_water_2, param.m_piston_2, param_study.v_piston_20bar_water_2, param_study.v_fluid_20bar_water_2, param_study.x_p_20bar_water_2, param_study.x_fluid_20bar_water_2)
E_ref_20bar_M2 = E_total_20bar_M2[0]
t_30bar_M2, E_total_30bar_M2 = en.compute_energy_balance(param_study.sol_30bar_water_2.t, param_study.p_t_30bar_water_2, param_study.V_t_30bar_water_2, param_study.p_p_30bar_water_2, param_study.V_p_30bar_water_2, param.m_piston_2, param_study.v_piston_30bar_water_2, param_study.v_fluid_30bar_water_2, param_study.x_p_30bar_water_2, param_study.x_fluid_30bar_water_2)
E_ref_30bar_M2 = E_total_30bar_M2[0]
t_40bar_M2, E_total_40bar_M2 = en.compute_energy_balance(param_study.sol_40bar_water_2.t, param_study.p_t_40bar_water_2, param_study.V_t_40bar_water_2, param_study.p_p_40bar_water_2, param_study.V_p_40bar_water_2, param.m_piston_2, param_study.v_piston_40bar_water_2, param_study.v_fluid_40bar_water_2, param_study.x_p_40bar_water_2, param_study.x_fluid_40bar_water_2)
E_ref_40bar_M2 = E_total_40bar_M2[0]
t_50bar_M2, E_total_50bar_M2 = en.compute_energy_balance(param_study.sol_50bar_water_2.t, param_study.p_t_50bar_water_2, param_study.V_t_50bar_water_2, param_study.p_p_50bar_water_2, param_study.V_p_50bar_water_2, param.m_piston_2, param_study.v_piston_50bar_water_2, param_study.v_fluid_50bar_water_2, param_study.x_p_50bar_water_2, param_study.x_fluid_50bar_water_2)
E_ref_50bar_M2 = E_total_50bar_M2[0]
t_60bar_M2, E_total_60bar_M2 = en.compute_energy_balance(param_study.sol_60bar_water_2.t, param_study.p_t_60bar_water_2, param_study.V_t_60bar_water_2, param_study.p_p_60bar_water_2, param_study.V_p_60bar_water_2, param.m_piston_2, param_study.v_piston_60bar_water_2, param_study.v_fluid_60bar_water_2, param_study.x_p_60bar_water_2, param_study.x_fluid_60bar_water_2)
E_ref_60bar_M2 = E_total_60bar_M2[0]
t_75bar_M2, E_total_75bar_M2 = en.compute_energy_balance(param_study.sol_75bar_water_2.t, param_study.p_t_75bar_water_2, param_study.V_t_75bar_water_2, param_study.p_p_75bar_water_2, param_study.V_p_75bar_water_2, param.m_piston_2, param_study.v_piston_75bar_water_2, param_study.v_fluid_75bar_water_2, param_study.x_p_75bar_water_2, param_study.x_fluid_75bar_water_2)
E_ref_75bar_M2 = E_total_75bar_M2[0]
t_100bar_M2, E_total_100bar_M2 = en.compute_energy_balance(param_study.sol_100bar_water_2.t, param_study.p_t_100bar_water_2, param_study.V_t_100bar_water_2, param_study.p_p_100bar_water_2, param_study.V_p_100bar_water_2, param.m_piston_2, param_study.v_piston_100bar_water_2, param_study.v_fluid_100bar_water_2, param_study.x_p_100bar_water_2, param_study.x_fluid_100bar_water_2)
E_ref_100bar_M2 = E_total_100bar_M2[0]

#1 MPa, water, usual pda#
#control-volume energy#
e_ac_sig_water_10bar_01, e_in_10bar_01, e_piston_10bar_01, e_out_10bar_01= en.compute_e_ac_from_signals(param_study.sol_10bar_water_01.t, param_study.v_piston_10bar_water_01, param_study.v_fluid_10bar_water_01[n,:], param_study.p_p_10bar_water_01, param_study.p_t_10bar_water_01, param_study.v_fluid_10bar_water_01, param.rho_aluminium, param.th_piston_01, param_study.a0_values_10bar_water_01)
e_ac_sig_water_10bar_05, e_in_10bar_05, e_piston_10bar_05, e_out_10bar_05= en.compute_e_ac_from_signals(param_study.sol_10bar_water_05.t, param_study.v_piston_10bar_water_05, param_study.v_fluid_10bar_water_05[n,:], param_study.p_p_10bar_water_05, param_study.p_t_10bar_water_05, param_study.v_fluid_10bar_water_05, param.rho_aluminium, param.th_piston_05, param_study.a0_values_10bar_water_05)
e_ac_sig_water_10bar_09, e_in_10bar_09, e_piston_10bar_09, e_out_10bar_09= en.compute_e_ac_from_signals(param_study.sol_10bar_water_09.t, param_study.v_piston_10bar_water_09, param_study.v_fluid_10bar_water_09[n,:], param_study.p_p_10bar_water_09, param_study.p_t_10bar_water_09, param_study.v_fluid_10bar_water_09, param.rho_aluminium, param.th_piston_09, param_study.a0_values_10bar_water_09)
e_ac_sig_water_10bar_1, e_in_10bar_1, e_piston_10bar_1, e_out_10bar_1= en.compute_e_ac_from_signals(param_study.sol_10bar_water_1.t, param_study.v_piston_10bar_water_1, param_study.v_fluid_10bar_water_1[n,:], param_study.p_p_10bar_water_1, param_study.p_t_10bar_water_1, param_study.v_fluid_10bar_water_1, param.rho_aluminium, param.th_piston_1, param_study.a0_values_10bar_water_1)
e_ac_sig_water_10bar_15, e_in_10bar_15, e_piston_10bar_15, e_out_10bar_15= en.compute_e_ac_from_signals(param_study.sol_10bar_water_15.t, param_study.v_piston_10bar_water_15, param_study.v_fluid_10bar_water_15[n,:], param_study.p_p_10bar_water_15, param_study.p_t_10bar_water_15, param_study.v_fluid_10bar_water_15, param.rho_aluminium, param.th_piston_15, param_study.a0_values_10bar_water_15)
e_ac_sig_water_10bar_2, e_in_10bar_2, e_piston_10bar_2, e_out_10bar_2= en.compute_e_ac_from_signals(param_study.sol_10bar_water_2.t, param_study.v_piston_10bar_water_2, param_study.v_fluid_10bar_water_2[n,:], param_study.p_p_10bar_water_2, param_study.p_t_10bar_water_2, param_study.v_fluid_10bar_water_2, param.rho_aluminium, param.th_piston_2, param_study.a0_values_10bar_water_2)
e_ac_sig_water_10bar_no_piston, e_in_10bar_no_piston, e_out_10bar_no_piston = en.compute_e_ac_from_signals_no_piston(param_study.sol_10bar_water_no_piston.t, param_study.v_fluid_10bar_water_no_piston[0,:], param_study.v_fluid_10bar_water_no_piston[n,:], param_study.p_p_10bar_water_no_piston, param_study.p_t_10bar_water_no_piston, param_study.v_fluid_10bar_water_no_piston)

#mean control-volume energy between the two first compression peaks#
e_ac_sig_water_10bar_01_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_10bar_water_01.t, e_ac_sig_water_10bar_01, t_first_peak_10bar_water_01, t_second_peak_10bar_water_01)
e_in_water_10bar_01_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_10bar_water_01.t, e_in_10bar_01, t_first_peak_10bar_water_01, t_second_peak_10bar_water_01)
e_out_water_10bar_01_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_10bar_water_01.t, e_out_10bar_01, t_first_peak_10bar_water_01, t_second_peak_10bar_water_01)
e_piston_water_10bar_01_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_10bar_water_01.t, e_piston_10bar_01, t_first_peak_10bar_water_01, t_second_peak_10bar_water_01)

e_ac_sig_water_10bar_05_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_10bar_water_05.t, e_ac_sig_water_10bar_05, t_first_peak_10bar_water_05, t_second_peak_10bar_water_05)
e_in_water_10bar_05_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_10bar_water_05.t, e_in_10bar_05, t_first_peak_10bar_water_05, t_second_peak_10bar_water_05)
e_out_water_10bar_05_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_10bar_water_05.t, e_out_10bar_05, t_first_peak_10bar_water_05, t_second_peak_10bar_water_05)
e_piston_water_10bar_05_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_10bar_water_05.t, e_piston_10bar_05, t_first_peak_10bar_water_05, t_second_peak_10bar_water_05)

e_ac_sig_water_10bar_09_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_10bar_water_09.t, e_ac_sig_water_10bar_09, t_first_peak_10bar_water_1, t_second_peak_10bar_water_1)
e_in_water_10bar_09_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_10bar_water_09.t, e_in_10bar_09, t_first_peak_10bar_water_1, t_second_peak_10bar_water_1)
e_out_water_10bar_09_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_10bar_water_09.t, e_out_10bar_09, t_first_peak_10bar_water_1, t_second_peak_10bar_water_1)
e_piston_water_10bar_09_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_10bar_water_09.t, e_piston_10bar_09, t_first_peak_10bar_water_1, t_second_peak_10bar_water_1)

e_ac_sig_water_10bar_1_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_10bar_water_1.t, e_ac_sig_water_10bar_1, t_first_peak_10bar_water_1, t_second_peak_10bar_water_1)
e_in_water_10bar_1_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_10bar_water_1.t, e_in_10bar_1, t_first_peak_10bar_water_1, t_second_peak_10bar_water_1)
e_out_water_10bar_1_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_10bar_water_1.t, e_out_10bar_1, t_first_peak_10bar_water_1, t_second_peak_10bar_water_1)
e_piston_water_10bar_1_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_10bar_water_1.t, e_piston_10bar_1, t_first_peak_10bar_water_1, t_second_peak_10bar_water_1)

e_ac_sig_water_10bar_15_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_10bar_water_15.t, e_ac_sig_water_10bar_15, t_first_peak_10bar_water_15, t_second_peak_10bar_water_15)
e_in_water_10bar_15_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_10bar_water_15.t, e_in_10bar_15, t_first_peak_10bar_water_15, t_second_peak_10bar_water_15)
e_out_water_10bar_15_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_10bar_water_15.t, e_out_10bar_15, t_first_peak_10bar_water_15, t_second_peak_10bar_water_15)
e_piston_water_10bar_15_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_10bar_water_15.t, e_piston_10bar_15, t_first_peak_10bar_water_15, t_second_peak_10bar_water_15)

e_ac_sig_water_10bar_2_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_10bar_water_2.t, e_ac_sig_water_10bar_2, t_first_peak_10bar_water_2, t_second_peak_10bar_water_2)
e_in_water_10bar_2_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_10bar_water_2.t, e_in_10bar_2, t_first_peak_10bar_water_2, t_second_peak_10bar_water_2)
e_out_water_10bar_2_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_10bar_water_2.t, e_out_10bar_2, t_first_peak_10bar_water_2, t_second_peak_10bar_water_2)
e_piston_water_10bar_2_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_10bar_water_2.t, e_piston_10bar_2, t_first_peak_10bar_water_2, t_second_peak_10bar_water_2)

e_ac_sig_water_10bar_no_piston_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_10bar_water_no_piston.t, e_ac_sig_water_10bar_no_piston, t_first_peak_10bar_water_no_piston, t_second_peak_10bar_water_no_piston)
e_in_water_10bar_no_piston_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_10bar_water_no_piston.t, e_in_10bar_no_piston, t_first_peak_10bar_water_no_piston, t_second_peak_10bar_water_no_piston)
e_out_water_10bar_no_piston_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_10bar_water_no_piston.t, e_out_10bar_no_piston, t_first_peak_10bar_water_no_piston, t_second_peak_10bar_water_no_piston)

#<E_cv^*>
E_ac_sig_water_over_E_ref_10bar_from_peak_to_peak_different_M = np.array([
    A_p*e_ac_sig_water_10bar_no_piston_from_peak_to_peak/E_ref_10bar_no_piston,
    e_ac_sig_water_10bar_01_from_peak_to_peak*A_p/E_ref_10bar_M01,
    e_ac_sig_water_10bar_05_from_peak_to_peak*A_p/E_ref_10bar_M05,
    e_ac_sig_water_10bar_1_from_peak_to_peak*A_p/E_ref_10bar_M1,
    e_ac_sig_water_10bar_15_from_peak_to_peak*A_p/E_ref_10bar_M15,
    e_ac_sig_water_10bar_2_from_peak_to_peak*A_p/E_ref_10bar_M2
])

#<E_in^*>
E_in_water_over_E_ref_10bar_from_peak_to_peak_different_M = np.array([
    e_in_water_10bar_no_piston_from_peak_to_peak*A_p/E_ref_10bar_no_piston,
    e_in_water_10bar_01_from_peak_to_peak*A_p/E_ref_10bar_M01,
    e_in_water_10bar_05_from_peak_to_peak*A_p/E_ref_10bar_M05,
    e_in_water_10bar_1_from_peak_to_peak*A_p/E_ref_10bar_M1,
    e_in_water_10bar_15_from_peak_to_peak*A_p/E_ref_10bar_M15,
    e_in_water_10bar_2_from_peak_to_peak*A_p/E_ref_10bar_M2
])

#<E_out^*>
E_out_water_over_E_ref_10bar_from_peak_to_peak_different_M = np.array([
    e_out_water_10bar_no_piston_from_peak_to_peak*A_p/E_ref_10bar_no_piston,
    e_out_water_10bar_01_from_peak_to_peak*A_p/E_ref_10bar_M01,
    e_out_water_10bar_05_from_peak_to_peak*A_p/E_ref_10bar_M05,
    e_out_water_10bar_1_from_peak_to_peak*A_p/E_ref_10bar_M1,
    e_out_water_10bar_15_from_peak_to_peak*A_p/E_ref_10bar_M15,
    e_out_water_10bar_2_from_peak_to_peak*A_p/E_ref_10bar_M2
])

#<E_piston^*>
E_piston_water_over_E_ref_10bar_from_peak_to_peak_different_M = np.array([
    0,
    e_piston_water_10bar_01_from_peak_to_peak*A_p/E_ref_10bar_M01,
    e_piston_water_10bar_05_from_peak_to_peak*A_p/E_ref_10bar_M05,
    e_piston_water_10bar_1_from_peak_to_peak*A_p/E_ref_10bar_M1,
    e_piston_water_10bar_15_from_peak_to_peak*A_p/E_ref_10bar_M15,
    e_piston_water_10bar_2_from_peak_to_peak*A_p/E_ref_10bar_M2
])

#2 MPa, water, usual pda#

e_ac_sig_water_20bar_01, e_in_20bar_01, e_piston_20bar_01, e_out_20bar_01= en.compute_e_ac_from_signals(param_study.sol_20bar_water_01.t, param_study.v_piston_20bar_water_01, param_study.v_fluid_20bar_water_01[n,:], param_study.p_p_20bar_water_01, param_study.p_t_20bar_water_01, param_study.v_fluid_20bar_water_01, param.rho_aluminium, param.th_piston_01, param_study.a0_values_20bar_water_01)
e_ac_sig_water_20bar_05, e_in_20bar_05, e_piston_20bar_05, e_out_20bar_05= en.compute_e_ac_from_signals(param_study.sol_20bar_water_05.t, param_study.v_piston_20bar_water_05, param_study.v_fluid_20bar_water_05[n,:], param_study.p_p_20bar_water_05, param_study.p_t_20bar_water_05, param_study.v_fluid_20bar_water_05, param.rho_aluminium, param.th_piston_05, param_study.a0_values_20bar_water_05)
e_ac_sig_water_20bar_1, e_in_20bar_1, e_piston_20bar_1, e_out_20bar_1= en.compute_e_ac_from_signals(param_study.sol_20bar_water_1.t, param_study.v_piston_20bar_water_1, param_study.v_fluid_20bar_water_1[n,:], param_study.p_p_20bar_water_1, param_study.p_t_20bar_water_1, param_study.v_fluid_20bar_water_1, param.rho_aluminium, param.th_piston_1, param_study.a0_values_20bar_water_1)
e_ac_sig_water_20bar_15, e_in_20bar_15, e_piston_20bar_15, e_out_20bar_15= en.compute_e_ac_from_signals(param_study.sol_20bar_water_15.t, param_study.v_piston_20bar_water_15, param_study.v_fluid_20bar_water_15[n,:], param_study.p_p_20bar_water_15, param_study.p_t_20bar_water_15, param_study.v_fluid_20bar_water_15, param.rho_aluminium, param.th_piston_15, param_study.a0_values_20bar_water_15)
e_ac_sig_water_20bar_2, e_in_20bar_2, e_piston_20bar_2, e_out_20bar_2= en.compute_e_ac_from_signals(param_study.sol_20bar_water_2.t, param_study.v_piston_20bar_water_2, param_study.v_fluid_20bar_water_2[n,:], param_study.p_p_20bar_water_2, param_study.p_t_20bar_water_2, param_study.v_fluid_20bar_water_2, param.rho_aluminium, param.th_piston_2, param_study.a0_values_20bar_water_2)
e_ac_sig_water_20bar_no_piston, e_in_20bar_no_piston, e_out_20bar_no_piston = en.compute_e_ac_from_signals_no_piston(param_study.sol_20bar_water_no_piston.t, param_study.v_fluid_20bar_water_no_piston[0,:], param_study.v_fluid_20bar_water_no_piston[n,:], param_study.p_p_20bar_water_no_piston, param_study.p_t_20bar_water_no_piston, param_study.v_fluid_20bar_water_no_piston)

e_ac_sig_water_20bar_01_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_20bar_water_01.t, e_ac_sig_water_20bar_01, t_first_peak_20bar_water_01, t_second_peak_20bar_water_01)
e_in_water_20bar_01_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_20bar_water_01.t, e_in_20bar_01, t_first_peak_20bar_water_01, t_second_peak_20bar_water_01)
e_out_water_20bar_01_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_20bar_water_01.t, e_out_20bar_01, t_first_peak_20bar_water_01, t_second_peak_20bar_water_01)
e_piston_water_20bar_01_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_20bar_water_01.t, e_piston_20bar_01, t_first_peak_20bar_water_01, t_second_peak_20bar_water_01)

e_ac_sig_water_20bar_05_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_20bar_water_05.t, e_ac_sig_water_20bar_05, t_first_peak_20bar_water_05, t_second_peak_20bar_water_05)
e_in_water_20bar_05_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_20bar_water_05.t, e_in_20bar_05, t_first_peak_20bar_water_05, t_second_peak_20bar_water_05)
e_out_water_20bar_05_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_20bar_water_05.t, e_out_20bar_05, t_first_peak_20bar_water_05, t_second_peak_20bar_water_05)
e_piston_water_20bar_05_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_20bar_water_05.t, e_piston_20bar_05, t_first_peak_20bar_water_05, t_second_peak_20bar_water_05)

e_ac_sig_water_20bar_1_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_20bar_water_1.t, e_ac_sig_water_20bar_1, t_first_peak_20bar_water_1, t_second_peak_20bar_water_1)
e_in_water_20bar_1_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_20bar_water_1.t, e_in_20bar_1, t_first_peak_20bar_water_1, t_second_peak_20bar_water_1)
e_out_water_20bar_1_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_20bar_water_1.t, e_out_20bar_1, t_first_peak_20bar_water_1, t_second_peak_20bar_water_1)
e_piston_water_20bar_1_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_20bar_water_1.t, e_piston_20bar_1, t_first_peak_20bar_water_1, t_second_peak_20bar_water_1)

e_ac_sig_water_20bar_15_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_20bar_water_15.t, e_ac_sig_water_20bar_15, t_first_peak_20bar_water_15, t_second_peak_20bar_water_15)
e_in_water_20bar_15_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_20bar_water_15.t, e_in_20bar_15, t_first_peak_20bar_water_15, t_second_peak_20bar_water_15)
e_out_water_20bar_15_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_20bar_water_15.t, e_out_20bar_15, t_first_peak_20bar_water_15, t_second_peak_20bar_water_15)
e_piston_water_20bar_15_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_20bar_water_15.t, e_piston_20bar_15, t_first_peak_20bar_water_15, t_second_peak_20bar_water_15)

e_ac_sig_water_20bar_2_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_20bar_water_2.t, e_ac_sig_water_20bar_2, t_first_peak_20bar_water_2, t_second_peak_20bar_water_2)
e_in_water_20bar_2_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_20bar_water_2.t, e_in_20bar_2, t_first_peak_20bar_water_2, t_second_peak_20bar_water_2)
e_out_water_20bar_2_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_20bar_water_2.t, e_out_20bar_2, t_first_peak_20bar_water_2, t_second_peak_20bar_water_2)
e_piston_water_20bar_2_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_20bar_water_2.t, e_piston_20bar_2, t_first_peak_20bar_water_2, t_second_peak_20bar_water_2)

e_ac_sig_water_20bar_no_piston_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_20bar_water_no_piston.t, e_ac_sig_water_20bar_no_piston, t_first_peak_20bar_water_no_piston, t_second_peak_20bar_water_no_piston)
e_in_water_20bar_no_piston_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_20bar_water_no_piston.t, e_in_20bar_no_piston, t_first_peak_20bar_water_no_piston, t_second_peak_20bar_water_no_piston)
e_out_water_20bar_no_piston_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_20bar_water_no_piston.t, e_out_20bar_no_piston, t_first_peak_20bar_water_no_piston, t_second_peak_20bar_water_no_piston)

E_ac_sig_water_over_E_ref_20bar_from_peak_to_peak_different_M = np.array([
    A_p*e_ac_sig_water_20bar_no_piston_from_peak_to_peak/E_ref_20bar_no_piston,
    e_ac_sig_water_20bar_01_from_peak_to_peak*A_p/E_ref_20bar_M01,
    e_ac_sig_water_20bar_05_from_peak_to_peak*A_p/E_ref_20bar_M05,
    e_ac_sig_water_20bar_1_from_peak_to_peak*A_p/E_ref_20bar_M1,
    e_ac_sig_water_20bar_15_from_peak_to_peak*A_p/E_ref_20bar_M15,
    e_ac_sig_water_20bar_2_from_peak_to_peak*A_p/E_ref_20bar_M2
])

E_in_water_over_E_ref_20bar_from_peak_to_peak_different_M = np.array([
    e_in_water_20bar_no_piston_from_peak_to_peak*A_p/E_ref_20bar_no_piston,
    e_in_water_20bar_01_from_peak_to_peak*A_p/E_ref_20bar_M01,
    e_in_water_20bar_05_from_peak_to_peak*A_p/E_ref_20bar_M05,
    e_in_water_20bar_1_from_peak_to_peak*A_p/E_ref_20bar_M1,
    e_in_water_20bar_15_from_peak_to_peak*A_p/E_ref_20bar_M15,
    e_in_water_20bar_2_from_peak_to_peak*A_p/E_ref_20bar_M2
])

E_out_water_over_E_ref_20bar_from_peak_to_peak_different_M = np.array([
    e_out_water_20bar_no_piston_from_peak_to_peak*A_p/E_ref_20bar_no_piston,
    e_out_water_20bar_01_from_peak_to_peak*A_p/E_ref_20bar_M01,
    e_out_water_20bar_05_from_peak_to_peak*A_p/E_ref_20bar_M05,
    e_out_water_20bar_1_from_peak_to_peak*A_p/E_ref_20bar_M1,
    e_out_water_20bar_15_from_peak_to_peak*A_p/E_ref_20bar_M15,
    e_out_water_20bar_2_from_peak_to_peak*A_p/E_ref_20bar_M2
])

E_piston_water_over_E_ref_20bar_from_peak_to_peak_different_M = np.array([
    0,
    e_piston_water_20bar_01_from_peak_to_peak*A_p/E_ref_20bar_M01,
    e_piston_water_20bar_05_from_peak_to_peak*A_p/E_ref_20bar_M05,
    e_piston_water_20bar_1_from_peak_to_peak*A_p/E_ref_20bar_M1,
    e_piston_water_20bar_15_from_peak_to_peak*A_p/E_ref_20bar_M15,
    e_piston_water_20bar_2_from_peak_to_peak*A_p/E_ref_20bar_M2
])

#3 MPa, water, usual pda#

e_ac_sig_water_30bar_01, e_in_30bar_01, e_piston_30bar_01, e_out_30bar_01= en.compute_e_ac_from_signals(param_study.sol_30bar_water_01.t, param_study.v_piston_30bar_water_01, param_study.v_fluid_30bar_water_01[n,:], param_study.p_p_30bar_water_01, param_study.p_t_30bar_water_01, param_study.v_fluid_30bar_water_01, param.rho_aluminium, param.th_piston_01, param_study.a0_values_30bar_water_01)
e_ac_sig_water_30bar_05, e_in_30bar_05, e_piston_30bar_05, e_out_30bar_05= en.compute_e_ac_from_signals(param_study.sol_30bar_water_05.t, param_study.v_piston_30bar_water_05, param_study.v_fluid_30bar_water_05[n,:], param_study.p_p_30bar_water_05, param_study.p_t_30bar_water_05, param_study.v_fluid_30bar_water_05, param.rho_aluminium, param.th_piston_05, param_study.a0_values_30bar_water_05)
e_ac_sig_water_30bar_1, e_in_30bar_1, e_piston_30bar_1, e_out_30bar_1= en.compute_e_ac_from_signals(param_study.sol_30bar_water_1.t, param_study.v_piston_30bar_water_1, param_study.v_fluid_30bar_water_1[n,:], param_study.p_p_30bar_water_1, param_study.p_t_30bar_water_1, param_study.v_fluid_30bar_water_1, param.rho_aluminium, param.th_piston_1, param_study.a0_values_30bar_water_1)
e_ac_sig_water_30bar_15, e_in_30bar_15, e_piston_30bar_15, e_out_30bar_15= en.compute_e_ac_from_signals(param_study.sol_30bar_water_15.t, param_study.v_piston_30bar_water_15, param_study.v_fluid_30bar_water_15[n,:], param_study.p_p_30bar_water_15, param_study.p_t_30bar_water_15, param_study.v_fluid_30bar_water_15, param.rho_aluminium, param.th_piston_15, param_study.a0_values_30bar_water_15)
e_ac_sig_water_30bar_2, e_in_30bar_2, e_piston_30bar_2, e_out_30bar_2= en.compute_e_ac_from_signals(param_study.sol_30bar_water_2.t, param_study.v_piston_30bar_water_2, param_study.v_fluid_30bar_water_2[n,:], param_study.p_p_30bar_water_2, param_study.p_t_30bar_water_2, param_study.v_fluid_30bar_water_2, param.rho_aluminium, param.th_piston_2, param_study.a0_values_30bar_water_2)
e_ac_sig_water_30bar_no_piston, e_in_30bar_no_piston, e_out_30bar_no_piston = en.compute_e_ac_from_signals_no_piston(param_study.sol_30bar_water_no_piston.t, param_study.v_fluid_30bar_water_no_piston[0,:], param_study.v_fluid_30bar_water_no_piston[n,:], param_study.p_p_30bar_water_no_piston, param_study.p_t_30bar_water_no_piston, param_study.v_fluid_30bar_water_no_piston)

e_ac_sig_water_30bar_01_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_30bar_water_01.t, e_ac_sig_water_30bar_01, t_first_peak_30bar_water_01, t_second_peak_30bar_water_01)
e_in_water_30bar_01_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_30bar_water_01.t, e_in_30bar_01, t_first_peak_30bar_water_01, t_second_peak_30bar_water_01)
e_out_water_30bar_01_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_30bar_water_01.t, e_out_30bar_01, t_first_peak_30bar_water_01, t_second_peak_30bar_water_01)
e_piston_water_30bar_01_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_30bar_water_01.t, e_piston_30bar_01, t_first_peak_30bar_water_01, t_second_peak_30bar_water_01)

e_ac_sig_water_30bar_05_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_30bar_water_05.t, e_ac_sig_water_30bar_05, t_first_peak_30bar_water_05, t_second_peak_30bar_water_05)
e_in_water_30bar_05_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_30bar_water_05.t, e_in_30bar_05, t_first_peak_30bar_water_05, t_second_peak_30bar_water_05)
e_out_water_30bar_05_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_30bar_water_05.t, e_out_30bar_05, t_first_peak_30bar_water_05, t_second_peak_30bar_water_05)
e_piston_water_30bar_05_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_30bar_water_05.t, e_piston_30bar_05, t_first_peak_30bar_water_05, t_second_peak_30bar_water_05)

e_ac_sig_water_30bar_1_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_30bar_water_1.t, e_ac_sig_water_30bar_1, t_first_peak_30bar_water_1, t_second_peak_30bar_water_1)
e_in_water_30bar_1_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_30bar_water_1.t, e_in_30bar_1, t_first_peak_30bar_water_1, t_second_peak_30bar_water_1)
e_out_water_30bar_1_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_30bar_water_1.t, e_out_30bar_1, t_first_peak_30bar_water_1, t_second_peak_30bar_water_1)
e_piston_water_30bar_1_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_30bar_water_1.t, e_piston_30bar_1, t_first_peak_30bar_water_1, t_second_peak_30bar_water_1)

e_ac_sig_water_30bar_15_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_30bar_water_15.t, e_ac_sig_water_30bar_15, t_first_peak_30bar_water_15, t_second_peak_30bar_water_15)
e_in_water_30bar_15_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_30bar_water_15.t, e_in_30bar_15, t_first_peak_30bar_water_15, t_second_peak_30bar_water_15)
e_out_water_30bar_15_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_30bar_water_15.t, e_out_30bar_15, t_first_peak_30bar_water_15, t_second_peak_30bar_water_15)
e_piston_water_30bar_15_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_30bar_water_15.t, e_piston_30bar_15, t_first_peak_30bar_water_15, t_second_peak_30bar_water_15)

e_ac_sig_water_30bar_2_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_30bar_water_2.t, e_ac_sig_water_30bar_2, t_first_peak_30bar_water_2, t_second_peak_30bar_water_2)
e_in_water_30bar_2_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_30bar_water_2.t, e_in_30bar_2, t_first_peak_30bar_water_2, t_second_peak_30bar_water_2)
e_out_water_30bar_2_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_30bar_water_2.t, e_out_30bar_2, t_first_peak_30bar_water_2, t_second_peak_30bar_water_2)
e_piston_water_30bar_2_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_30bar_water_2.t, e_piston_30bar_2, t_first_peak_30bar_water_2, t_second_peak_30bar_water_2)

e_ac_sig_water_30bar_no_piston_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_30bar_water_no_piston.t, e_ac_sig_water_30bar_no_piston, t_first_peak_30bar_water_no_piston, t_second_peak_30bar_water_no_piston)
e_in_water_30bar_no_piston_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_30bar_water_no_piston.t, e_in_30bar_no_piston, t_first_peak_30bar_water_no_piston, t_second_peak_30bar_water_no_piston)
e_out_water_30bar_no_piston_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_30bar_water_no_piston.t, e_out_30bar_no_piston, t_first_peak_30bar_water_no_piston, t_second_peak_30bar_water_no_piston)

E_ac_sig_water_over_E_ref_30bar_from_peak_to_peak_different_M = np.array([
    A_p*e_ac_sig_water_30bar_no_piston_from_peak_to_peak/E_ref_30bar_no_piston,
    e_ac_sig_water_30bar_01_from_peak_to_peak*A_p/E_ref_30bar_M01,
    e_ac_sig_water_30bar_05_from_peak_to_peak*A_p/E_ref_30bar_M05,
    e_ac_sig_water_30bar_1_from_peak_to_peak*A_p/E_ref_30bar_M1,
    e_ac_sig_water_30bar_15_from_peak_to_peak*A_p/E_ref_30bar_M15,
    e_ac_sig_water_30bar_2_from_peak_to_peak*A_p/E_ref_30bar_M2
])

E_in_water_over_E_ref_30bar_from_peak_to_peak_different_M = np.array([
    e_in_water_30bar_no_piston_from_peak_to_peak*A_p/E_ref_30bar_no_piston,
    e_in_water_30bar_01_from_peak_to_peak*A_p/E_ref_30bar_M01,
    e_in_water_30bar_05_from_peak_to_peak*A_p/E_ref_30bar_M05,
    e_in_water_30bar_1_from_peak_to_peak*A_p/E_ref_30bar_M1,
    e_in_water_30bar_15_from_peak_to_peak*A_p/E_ref_30bar_M15,
    e_in_water_30bar_2_from_peak_to_peak*A_p/E_ref_30bar_M2
])

E_out_water_over_E_ref_30bar_from_peak_to_peak_different_M = np.array([
    e_out_water_30bar_no_piston_from_peak_to_peak*A_p/E_ref_30bar_no_piston,
    e_out_water_30bar_01_from_peak_to_peak*A_p/E_ref_30bar_M01,
    e_out_water_30bar_05_from_peak_to_peak*A_p/E_ref_30bar_M05,
    e_out_water_30bar_1_from_peak_to_peak*A_p/E_ref_30bar_M1,
    e_out_water_30bar_15_from_peak_to_peak*A_p/E_ref_30bar_M15,
    e_out_water_30bar_2_from_peak_to_peak*A_p/E_ref_30bar_M2
])

E_piston_water_over_E_ref_30bar_from_peak_to_peak_different_M = np.array([
    0,
    e_piston_water_30bar_01_from_peak_to_peak*A_p/E_ref_30bar_M01,
    e_piston_water_30bar_05_from_peak_to_peak*A_p/E_ref_30bar_M05,
    e_piston_water_30bar_1_from_peak_to_peak*A_p/E_ref_30bar_M1,
    e_piston_water_30bar_15_from_peak_to_peak*A_p/E_ref_30bar_M15,
    e_piston_water_30bar_2_from_peak_to_peak*A_p/E_ref_30bar_M2
])



#4 MPa, water, usual pda#

e_ac_sig_water_40bar_01, e_in_40bar_01, e_piston_40bar_01, e_out_40bar_01= en.compute_e_ac_from_signals(param_study.sol_40bar_water_01.t, param_study.v_piston_40bar_water_01, param_study.v_fluid_40bar_water_01[n,:], param_study.p_p_40bar_water_01, param_study.p_t_40bar_water_01, param_study.v_fluid_40bar_water_01, param.rho_aluminium, param.th_piston_01, param_study.a0_values_40bar_water_01)
e_ac_sig_water_40bar_05, e_in_40bar_05, e_piston_40bar_05, e_out_40bar_05= en.compute_e_ac_from_signals(param_study.sol_40bar_water_05.t, param_study.v_piston_40bar_water_05, param_study.v_fluid_40bar_water_05[n,:], param_study.p_p_40bar_water_05, param_study.p_t_40bar_water_05, param_study.v_fluid_40bar_water_05, param.rho_aluminium, param.th_piston_05, param_study.a0_values_40bar_water_05)
e_ac_sig_water_40bar_1, e_in_40bar_1, e_piston_40bar_1, e_out_40bar_1= en.compute_e_ac_from_signals(param_study.sol_40bar_water_1.t, param_study.v_piston_40bar_water_1, param_study.v_fluid_40bar_water_1[n,:], param_study.p_p_40bar_water_1, param_study.p_t_40bar_water_1, param_study.v_fluid_40bar_water_1, param.rho_aluminium, param.th_piston_1, param_study.a0_values_40bar_water_1)
e_ac_sig_water_40bar_15, e_in_40bar_15, e_piston_40bar_15, e_out_40bar_15= en.compute_e_ac_from_signals(param_study.sol_40bar_water_15.t, param_study.v_piston_40bar_water_15, param_study.v_fluid_40bar_water_15[n,:], param_study.p_p_40bar_water_15, param_study.p_t_40bar_water_15, param_study.v_fluid_40bar_water_15, param.rho_aluminium, param.th_piston_15, param_study.a0_values_40bar_water_15)
e_ac_sig_water_40bar_2, e_in_40bar_2, e_piston_40bar_2, e_out_40bar_2= en.compute_e_ac_from_signals(param_study.sol_40bar_water_2.t, param_study.v_piston_40bar_water_2, param_study.v_fluid_40bar_water_2[n,:], param_study.p_p_40bar_water_2, param_study.p_t_40bar_water_2, param_study.v_fluid_40bar_water_2, param.rho_aluminium, param.th_piston_2, param_study.a0_values_40bar_water_2)
e_ac_sig_water_40bar_no_piston, e_in_40bar_no_piston, e_out_40bar_no_piston = en.compute_e_ac_from_signals_no_piston(param_study.sol_40bar_water_no_piston.t, param_study.v_fluid_40bar_water_no_piston[0,:], param_study.v_fluid_40bar_water_no_piston[n,:], param_study.p_p_40bar_water_no_piston, param_study.p_t_40bar_water_no_piston, param_study.v_fluid_40bar_water_no_piston)

e_ac_sig_water_40bar_01_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_40bar_water_01.t, e_ac_sig_water_40bar_01, t_first_peak_40bar_water_01, t_second_peak_40bar_water_01)
e_in_water_40bar_01_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_40bar_water_01.t, e_in_40bar_01, t_first_peak_40bar_water_01, t_second_peak_40bar_water_01)
e_out_water_40bar_01_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_40bar_water_01.t, e_out_40bar_01, t_first_peak_40bar_water_01, t_second_peak_40bar_water_01)
e_piston_water_40bar_01_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_40bar_water_01.t, e_piston_40bar_01, t_first_peak_40bar_water_01, t_second_peak_40bar_water_01)

e_ac_sig_water_40bar_05_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_40bar_water_05.t, e_ac_sig_water_40bar_05, t_first_peak_40bar_water_05, t_second_peak_40bar_water_05)
e_in_water_40bar_05_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_40bar_water_05.t, e_in_40bar_05, t_first_peak_40bar_water_05, t_second_peak_40bar_water_05)
e_out_water_40bar_05_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_40bar_water_05.t, e_out_40bar_05, t_first_peak_40bar_water_05, t_second_peak_40bar_water_05)
e_piston_water_40bar_05_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_40bar_water_05.t, e_piston_40bar_05, t_first_peak_40bar_water_05, t_second_peak_40bar_water_05)

e_ac_sig_water_40bar_1_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_40bar_water_1.t, e_ac_sig_water_40bar_1, t_first_peak_40bar_water_1, t_second_peak_40bar_water_1)
e_in_water_40bar_1_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_40bar_water_1.t, e_in_40bar_1, t_first_peak_40bar_water_1, t_second_peak_40bar_water_1)
e_out_water_40bar_1_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_40bar_water_1.t, e_out_40bar_1, t_first_peak_40bar_water_1, t_second_peak_40bar_water_1)
e_piston_water_40bar_1_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_40bar_water_1.t, e_piston_40bar_1, t_first_peak_40bar_water_1, t_second_peak_40bar_water_1)

e_ac_sig_water_40bar_15_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_40bar_water_15.t, e_ac_sig_water_40bar_15, t_first_peak_40bar_water_15, t_second_peak_40bar_water_15)
e_in_water_40bar_15_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_40bar_water_15.t, e_in_40bar_15, t_first_peak_40bar_water_15, t_second_peak_40bar_water_15)
e_out_water_40bar_15_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_40bar_water_15.t, e_out_40bar_15, t_first_peak_40bar_water_15, t_second_peak_40bar_water_15)
e_piston_water_40bar_15_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_40bar_water_15.t, e_piston_40bar_15, t_first_peak_40bar_water_15, t_second_peak_40bar_water_15)

e_ac_sig_water_40bar_2_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_40bar_water_2.t, e_ac_sig_water_40bar_2, t_first_peak_40bar_water_2, t_second_peak_40bar_water_2)
e_in_water_40bar_2_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_40bar_water_2.t, e_in_40bar_2, t_first_peak_40bar_water_2, t_second_peak_40bar_water_2)
e_out_water_40bar_2_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_40bar_water_2.t, e_out_40bar_2, t_first_peak_40bar_water_2, t_second_peak_40bar_water_2)
e_piston_water_40bar_2_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_40bar_water_2.t, e_piston_40bar_2, t_first_peak_40bar_water_2, t_second_peak_40bar_water_2)

e_ac_sig_water_40bar_no_piston_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_40bar_water_no_piston.t, e_ac_sig_water_40bar_no_piston, t_first_peak_40bar_water_no_piston, t_second_peak_40bar_water_no_piston)
e_in_water_40bar_no_piston_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_40bar_water_no_piston.t, e_in_40bar_no_piston, t_first_peak_40bar_water_no_piston, t_second_peak_40bar_water_no_piston)
e_out_water_40bar_no_piston_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_40bar_water_no_piston.t, e_out_40bar_no_piston, t_first_peak_40bar_water_no_piston, t_second_peak_40bar_water_no_piston)

E_ac_sig_water_over_E_ref_40bar_from_peak_to_peak_different_M = np.array([
    A_p*e_ac_sig_water_40bar_no_piston_from_peak_to_peak/E_ref_40bar_no_piston,
    e_ac_sig_water_40bar_01_from_peak_to_peak*A_p/E_ref_40bar_M01,
    e_ac_sig_water_40bar_05_from_peak_to_peak*A_p/E_ref_40bar_M05,
    e_ac_sig_water_40bar_1_from_peak_to_peak*A_p/E_ref_40bar_M1,
    e_ac_sig_water_40bar_15_from_peak_to_peak*A_p/E_ref_40bar_M15,
    e_ac_sig_water_40bar_2_from_peak_to_peak*A_p/E_ref_40bar_M2
])

E_in_water_over_E_ref_40bar_from_peak_to_peak_different_M = np.array([
    e_in_water_40bar_no_piston_from_peak_to_peak*A_p/E_ref_40bar_no_piston,
    e_in_water_40bar_01_from_peak_to_peak*A_p/E_ref_40bar_M01,
    e_in_water_40bar_05_from_peak_to_peak*A_p/E_ref_40bar_M05,
    e_in_water_40bar_1_from_peak_to_peak*A_p/E_ref_40bar_M1,
    e_in_water_40bar_15_from_peak_to_peak*A_p/E_ref_40bar_M15,
    e_in_water_40bar_2_from_peak_to_peak*A_p/E_ref_40bar_M2
])

E_out_water_over_E_ref_40bar_from_peak_to_peak_different_M = np.array([
    e_out_water_40bar_no_piston_from_peak_to_peak*A_p/E_ref_40bar_no_piston,
    e_out_water_40bar_01_from_peak_to_peak*A_p/E_ref_40bar_M01,
    e_out_water_40bar_05_from_peak_to_peak*A_p/E_ref_40bar_M05,
    e_out_water_40bar_1_from_peak_to_peak*A_p/E_ref_40bar_M1,
    e_out_water_40bar_15_from_peak_to_peak*A_p/E_ref_40bar_M15,
    e_out_water_40bar_2_from_peak_to_peak*A_p/E_ref_40bar_M2
])

E_piston_water_over_E_ref_40bar_from_peak_to_peak_different_M = np.array([
    0,
    e_piston_water_40bar_01_from_peak_to_peak*A_p/E_ref_40bar_M01,
    e_piston_water_40bar_05_from_peak_to_peak*A_p/E_ref_40bar_M05,
    e_piston_water_40bar_1_from_peak_to_peak*A_p/E_ref_40bar_M1,
    e_piston_water_40bar_15_from_peak_to_peak*A_p/E_ref_40bar_M15,
    e_piston_water_40bar_2_from_peak_to_peak*A_p/E_ref_40bar_M2
])


#5 MPa, water, usual pda#

e_ac_sig_water_50bar_01, e_in_50bar_01, e_piston_50bar_01, e_out_50bar_01= en.compute_e_ac_from_signals(param_study.sol_50bar_water_01.t, param_study.v_piston_50bar_water_01, param_study.v_fluid_50bar_water_01[n,:], param_study.p_p_50bar_water_01, param_study.p_t_50bar_water_01, param_study.v_fluid_50bar_water_01, param.rho_aluminium, param.th_piston_01, param_study.a0_values_50bar_water_01)
e_ac_sig_water_50bar_05, e_in_50bar_05, e_piston_50bar_05, e_out_50bar_05= en.compute_e_ac_from_signals(param_study.sol_50bar_water_05.t, param_study.v_piston_50bar_water_05, param_study.v_fluid_50bar_water_05[n,:], param_study.p_p_50bar_water_05, param_study.p_t_50bar_water_05, param_study.v_fluid_50bar_water_05, param.rho_aluminium, param.th_piston_05, param_study.a0_values_50bar_water_05)
e_ac_sig_water_50bar_1, e_in_50bar_1, e_piston_50bar_1, e_out_50bar_1= en.compute_e_ac_from_signals(param_study.sol_50bar_water_1.t, param_study.v_piston_50bar_water_1, param_study.v_fluid_50bar_water_1[n,:], param_study.p_p_50bar_water_1, param_study.p_t_50bar_water_1, param_study.v_fluid_50bar_water_1, param.rho_aluminium, param.th_piston_1, param_study.a0_values_50bar_water_1)
e_ac_sig_water_50bar_15, e_in_50bar_15, e_piston_50bar_15, e_out_50bar_15= en.compute_e_ac_from_signals(param_study.sol_50bar_water_15.t, param_study.v_piston_50bar_water_15, param_study.v_fluid_50bar_water_15[n,:], param_study.p_p_50bar_water_15, param_study.p_t_50bar_water_15, param_study.v_fluid_50bar_water_15, param.rho_aluminium, param.th_piston_15, param_study.a0_values_50bar_water_15)
e_ac_sig_water_50bar_2, e_in_50bar_2, e_piston_50bar_2, e_out_50bar_2= en.compute_e_ac_from_signals(param_study.sol_50bar_water_2.t, param_study.v_piston_50bar_water_2, param_study.v_fluid_50bar_water_2[n,:], param_study.p_p_50bar_water_2, param_study.p_t_50bar_water_2, param_study.v_fluid_50bar_water_2, param.rho_aluminium, param.th_piston_2, param_study.a0_values_50bar_water_2)
e_ac_sig_water_50bar_no_piston, e_in_50bar_no_piston, e_out_50bar_no_piston = en.compute_e_ac_from_signals_no_piston(param_study.sol_50bar_water_no_piston.t, param_study.v_fluid_50bar_water_no_piston[0,:], param_study.v_fluid_50bar_water_no_piston[n,:], param_study.p_p_50bar_water_no_piston, param_study.p_t_50bar_water_no_piston, param_study.v_fluid_50bar_water_no_piston)

e_ac_sig_water_50bar_01_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_50bar_water_01.t, e_ac_sig_water_50bar_01, t_first_peak_50bar_water_01, t_second_peak_50bar_water_01)
e_in_water_50bar_01_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_50bar_water_01.t, e_in_50bar_01, t_first_peak_50bar_water_01, t_second_peak_50bar_water_01)
e_out_water_50bar_01_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_50bar_water_01.t, e_out_50bar_01, t_first_peak_50bar_water_01, t_second_peak_50bar_water_01)
e_piston_water_50bar_01_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_50bar_water_01.t, e_piston_50bar_01, t_first_peak_50bar_water_01, t_second_peak_50bar_water_01)

e_ac_sig_water_50bar_05_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_50bar_water_05.t, e_ac_sig_water_50bar_05, t_first_peak_50bar_water_05, t_second_peak_50bar_water_05)
e_in_water_50bar_05_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_50bar_water_05.t, e_in_50bar_05, t_first_peak_50bar_water_05, t_second_peak_50bar_water_05)
e_out_water_50bar_05_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_50bar_water_05.t, e_out_50bar_05, t_first_peak_50bar_water_05, t_second_peak_50bar_water_05)
e_piston_water_50bar_05_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_50bar_water_05.t, e_piston_50bar_05, t_first_peak_50bar_water_05, t_second_peak_50bar_water_05)

e_ac_sig_water_50bar_1_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_50bar_water_1.t, e_ac_sig_water_50bar_1, t_first_peak_50bar_water_1, t_second_peak_50bar_water_1)
e_in_water_50bar_1_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_50bar_water_1.t, e_in_50bar_1, t_first_peak_50bar_water_1, t_second_peak_50bar_water_1)
e_out_water_50bar_1_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_50bar_water_1.t, e_out_50bar_1, t_first_peak_50bar_water_1, t_second_peak_50bar_water_1)
e_piston_water_50bar_1_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_50bar_water_1.t, e_piston_50bar_1, t_first_peak_50bar_water_1, t_second_peak_50bar_water_1)

e_ac_sig_water_50bar_15_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_50bar_water_15.t, e_ac_sig_water_50bar_15, t_first_peak_50bar_water_15, t_second_peak_50bar_water_15)
e_in_water_50bar_15_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_50bar_water_15.t, e_in_50bar_15, t_first_peak_50bar_water_15, t_second_peak_50bar_water_15)
e_out_water_50bar_15_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_50bar_water_15.t, e_out_50bar_15, t_first_peak_50bar_water_15, t_second_peak_50bar_water_15)
e_piston_water_50bar_15_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_50bar_water_15.t, e_piston_50bar_15, t_first_peak_50bar_water_15, t_second_peak_50bar_water_15)

e_ac_sig_water_50bar_2_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_50bar_water_2.t, e_ac_sig_water_50bar_2, t_first_peak_50bar_water_2, t_second_peak_50bar_water_2)
e_in_water_50bar_2_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_50bar_water_2.t, e_in_50bar_2, t_first_peak_50bar_water_2, t_second_peak_50bar_water_2)
e_out_water_50bar_2_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_50bar_water_2.t, e_out_50bar_2, t_first_peak_50bar_water_2, t_second_peak_50bar_water_2)
e_piston_water_50bar_2_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_50bar_water_2.t, e_piston_50bar_2, t_first_peak_50bar_water_2, t_second_peak_50bar_water_2)

e_ac_sig_water_50bar_no_piston_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_50bar_water_no_piston.t, e_ac_sig_water_50bar_no_piston, t_first_peak_50bar_water_no_piston, t_second_peak_50bar_water_no_piston)
e_in_water_50bar_no_piston_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_50bar_water_no_piston.t, e_in_50bar_no_piston, t_first_peak_50bar_water_no_piston, t_second_peak_50bar_water_no_piston)
e_out_water_50bar_no_piston_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_50bar_water_no_piston.t, e_out_50bar_no_piston, t_first_peak_50bar_water_no_piston, t_second_peak_50bar_water_no_piston)

E_ac_sig_water_over_E_ref_50bar_from_peak_to_peak_different_M = np.array([
    A_p*e_ac_sig_water_50bar_no_piston_from_peak_to_peak/E_ref_50bar_no_piston,
    e_ac_sig_water_50bar_01_from_peak_to_peak*A_p/E_ref_50bar_M01,
    e_ac_sig_water_50bar_05_from_peak_to_peak*A_p/E_ref_50bar_M05,
    e_ac_sig_water_50bar_1_from_peak_to_peak*A_p/E_ref_50bar_M1,
    e_ac_sig_water_50bar_15_from_peak_to_peak*A_p/E_ref_50bar_M15,
    e_ac_sig_water_50bar_2_from_peak_to_peak*A_p/E_ref_50bar_M2
])

E_in_water_over_E_ref_50bar_from_peak_to_peak_different_M = np.array([
    e_in_water_50bar_no_piston_from_peak_to_peak*A_p/E_ref_50bar_no_piston,
    e_in_water_50bar_01_from_peak_to_peak*A_p/E_ref_50bar_M01,
    e_in_water_50bar_05_from_peak_to_peak*A_p/E_ref_50bar_M05,
    e_in_water_50bar_1_from_peak_to_peak*A_p/E_ref_50bar_M1,
    e_in_water_50bar_15_from_peak_to_peak*A_p/E_ref_50bar_M15,
    e_in_water_50bar_2_from_peak_to_peak*A_p/E_ref_50bar_M2
])

E_out_water_over_E_ref_50bar_from_peak_to_peak_different_M = np.array([
    e_out_water_50bar_no_piston_from_peak_to_peak*A_p/E_ref_50bar_no_piston,
    e_out_water_50bar_01_from_peak_to_peak*A_p/E_ref_50bar_M01,
    e_out_water_50bar_05_from_peak_to_peak*A_p/E_ref_50bar_M05,
    e_out_water_50bar_1_from_peak_to_peak*A_p/E_ref_50bar_M1,
    e_out_water_50bar_15_from_peak_to_peak*A_p/E_ref_50bar_M15,
    e_out_water_50bar_2_from_peak_to_peak*A_p/E_ref_50bar_M2
])

E_piston_water_over_E_ref_50bar_from_peak_to_peak_different_M = np.array([
    0,
    e_piston_water_50bar_01_from_peak_to_peak*A_p/E_ref_50bar_M01,
    e_piston_water_50bar_05_from_peak_to_peak*A_p/E_ref_50bar_M05,
    e_piston_water_50bar_1_from_peak_to_peak*A_p/E_ref_50bar_M1,
    e_piston_water_50bar_15_from_peak_to_peak*A_p/E_ref_50bar_M15,
    e_piston_water_50bar_2_from_peak_to_peak*A_p/E_ref_50bar_M2
])

#6 MPa, water, usual pda#

e_ac_sig_water_60bar_01, e_in_60bar_01, e_piston_60bar_01, e_out_60bar_01= en.compute_e_ac_from_signals(param_study.sol_60bar_water_01.t, param_study.v_piston_60bar_water_01, param_study.v_fluid_60bar_water_01[n,:], param_study.p_p_60bar_water_01, param_study.p_t_60bar_water_01, param_study.v_fluid_60bar_water_01, param.rho_aluminium, param.th_piston_01, param_study.a0_values_60bar_water_01)
e_ac_sig_water_60bar_05, e_in_60bar_05, e_piston_60bar_05, e_out_60bar_05= en.compute_e_ac_from_signals(param_study.sol_60bar_water_05.t, param_study.v_piston_60bar_water_05, param_study.v_fluid_60bar_water_05[n,:], param_study.p_p_60bar_water_05, param_study.p_t_60bar_water_05, param_study.v_fluid_60bar_water_05, param.rho_aluminium, param.th_piston_05, param_study.a0_values_60bar_water_05)
e_ac_sig_water_60bar_1, e_in_60bar_1, e_piston_60bar_1, e_out_60bar_1= en.compute_e_ac_from_signals(param_study.sol_60bar_water_1.t, param_study.v_piston_60bar_water_1, param_study.v_fluid_60bar_water_1[n,:], param_study.p_p_60bar_water_1, param_study.p_t_60bar_water_1, param_study.v_fluid_60bar_water_1, param.rho_aluminium, param.th_piston_1, param_study.a0_values_60bar_water_1)
e_ac_sig_water_60bar_15, e_in_60bar_15, e_piston_60bar_15, e_out_60bar_15= en.compute_e_ac_from_signals(param_study.sol_60bar_water_15.t, param_study.v_piston_60bar_water_15, param_study.v_fluid_60bar_water_15[n,:], param_study.p_p_60bar_water_15, param_study.p_t_60bar_water_15, param_study.v_fluid_60bar_water_15, param.rho_aluminium, param.th_piston_15, param_study.a0_values_60bar_water_15)
e_ac_sig_water_60bar_2, e_in_60bar_2, e_piston_60bar_2, e_out_60bar_2= en.compute_e_ac_from_signals(param_study.sol_60bar_water_2.t, param_study.v_piston_60bar_water_2, param_study.v_fluid_60bar_water_2[n,:], param_study.p_p_60bar_water_2, param_study.p_t_60bar_water_2, param_study.v_fluid_60bar_water_2, param.rho_aluminium, param.th_piston_2, param_study.a0_values_60bar_water_2)
e_ac_sig_water_60bar_no_piston, e_in_60bar_no_piston, e_out_60bar_no_piston = en.compute_e_ac_from_signals_no_piston(param_study.sol_60bar_water_no_piston.t, param_study.v_fluid_60bar_water_no_piston[0,:], param_study.v_fluid_60bar_water_no_piston[n,:], param_study.p_p_60bar_water_no_piston, param_study.p_t_60bar_water_no_piston, param_study.v_fluid_60bar_water_no_piston)

e_ac_sig_water_60bar_01_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_60bar_water_01.t, e_ac_sig_water_60bar_01, t_first_peak_60bar_water_01, t_second_peak_60bar_water_01)
e_in_water_60bar_01_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_60bar_water_01.t, e_in_60bar_01, t_first_peak_60bar_water_01, t_second_peak_60bar_water_01)
e_out_water_60bar_01_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_60bar_water_01.t, e_out_60bar_01, t_first_peak_60bar_water_01, t_second_peak_60bar_water_01)
e_piston_water_60bar_01_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_60bar_water_01.t, e_piston_60bar_01, t_first_peak_60bar_water_01, t_second_peak_60bar_water_01)

e_ac_sig_water_60bar_05_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_60bar_water_05.t, e_ac_sig_water_60bar_05, t_first_peak_60bar_water_05, t_second_peak_60bar_water_05)
e_in_water_60bar_05_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_60bar_water_05.t, e_in_60bar_05, t_first_peak_60bar_water_05, t_second_peak_60bar_water_05)
e_out_water_60bar_05_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_60bar_water_05.t, e_out_60bar_05, t_first_peak_60bar_water_05, t_second_peak_60bar_water_05)
e_piston_water_60bar_05_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_60bar_water_05.t, e_piston_60bar_05, t_first_peak_60bar_water_05, t_second_peak_60bar_water_05)

e_ac_sig_water_60bar_1_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_60bar_water_1.t, e_ac_sig_water_60bar_1, t_first_peak_60bar_water_1, t_second_peak_60bar_water_1)
e_in_water_60bar_1_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_60bar_water_1.t, e_in_60bar_1, t_first_peak_60bar_water_1, t_second_peak_60bar_water_1)
e_out_water_60bar_1_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_60bar_water_1.t, e_out_60bar_1, t_first_peak_60bar_water_1, t_second_peak_60bar_water_1)
e_piston_water_60bar_1_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_60bar_water_1.t, e_piston_60bar_1, t_first_peak_60bar_water_1, t_second_peak_60bar_water_1)

e_ac_sig_water_60bar_15_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_60bar_water_15.t, e_ac_sig_water_60bar_15, t_first_peak_60bar_water_15, t_second_peak_60bar_water_15)
e_in_water_60bar_15_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_60bar_water_15.t, e_in_60bar_15, t_first_peak_60bar_water_15, t_second_peak_60bar_water_15)
e_out_water_60bar_15_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_60bar_water_15.t, e_out_60bar_15, t_first_peak_60bar_water_15, t_second_peak_60bar_water_15)
e_piston_water_60bar_15_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_60bar_water_15.t, e_piston_60bar_15, t_first_peak_60bar_water_15, t_second_peak_60bar_water_15)

e_ac_sig_water_60bar_2_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_60bar_water_2.t, e_ac_sig_water_60bar_2, t_first_peak_60bar_water_2, t_second_peak_60bar_water_2)
e_in_water_60bar_2_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_60bar_water_2.t, e_in_60bar_2, t_first_peak_60bar_water_2, t_second_peak_60bar_water_2)
e_out_water_60bar_2_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_60bar_water_2.t, e_out_60bar_2, t_first_peak_60bar_water_2, t_second_peak_60bar_water_2)
e_piston_water_60bar_2_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_60bar_water_2.t, e_piston_60bar_2, t_first_peak_60bar_water_2, t_second_peak_60bar_water_2)

e_ac_sig_water_60bar_no_piston_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_60bar_water_no_piston.t, e_ac_sig_water_60bar_no_piston, t_first_peak_60bar_water_no_piston, t_second_peak_60bar_water_no_piston)
e_in_water_60bar_no_piston_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_60bar_water_no_piston.t, e_in_60bar_no_piston, t_first_peak_60bar_water_no_piston, t_second_peak_60bar_water_no_piston)
e_out_water_60bar_no_piston_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_60bar_water_no_piston.t, e_out_60bar_no_piston, t_first_peak_60bar_water_no_piston, t_second_peak_60bar_water_no_piston)

E_ac_sig_water_over_E_ref_60bar_from_peak_to_peak_different_M = np.array([
    A_p*e_ac_sig_water_60bar_no_piston_from_peak_to_peak/E_ref_60bar_no_piston,
    e_ac_sig_water_60bar_01_from_peak_to_peak*A_p/E_ref_60bar_M01,
    e_ac_sig_water_60bar_05_from_peak_to_peak*A_p/E_ref_60bar_M05,
    e_ac_sig_water_60bar_1_from_peak_to_peak*A_p/E_ref_60bar_M1,
    e_ac_sig_water_60bar_15_from_peak_to_peak*A_p/E_ref_60bar_M15,
    e_ac_sig_water_60bar_2_from_peak_to_peak*A_p/E_ref_60bar_M2
])

E_in_water_over_E_ref_60bar_from_peak_to_peak_different_M = np.array([
    e_in_water_60bar_no_piston_from_peak_to_peak*A_p/E_ref_60bar_no_piston,
    e_in_water_60bar_01_from_peak_to_peak*A_p/E_ref_60bar_M01,
    e_in_water_60bar_05_from_peak_to_peak*A_p/E_ref_60bar_M05,
    e_in_water_60bar_1_from_peak_to_peak*A_p/E_ref_60bar_M1,
    e_in_water_60bar_15_from_peak_to_peak*A_p/E_ref_60bar_M15,
    e_in_water_60bar_2_from_peak_to_peak*A_p/E_ref_60bar_M2
])

E_out_water_over_E_ref_60bar_from_peak_to_peak_different_M = np.array([
    e_out_water_60bar_no_piston_from_peak_to_peak*A_p/E_ref_60bar_no_piston,
    e_out_water_60bar_01_from_peak_to_peak*A_p/E_ref_60bar_M01,
    e_out_water_60bar_05_from_peak_to_peak*A_p/E_ref_60bar_M05,
    e_out_water_60bar_1_from_peak_to_peak*A_p/E_ref_60bar_M1,
    e_out_water_60bar_15_from_peak_to_peak*A_p/E_ref_60bar_M15,
    e_out_water_60bar_2_from_peak_to_peak*A_p/E_ref_60bar_M2
])

E_piston_water_over_E_ref_60bar_from_peak_to_peak_different_M = np.array([
    0,
    e_piston_water_60bar_01_from_peak_to_peak*A_p/E_ref_60bar_M01,
    e_piston_water_60bar_05_from_peak_to_peak*A_p/E_ref_60bar_M05,
    e_piston_water_60bar_1_from_peak_to_peak*A_p/E_ref_60bar_M1,
    e_piston_water_60bar_15_from_peak_to_peak*A_p/E_ref_60bar_M15,
    e_piston_water_60bar_2_from_peak_to_peak*A_p/E_ref_60bar_M2
])

#7.5 MPa, water, usual pda#

e_ac_sig_water_75bar_01, e_in_75bar_01, e_piston_75bar_01, e_out_75bar_01= en.compute_e_ac_from_signals(param_study.sol_75bar_water_01.t, param_study.v_piston_75bar_water_01, param_study.v_fluid_75bar_water_01[n,:], param_study.p_p_75bar_water_01, param_study.p_t_75bar_water_01, param_study.v_fluid_75bar_water_01, param.rho_aluminium, param.th_piston_01, param_study.a0_values_75bar_water_01)
e_ac_sig_water_75bar_05, e_in_75bar_05, e_piston_75bar_05, e_out_75bar_05= en.compute_e_ac_from_signals(param_study.sol_75bar_water_05.t, param_study.v_piston_75bar_water_05, param_study.v_fluid_75bar_water_05[n,:], param_study.p_p_75bar_water_05, param_study.p_t_75bar_water_05, param_study.v_fluid_75bar_water_05, param.rho_aluminium, param.th_piston_05, param_study.a0_values_75bar_water_05)
e_ac_sig_water_75bar_1, e_in_75bar_1, e_piston_75bar_1, e_out_75bar_1= en.compute_e_ac_from_signals(param_study.sol_75bar_water_1.t, param_study.v_piston_75bar_water_1, param_study.v_fluid_75bar_water_1[n,:], param_study.p_p_75bar_water_1, param_study.p_t_75bar_water_1, param_study.v_fluid_75bar_water_1, param.rho_aluminium, param.th_piston_1, param_study.a0_values_75bar_water_1)
e_ac_sig_water_75bar_15, e_in_75bar_15, e_piston_75bar_15, e_out_75bar_15= en.compute_e_ac_from_signals(param_study.sol_75bar_water_15.t, param_study.v_piston_75bar_water_15, param_study.v_fluid_75bar_water_15[n,:], param_study.p_p_75bar_water_15, param_study.p_t_75bar_water_15, param_study.v_fluid_75bar_water_15, param.rho_aluminium, param.th_piston_15, param_study.a0_values_75bar_water_15)
e_ac_sig_water_75bar_2, e_in_75bar_2, e_piston_75bar_2, e_out_75bar_2= en.compute_e_ac_from_signals(param_study.sol_75bar_water_2.t, param_study.v_piston_75bar_water_2, param_study.v_fluid_75bar_water_2[n,:], param_study.p_p_75bar_water_2, param_study.p_t_75bar_water_2, param_study.v_fluid_75bar_water_2, param.rho_aluminium, param.th_piston_2, param_study.a0_values_75bar_water_2)
e_ac_sig_water_75bar_no_piston, e_in_75bar_no_piston, e_out_75bar_no_piston = en.compute_e_ac_from_signals_no_piston(param_study.sol_75bar_water_no_piston.t, param_study.v_fluid_75bar_water_no_piston[0,:], param_study.v_fluid_75bar_water_no_piston[n,:], param_study.p_p_75bar_water_no_piston, param_study.p_t_75bar_water_no_piston, param_study.v_fluid_75bar_water_no_piston)

e_ac_sig_water_75bar_01_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_75bar_water_01.t, e_ac_sig_water_75bar_01, t_first_peak_75bar_water_01, t_second_peak_75bar_water_01)
e_in_water_75bar_01_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_75bar_water_01.t, e_in_75bar_01, t_first_peak_75bar_water_01, t_second_peak_75bar_water_01)
e_out_water_75bar_01_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_75bar_water_01.t, e_out_75bar_01, t_first_peak_75bar_water_01, t_second_peak_75bar_water_01)
e_piston_water_75bar_01_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_75bar_water_01.t, e_piston_75bar_01, t_first_peak_75bar_water_01, t_second_peak_75bar_water_01)

e_ac_sig_water_75bar_05_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_75bar_water_05.t, e_ac_sig_water_75bar_05, t_first_peak_75bar_water_05, t_second_peak_75bar_water_05)
e_in_water_75bar_05_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_75bar_water_05.t, e_in_75bar_05, t_first_peak_75bar_water_05, t_second_peak_75bar_water_05)
e_out_water_75bar_05_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_75bar_water_05.t, e_out_75bar_05, t_first_peak_75bar_water_05, t_second_peak_75bar_water_05)
e_piston_water_75bar_05_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_75bar_water_05.t, e_piston_75bar_05, t_first_peak_75bar_water_05, t_second_peak_75bar_water_05)

e_ac_sig_water_75bar_1_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_75bar_water_1.t, e_ac_sig_water_75bar_1, t_first_peak_75bar_water_1, t_second_peak_75bar_water_1)
e_in_water_75bar_1_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_75bar_water_1.t, e_in_75bar_1, t_first_peak_75bar_water_1, t_second_peak_75bar_water_1)
e_out_water_75bar_1_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_75bar_water_1.t, e_out_75bar_1, t_first_peak_75bar_water_1, t_second_peak_75bar_water_1)
e_piston_water_75bar_1_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_75bar_water_1.t, e_piston_75bar_1, t_first_peak_75bar_water_1, t_second_peak_75bar_water_1)

e_ac_sig_water_75bar_15_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_75bar_water_15.t, e_ac_sig_water_75bar_15, t_first_peak_75bar_water_15, t_second_peak_75bar_water_15)
e_in_water_75bar_15_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_75bar_water_15.t, e_in_75bar_15, t_first_peak_75bar_water_15, t_second_peak_75bar_water_15)
e_out_water_75bar_15_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_75bar_water_15.t, e_out_75bar_15, t_first_peak_75bar_water_15, t_second_peak_75bar_water_15)
e_piston_water_75bar_15_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_75bar_water_15.t, e_piston_75bar_15, t_first_peak_75bar_water_15, t_second_peak_75bar_water_15)

e_ac_sig_water_75bar_2_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_75bar_water_2.t, e_ac_sig_water_75bar_2, t_first_peak_75bar_water_2, t_second_peak_75bar_water_2)
e_in_water_75bar_2_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_75bar_water_2.t, e_in_75bar_2, t_first_peak_75bar_water_2, t_second_peak_75bar_water_2)
e_out_water_75bar_2_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_75bar_water_2.t, e_out_75bar_2, t_first_peak_75bar_water_2, t_second_peak_75bar_water_2)
e_piston_water_75bar_2_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_75bar_water_2.t, e_piston_75bar_2, t_first_peak_75bar_water_2, t_second_peak_75bar_water_2)

e_ac_sig_water_75bar_no_piston_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_75bar_water_no_piston.t, e_ac_sig_water_75bar_no_piston, t_first_peak_75bar_water_no_piston, t_second_peak_75bar_water_no_piston)
e_in_water_75bar_no_piston_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_75bar_water_no_piston.t, e_in_75bar_no_piston, t_first_peak_75bar_water_no_piston, t_second_peak_75bar_water_no_piston)
e_out_water_75bar_no_piston_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_75bar_water_no_piston.t, e_out_75bar_no_piston, t_first_peak_75bar_water_no_piston, t_second_peak_75bar_water_no_piston)

E_ac_sig_water_over_E_ref_75bar_from_peak_to_peak_different_M = np.array([
    A_p*e_ac_sig_water_75bar_no_piston_from_peak_to_peak/E_ref_75bar_no_piston,
    e_ac_sig_water_75bar_01_from_peak_to_peak*A_p/E_ref_75bar_M01,
    e_ac_sig_water_75bar_05_from_peak_to_peak*A_p/E_ref_75bar_M05,
    e_ac_sig_water_75bar_1_from_peak_to_peak*A_p/E_ref_75bar_M1,
    e_ac_sig_water_75bar_15_from_peak_to_peak*A_p/E_ref_75bar_M15,
    e_ac_sig_water_75bar_2_from_peak_to_peak*A_p/E_ref_75bar_M2
])

E_in_water_over_E_ref_75bar_from_peak_to_peak_different_M = np.array([
    e_in_water_75bar_no_piston_from_peak_to_peak*A_p/E_ref_75bar_no_piston,
    e_in_water_75bar_01_from_peak_to_peak*A_p/E_ref_75bar_M01,
    e_in_water_75bar_05_from_peak_to_peak*A_p/E_ref_75bar_M05,
    e_in_water_75bar_1_from_peak_to_peak*A_p/E_ref_75bar_M1,
    e_in_water_75bar_15_from_peak_to_peak*A_p/E_ref_75bar_M15,
    e_in_water_75bar_2_from_peak_to_peak*A_p/E_ref_75bar_M2
])

E_out_water_over_E_ref_75bar_from_peak_to_peak_different_M = np.array([
    e_out_water_75bar_no_piston_from_peak_to_peak*A_p/E_ref_75bar_no_piston,
    e_out_water_75bar_01_from_peak_to_peak*A_p/E_ref_75bar_M01,
    e_out_water_75bar_05_from_peak_to_peak*A_p/E_ref_75bar_M05,
    e_out_water_75bar_1_from_peak_to_peak*A_p/E_ref_75bar_M1,
    e_out_water_75bar_15_from_peak_to_peak*A_p/E_ref_75bar_M15,
    e_out_water_75bar_2_from_peak_to_peak*A_p/E_ref_75bar_M2
])

E_piston_water_over_E_ref_75bar_from_peak_to_peak_different_M = np.array([
    0,
    e_piston_water_75bar_01_from_peak_to_peak*A_p/E_ref_75bar_M01,
    e_piston_water_75bar_05_from_peak_to_peak*A_p/E_ref_75bar_M05,
    e_piston_water_75bar_1_from_peak_to_peak*A_p/E_ref_75bar_M1,
    e_piston_water_75bar_15_from_peak_to_peak*A_p/E_ref_75bar_M15,
    e_piston_water_75bar_2_from_peak_to_peak*A_p/E_ref_75bar_M2
])

#10 MPa, water, usual pda#

e_ac_sig_water_100bar_01, e_in_100bar_01, e_piston_100bar_01, e_out_100bar_01= en.compute_e_ac_from_signals(param_study.sol_100bar_water_01.t, param_study.v_piston_100bar_water_01, param_study.v_fluid_100bar_water_01[n,:], param_study.p_p_100bar_water_01, param_study.p_t_100bar_water_01, param_study.v_fluid_100bar_water_01, param.rho_aluminium, param.th_piston_01, param_study.a0_values_100bar_water_01)
e_ac_sig_water_100bar_05, e_in_100bar_05, e_piston_100bar_05, e_out_100bar_05= en.compute_e_ac_from_signals(param_study.sol_100bar_water_05.t, param_study.v_piston_100bar_water_05, param_study.v_fluid_100bar_water_05[n,:], param_study.p_p_100bar_water_05, param_study.p_t_100bar_water_05, param_study.v_fluid_100bar_water_05, param.rho_aluminium, param.th_piston_05, param_study.a0_values_100bar_water_05)
e_ac_sig_water_100bar_1, e_in_100bar_1, e_piston_100bar_1, e_out_100bar_1= en.compute_e_ac_from_signals(param_study.sol_100bar_water_1.t, param_study.v_piston_100bar_water_1, param_study.v_fluid_100bar_water_1[n,:], param_study.p_p_100bar_water_1, param_study.p_t_100bar_water_1, param_study.v_fluid_100bar_water_1, param.rho_aluminium, param.th_piston_1, param_study.a0_values_100bar_water_1)
e_ac_sig_water_100bar_15, e_in_100bar_15, e_piston_100bar_15, e_out_100bar_15= en.compute_e_ac_from_signals(param_study.sol_100bar_water_15.t, param_study.v_piston_100bar_water_15, param_study.v_fluid_100bar_water_15[n,:], param_study.p_p_100bar_water_15, param_study.p_t_100bar_water_15, param_study.v_fluid_100bar_water_15, param.rho_aluminium, param.th_piston_15, param_study.a0_values_100bar_water_15)
e_ac_sig_water_100bar_2, e_in_100bar_2, e_piston_100bar_2, e_out_100bar_2= en.compute_e_ac_from_signals(param_study.sol_100bar_water_2.t, param_study.v_piston_100bar_water_2, param_study.v_fluid_100bar_water_2[n,:], param_study.p_p_100bar_water_2, param_study.p_t_100bar_water_2, param_study.v_fluid_100bar_water_2, param.rho_aluminium, param.th_piston_2, param_study.a0_values_100bar_water_2)
e_ac_sig_water_100bar_no_piston, e_in_100bar_no_piston, e_out_100bar_no_piston = en.compute_e_ac_from_signals_no_piston(param_study.sol_100bar_water_no_piston.t, param_study.v_fluid_100bar_water_no_piston[0,:], param_study.v_fluid_100bar_water_no_piston[n,:], param_study.p_p_100bar_water_no_piston, param_study.p_t_100bar_water_no_piston, param_study.v_fluid_100bar_water_no_piston)

e_ac_sig_water_100bar_01_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_100bar_water_01.t, e_ac_sig_water_100bar_01, t_first_peak_100bar_water_01, t_second_peak_100bar_water_01)
e_in_water_100bar_01_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_100bar_water_01.t, e_in_100bar_01, t_first_peak_100bar_water_01, t_second_peak_100bar_water_01)
e_out_water_100bar_01_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_100bar_water_01.t, e_out_100bar_01, t_first_peak_100bar_water_01, t_second_peak_100bar_water_01)
e_piston_water_100bar_01_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_100bar_water_01.t, e_piston_100bar_01, t_first_peak_100bar_water_01, t_second_peak_100bar_water_01)

e_ac_sig_water_100bar_05_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_100bar_water_05.t, e_ac_sig_water_100bar_05, t_first_peak_100bar_water_05, t_second_peak_100bar_water_05)
e_in_water_100bar_05_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_100bar_water_05.t, e_in_100bar_05, t_first_peak_100bar_water_05, t_second_peak_100bar_water_05)
e_out_water_100bar_05_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_100bar_water_05.t, e_out_100bar_05, t_first_peak_100bar_water_05, t_second_peak_100bar_water_05)
e_piston_water_100bar_05_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_100bar_water_05.t, e_piston_100bar_05, t_first_peak_100bar_water_05, t_second_peak_100bar_water_05)

e_ac_sig_water_100bar_1_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_100bar_water_1.t, e_ac_sig_water_100bar_1, t_first_peak_100bar_water_1, t_second_peak_100bar_water_1)
e_in_water_100bar_1_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_100bar_water_1.t, e_in_100bar_1, t_first_peak_100bar_water_1, t_second_peak_100bar_water_1)
e_out_water_100bar_1_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_100bar_water_1.t, e_out_100bar_1, t_first_peak_100bar_water_1, t_second_peak_100bar_water_1)
e_piston_water_100bar_1_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_100bar_water_1.t, e_piston_100bar_1, t_first_peak_100bar_water_1, t_second_peak_100bar_water_1)

e_ac_sig_water_100bar_15_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_100bar_water_15.t, e_ac_sig_water_100bar_15, t_first_peak_100bar_water_15, t_second_peak_100bar_water_15)
e_in_water_100bar_15_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_100bar_water_15.t, e_in_100bar_15, t_first_peak_100bar_water_15, t_second_peak_100bar_water_15)
e_out_water_100bar_15_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_100bar_water_15.t, e_out_100bar_15, t_first_peak_100bar_water_15, t_second_peak_100bar_water_15)
e_piston_water_100bar_15_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_100bar_water_15.t, e_piston_100bar_15, t_first_peak_100bar_water_15, t_second_peak_100bar_water_15)

e_ac_sig_water_100bar_2_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_100bar_water_2.t, e_ac_sig_water_100bar_2, t_first_peak_100bar_water_2, t_second_peak_100bar_water_2)
e_in_water_100bar_2_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_100bar_water_2.t, e_in_100bar_2, t_first_peak_100bar_water_2, t_second_peak_100bar_water_2)
e_out_water_100bar_2_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_100bar_water_2.t, e_out_100bar_2, t_first_peak_100bar_water_2, t_second_peak_100bar_water_2)
e_piston_water_100bar_2_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_100bar_water_2.t, e_piston_100bar_2, t_first_peak_100bar_water_2, t_second_peak_100bar_water_2)

e_ac_sig_water_100bar_no_piston_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_100bar_water_no_piston.t, e_ac_sig_water_100bar_no_piston, t_first_peak_100bar_water_no_piston, t_second_peak_100bar_water_no_piston)
e_in_water_100bar_no_piston_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_100bar_water_no_piston.t, e_in_100bar_no_piston, t_first_peak_100bar_water_no_piston, t_second_peak_100bar_water_no_piston)
e_out_water_100bar_no_piston_from_peak_to_peak=en.mean_e_from_peak_to_peak(param_study.sol_100bar_water_no_piston.t, e_out_100bar_no_piston, t_first_peak_100bar_water_no_piston, t_second_peak_100bar_water_no_piston)

E_ac_sig_water_over_E_ref_100bar_from_peak_to_peak_different_M = np.array([
    A_p*e_ac_sig_water_100bar_no_piston_from_peak_to_peak/E_ref_100bar_no_piston,
    e_ac_sig_water_100bar_01_from_peak_to_peak*A_p/E_ref_100bar_M01,
    e_ac_sig_water_100bar_05_from_peak_to_peak*A_p/E_ref_100bar_M05,
    e_ac_sig_water_100bar_1_from_peak_to_peak*A_p/E_ref_100bar_M1,
    e_ac_sig_water_100bar_15_from_peak_to_peak*A_p/E_ref_100bar_M15,
    e_ac_sig_water_100bar_2_from_peak_to_peak*A_p/E_ref_100bar_M2
])

E_in_water_over_E_ref_100bar_from_peak_to_peak_different_M = np.array([
    e_in_water_100bar_no_piston_from_peak_to_peak*A_p/E_ref_100bar_no_piston,
    e_in_water_100bar_01_from_peak_to_peak*A_p/E_ref_100bar_M01,
    e_in_water_100bar_05_from_peak_to_peak*A_p/E_ref_100bar_M05,
    e_in_water_100bar_1_from_peak_to_peak*A_p/E_ref_100bar_M1,
    e_in_water_100bar_15_from_peak_to_peak*A_p/E_ref_100bar_M15,
    e_in_water_100bar_2_from_peak_to_peak*A_p/E_ref_100bar_M2
])

E_out_water_over_E_ref_100bar_from_peak_to_peak_different_M = np.array([
    e_out_water_100bar_no_piston_from_peak_to_peak*A_p/E_ref_100bar_no_piston,
    e_out_water_100bar_01_from_peak_to_peak*A_p/E_ref_100bar_M01,
    e_out_water_100bar_05_from_peak_to_peak*A_p/E_ref_100bar_M05,
    e_out_water_100bar_1_from_peak_to_peak*A_p/E_ref_100bar_M1,
    e_out_water_100bar_15_from_peak_to_peak*A_p/E_ref_100bar_M15,
    e_out_water_100bar_2_from_peak_to_peak*A_p/E_ref_100bar_M2
])

E_piston_water_over_E_ref_100bar_from_peak_to_peak_different_M = np.array([
    0,
    e_piston_water_100bar_01_from_peak_to_peak*A_p/E_ref_100bar_M01,
    e_piston_water_100bar_05_from_peak_to_peak*A_p/E_ref_100bar_M05,
    e_piston_water_100bar_1_from_peak_to_peak*A_p/E_ref_100bar_M1,
    e_piston_water_100bar_15_from_peak_to_peak*A_p/E_ref_100bar_M15,
    e_piston_water_100bar_2_from_peak_to_peak*A_p/E_ref_100bar_M2
])