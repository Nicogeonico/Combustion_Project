import cantera as ct
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# -----------------------------
# Helper Functions
# -----------------------------
def estimate_mach(T, velocity, gas):
    gamma = gas.cp_mass / gas.cv_mass
    R = ct.gas_constant / gas.mean_molecular_weight
    a = np.sqrt(gamma * R * T)
    return velocity / a

def apply_normal_shock_full(M1, T1, P1, gas):
    gamma = gas.cp_mass / gas.cv_mass
    M2 = np.sqrt(((gamma - 1) * M1**2 + 2) / (2 * gamma * M1**2 - (gamma - 1)))
    P2 = P1 * (1 + 2 * gamma / (gamma + 1) * (M1**2 - 1))
    temp_ratio = ((1 + ((gamma - 1) / 2) * M1**2) / (1 + ((gamma - 1) / 2) * M2**2))
    T2 = T1 * temp_ratio * (P2 / P1)
    R = ct.gas_constant / gas.mean_molecular_weight
    a2 = np.sqrt(gamma * R * T2)
    velocity2 = M2 * a2
    return M2, T2, P2, velocity2

# -----------------------------
# Step 1: Initial Setup
# -----------------------------
gas = ct.Solution('gri30.yaml')
initial_temperature = 1300  # K
initial_pressure = ct.one_atm
composition_initial = 'O2:2, N2:7.52'
gas.TPX = initial_temperature, initial_pressure, composition_initial

reactor = ct.IdealGasConstPressureReactor(gas)
sim = ct.ReactorNet([reactor])

# -----------------------------
# Step 2: Plug flow setup
# -----------------------------
velocity = 1400.0  # m/s
length = 0.7  # m
dx = 0.001  # m
n_steps = int(length / dx)
dt = dx / velocity
C2H4_injection_point_1 = 0.05
C2H4_injection_point_2 = 0.2
injection_length_1 = 0.2
mixing_time = 0.002
fuel_target_X = 0.05

# -----------------------------
# Step 3: Initialize tracking
# -----------------------------
positions, temperatures = [], []
C2H4_X, CO_X, CO2_X, H2O_X, OH_X = [], [], [], [], []
mach_numbers, shock_locations = [], []
speed_of_sound, thrust_per_area, velocities = [], [], []
shock_count = 0
max_shocks = 2
shock_threshold = 1.1
ignited = False
ignition_position = None
post_combustion_index = None

# -----------------------------
# Step 4: Run simulation
# -----------------------------
distance = 0.0
injecting_fuel = True
F_O_stoic = 1/3
phi_target = 1.0
V_local = 750

for i in range(n_steps):
    distance += dx
    sim.advance(sim.time + dt)

    if injecting_fuel and distance >= C2H4_injection_point_1:
        mix_fraction = min(((distance - C2H4_injection_point_1) / injection_length_1) ** 2, 1.0)
        added_C2H4 = 0.01 * mix_fraction

        current_X = reactor.thermo.X
        current_X[gas.species_index('C2H4')] += added_C2H4
        current_X /= np.sum(current_X)
        reactor.thermo.X = current_X
        reactor.syncState()

        X_fuel = reactor.thermo['C2H4'].X[0]
        X_O2 = reactor.thermo['O2'].X[0]

        if X_O2 > 1e-8:
            phi_current = (X_fuel / X_O2) / F_O_stoic
        else:
            phi_current = 10.0

        if phi_current >= phi_target:
            injecting_fuel = False

    T = reactor.T
    P = reactor.thermo.P
    X = reactor.thermo.X
    M = estimate_mach(T, velocity, gas)
    mach_numbers.append(M)

    gamma = gas.cp_mass / gas.cv_mass
    R_specific = ct.gas_constant / gas.mean_molecular_weight
    rho_local = P / (R_specific * T)
    a_local = np.sqrt(gamma * R_specific * T)
    M = V_local / a_local
    velocities.append(V_local)

    if i == 0:
        V_inlet = V_local
        rho_inlet = rho_local
    mass_flux_local = rho_local * V_local
    thrust_local = mass_flux_local * (V_local - V_inlet)
    thrust_per_area.append(thrust_local)

    if M > shock_threshold and shock_count < max_shocks:
        M2, T2, P2, V_local = apply_normal_shock_full(M, T, P, gas)
        gas.TPX = T2, P2, X
        reactor.syncState()
        shock_locations.append(distance)
        shock_count += 1

    positions.append(distance)
    temperatures.append(T)
    C2H4_X.append(X[gas.species_index('C2H4')])
    CO_X.append(X[gas.species_index('CO')])
    CO2_X.append(X[gas.species_index('CO2')])
    H2O_X.append(X[gas.species_index('H2O')])
    OH_X.append(X[gas.species_index('OH')])

    if not ignited and X[gas.species_index('C2H4')] > 1e-3 and T - initial_temperature > 100:
        ignited = True
        ignition_position = distance

    if ignited and ignition_position is not None and post_combustion_index is None and distance > ignition_position and X[gas.species_index('C2H4')] < 1e-6 and T > 1500:
        post_combustion_index = i

# -----------------------------
# Step 5: Combustion Calculations
# -----------------------------
initial_C2H4 = C2H4_X[0] if C2H4_X[0] > 0 else 1.0
final_C2H4 = C2H4_X[-1]
C2H4_conversion = 100 * (initial_C2H4 - final_C2H4) / initial_C2H4

h_initial = gas.enthalpy_mole
gas_equil = ct.Solution('gri30.yaml')
gas_equil.TPX = initial_temperature, initial_pressure, 'C2H4:1, O2:3, N2:7.52'
gas_equil.equilibrate('HP')
T_ad = gas_equil.T
h_final_equil = gas_equil.enthalpy_mole

if post_combustion_index is not None:
    post_combustion_gas = ct.Solution('gri30.yaml')
    raw_composition = {
        'C2H4': C2H4_X[post_combustion_index],
        'CO': CO_X[post_combustion_index],
        'CO2': CO2_X[post_combustion_index],
        'H2O': H2O_X[post_combustion_index],
        'OH': OH_X[post_combustion_index],
        'O2': gas['O2'].X[0],
        'N2': gas['N2'].X[0],
    }
    total = sum(raw_composition.values())

    if total > 1e-8:
        normalized_composition = {k: v / total for k, v in raw_composition.items()}
    else:
        normalized_composition = raw_composition

    post_combustion_gas.TPX = temperatures[post_combustion_index], initial_pressure, normalized_composition
    h_final_actual = post_combustion_gas.enthalpy_mole
else:
    h_final_actual = reactor.thermo.enthalpy_mole

delta_h_ideal = h_final_equil - h_initial
delta_h_actual = h_final_actual - h_initial
combustion_efficiency = 100 * delta_h_actual / delta_h_ideal if delta_h_ideal != 0 else 0.0

# -----------------------------
# Step 6: Print results
# -----------------------------
if ignition_position is not None:
    print(f"\nIgnition occurred at: {ignition_position:.3f} m")
else:
    print("\nNo ignition detected.")

print(f"C2H4 Conversion: {C2H4_conversion:.2f}%")
print(f"Adiabatic Flame Temp: {T_ad:.2f} K")
print(f"Combustion Efficiency: {combustion_efficiency:.2f}%")
print(f"Number of shocks detected: {shock_count}")

# -----------------------------
# Step 7: Plot results
# -----------------------------
zoom_range = 0.2

fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(positions, temperatures, label='Temperature')
axs[0, 0].axvline(C2H4_injection_point_1, color='black', linestyle='--', label='C₂H₄ Injected')
axs[0, 0].set_xlim(0)
axs[0, 0].legend()
axs[0, 0].grid()

axs[0, 1].plot(positions, C2H4_X, 'r', label='C₂H₄')
axs[0, 1].set_xlim(0)
axs[0, 1].legend()
axs[0, 1].grid()

axs[1, 0].plot(positions, CO_X, label='CO')
axs[1, 0].plot(positions, CO2_X, label='CO₂')
axs[1, 0].set_xlim(0)
axs[1, 0].legend()
axs[1, 0].grid()

axs[1, 1].plot(positions, H2O_X, label='H₂O')
axs[1, 1].plot(positions, OH_X, label='OH')
axs[1, 1].set_xlim(0)
axs[1, 1].legend()
axs[1, 1].grid()

plt.tight_layout()

# Mach number
plt.figure()
plt.plot(positions, mach_numbers, label='Mach Number')
plt.axhline(1.0, color='gray', linestyle='--', label='Sonic Limit')
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()
