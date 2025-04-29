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
composition_initial = 'O2:2, N2:7.52'  # oxidizer-only stream
gas.TPX = initial_temperature, initial_pressure, composition_initial

# -----------------------------
# Step 2: Plug flow setup
# -----------------------------
velocity = 820.0  # m/s (start subsonic)
length = 0.7  # m
dx = 0.001  # m
n_steps = int(length / dx)
dt = dx / velocity  # time step
CH3OH_injection_point = 0.05  # m
injection_length = 0.5     # m
mixing_time = 0.002  # seconds, you can tune this
fuel_target_X = 0.05  # mole fraction of ethylene after full mixing

reactor = ct.IdealGasConstPressureReactor(gas)
sim = ct.ReactorNet([reactor])

# -----------------------------
# Step 3: Initialize tracking
# -----------------------------
positions = []
temperatures = []
CH3OH_X, CO_X, CO2_X, H2O_X, OH_X = [], [], [], [], []
mach_numbers = []
shock_locations = []
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
for i in range(n_steps):
    distance += dx
    sim.advance(sim.time + dt)

    # Non-premixed fuel injection (gradual)
    if distance >= CH3OH_injection_point:
        mix_fraction = min((distance - CH3OH_injection_point) / injection_length, 1.0)
        O2 = 2.0 * (1 - mix_fraction)
        N2 = 7.52 * (1 - mix_fraction)
        CH3OH = 1.0 * mix_fraction
        new_comp = f'CH3OH:{CH3OH}, O2:{O2}, N2:{N2}'
        gas.TPX = reactor.T, reactor.thermo.P, new_comp
        reactor.syncState()

    T = reactor.T
    P = reactor.thermo.P
    X = reactor.thermo.X
    M = estimate_mach(T, velocity, gas)
    mach_numbers.append(M)

    # Shock check
    if M > shock_threshold and shock_count < max_shocks:
        M2, T2, P2, velocity = apply_normal_shock_full(M, T, P, gas)
        gas.TPX = T2, P2, X
        reactor.syncState()
        shock_locations.append(distance)
        shock_count += 1

    positions.append(distance)
    temperatures.append(T)
    CH3OH_X.append(X[gas.species_index('CH3OH')])
    CO_X.append(X[gas.species_index('CO')])
    CO2_X.append(X[gas.species_index('CO2')])
    H2O_X.append(X[gas.species_index('H2O')])
    OH_X.append(X[gas.species_index('OH')])

    if not ignited and X[gas.species_index('CH3OH')] < 0.9 and T - initial_temperature > 100:
        ignited = True
        ignition_position = distance

    if post_combustion_index is None and X[gas.species_index('CH3OH')] < 1e-4:
        post_combustion_index = i

# -----------------------------
# Step 5: Combustion Calculations
# -----------------------------
initial_CH3OH = CH3OH_X[0] if CH3OH_X[0] > 0 else 1.0
final_CH3OH = CH3OH_X[-1]
CH3OH_conversion = 100 * (initial_CH3OH - final_CH3OH) / initial_CH3OH

h_initial = gas.enthalpy_mole
gas_equil = ct.Solution('gri30.yaml')
gas_equil.TPX = initial_temperature, initial_pressure, 'CH3OH:1, O2:2, N2:7.52'
gas_equil.equilibrate('HP')
T_ad = gas_equil.T
h_final_equil = gas_equil.enthalpy_mole

if post_combustion_index is not None:
    post_combustion_gas = ct.Solution('gri30.yaml')
    raw_composition = {
        'CH3OH': CH3OH_X[post_combustion_index],
        'CO': CO_X[post_combustion_index],
        'CO2': CO2_X[post_combustion_index],
        'H2O': H2O_X[post_combustion_index],
        'OH': OH_X[post_combustion_index],
        'O2': gas['O2'].X[0],
        'N2': gas['N2'].X[0],
    }
    total = sum(raw_composition.values())
    normalized_composition = {k: v / total for k, v in raw_composition.items()}
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
    print(f"\n Ignition occurred at: {ignition_position:.3f} m")
else:
    print("\n No ignition detected.")

print(f" C₂H₄ Conversion: {CH3OH_conversion:.2f}%")
print(f" Adiabatic Flame Temp: {T_ad:.2f} K")
print(f" Combustion Efficiency: {combustion_efficiency:.2f}%")
print(f" Number of shocks detected: {shock_count}")

# -----------------------------
# Step 7: Plot results
# -----------------------------
zoom_range = 0.2

fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(positions, temperatures, label='Temperature')
axs[0, 0].axvline(CH3OH_injection_point, color='black', linestyle='--', label='C₂H₄ Injected')
if ignition_position:
    axs[0, 0].axvline(ignition_position, color='gray', linestyle='--', label='Ignition')
for j, shock_pos in enumerate(shock_locations):
    axs[0, 0].axvline(shock_pos, color='purple', linestyle=':', label=f'Shock {j+1}' if j == 0 else "")
axs[0, 0].set_title('Temperature')
axs[0, 0].set_xlim(0)
axs[0, 0].legend()
axs[0, 0].grid()

axs[0, 1].plot(positions, CH3OH_X, 'r', label='C₂H₄')
axs[0, 1].set_xlim(0)
axs[0, 1].set_title('C₂H₄ Depletion')
axs[0, 1].legend()
axs[0, 1].grid()

axs[1, 0].plot(positions, CO_X, label='CO')
axs[1, 0].plot(positions, CO2_X, label='CO₂')
axs[1, 0].set_xlim(0)
axs[1, 0].set_title('Carbon Products')
axs[1, 0].legend()
axs[1, 0].grid()

axs[1, 1].plot(positions, H2O_X, label='H₂O')
axs[1, 1].plot(positions, OH_X, label='OH')
axs[1, 1].set_xlim(0)
axs[1, 1].set_title('Water and Radicals')
axs[1, 1].legend()
axs[1, 1].grid()

plt.tight_layout()

# Mach number
plt.figure()
plt.plot(positions, mach_numbers, label='Mach Number')
plt.axhline(1.0, color='gray', linestyle='--', label='Sonic Limit')
for j, shock_pos in enumerate(shock_locations):
    plt.axvline(shock_pos, color='purple', linestyle=':', label=f'Shock {j+1}' if j == 0 else "")
plt.axvline(CH3OH_injection_point, color='black', linestyle='--', label='C₂H₄ Injected')
plt.title("Mach Number Along Reactor")
plt.xlabel("Distance (m)")
plt.ylabel("Mach")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
