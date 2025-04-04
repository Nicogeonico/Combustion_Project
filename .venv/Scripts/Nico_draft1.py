import cantera as ct
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Use this backend if you're running in PyCharm
import matplotlib.pyplot as plt

# -----------------------------
# Step 1: Load mechanism and set initial conditions
# -----------------------------
gas = ct.Solution('gri30.yaml')
initial_temperature = 1700  # K
initial_pressure = ct.one_atm
composition = 'CH4:1, O2:2, N2:7.52'  "1"
gas.TPX = initial_temperature, initial_pressure, composition

# -----------------------------
# Step 2: Plug flow setup
# -----------------------------
velocity = 200.0  # m/s
length = 0.7  # m
dx = 0.001  # m
n_steps = int(length / dx)
dt = dx / velocity  # time step

reactor = ct.IdealGasConstPressureReactor(gas)
sim = ct.ReactorNet([reactor])

# -----------------------------
# Step 3: Initialize storage
# -----------------------------
positions = []
temperatures = []
CH4_X, CO_X, CO2_X, H2O_X, OH_X = [], [], [], [], []

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

    T = reactor.T
    X = reactor.thermo.X

    positions.append(distance)
    temperatures.append(T)
    CH4_X.append(X[gas.species_index('CH4')])
    CO_X.append(X[gas.species_index('CO')])
    CO2_X.append(X[gas.species_index('CO2')])
    H2O_X.append(X[gas.species_index('H2O')])
    OH_X.append(X[gas.species_index('OH')])

    if not ignited and T - initial_temperature > 200:
        ignited = True
        ignition_position = distance

    if post_combustion_index is None and X[gas.species_index('CH4')] < 1e-4:
        post_combustion_index = i

# -----------------------------
# Step 5: Combustion performance calculations
# -----------------------------
initial_CH4 = CH4_X[0]
final_CH4 = CH4_X[-1]
CH4_conversion = 100 * (initial_CH4 - final_CH4) / initial_CH4

# Initial enthalpy
h_initial = gas.enthalpy_mole

# Equilibrium (ideal) state
gas_equil = ct.Solution('gri30.yaml')
gas_equil.TPX = initial_temperature, initial_pressure, composition
gas_equil.equilibrate('HP')
T_ad = gas_equil.T
h_final_equil = gas_equil.enthalpy_mole

# Actual enthalpy from post-combustion gas
if post_combustion_index is not None:
    post_combustion_gas = ct.Solution('gri30.yaml')

    # Build and normalize mole fractions manually
    raw_composition = {
        'CH4': CH4_X[post_combustion_index],
        'CO': CO_X[post_combustion_index],
        'CO2': CO2_X[post_combustion_index],
        'H2O': H2O_X[post_combustion_index],
        'OH': OH_X[post_combustion_index],
        'O2': gas['O2'].X[0],
        'N2': gas['N2'].X[0],
    }

    total = sum(raw_composition.values())
    normalized_composition = {species: val / total for species, val in raw_composition.items()}

    post_combustion_gas.TPX = (
        temperatures[post_combustion_index],
        initial_pressure,
        normalized_composition
    )

    h_final_actual = post_combustion_gas.enthalpy_mole
else:
    h_final_actual = reactor.thermo.enthalpy_mole

# Combustion efficiency
delta_h_ideal = h_final_equil - h_initial
delta_h_actual = h_final_actual - h_initial

if delta_h_ideal != 0:
    combustion_efficiency = 100 * delta_h_actual / delta_h_ideal
else:
    combustion_efficiency = 0.0

# -----------------------------
# Step 6: Print results
# -----------------------------
if ignition_position is not None:
    print(f"\n🔥 Ignition occurred at: {ignition_position:.3f} m (≈ {ignition_position / velocity:.6f} s)")
else:
    print("\n⚠️ No ignition detected.")

print(f"💨 CH₄ Conversion: {CH4_conversion:.2f}%")
print(f"🔥 Adiabatic Flame Temp: {T_ad:.2f} K")
print(f"⚙️ Combustion Efficiency (post-combustion): {combustion_efficiency:.2f}%")

# -----------------------------
# Step 7: Plot results
# -----------------------------
fig, axs = plt.subplots(2, 2, figsize=(14, 8))

# Temperature
axs[0, 0].plot(positions, temperatures, label='Temperature')
if ignition_position:
    axs[0, 0].axvline(ignition_position, color='gray', linestyle='--', label='Ignition Point')
axs[0, 0].set_title('Temperature Rise Along Combustor')
axs[0, 0].set_xlabel('Distance (m)')
axs[0, 0].set_ylabel('Temperature (K)')
axs[0, 0].legend()
axs[0, 0].grid(True)

# CH₄
axs[0, 1].plot(positions, CH4_X, 'r', label='CH₄')
axs[0, 1].set_title('CH₄ Depletion')
axs[0, 1].set_xlabel('Distance (m)')
axs[0, 1].set_ylabel('Mole Fraction')
axs[0, 1].legend()
axs[0, 1].grid(True)

# CO and CO₂
axs[1, 0].plot(positions, CO_X, label='CO')
axs[1, 0].plot(positions, CO2_X, label='CO₂')
axs[1, 0].set_title('Carbon Products')
axs[1, 0].set_xlabel('Distance (m)')
axs[1, 0].set_ylabel('Mole Fraction')
axs[1, 0].legend()
axs[1, 0].grid(True)

# H₂O and OH
axs[1, 1].plot(positions, H2O_X, label='H₂O')
axs[1, 1].plot(positions, OH_X, label='OH')
axs[1, 1].set_title('Water and Radical Species')
axs[1, 1].set_xlabel('Distance (m)')
axs[1, 1].set_ylabel('Mole Fraction')
axs[1, 1].legend()
axs[1, 1].grid(True)

plt.tight_layout()
plt.show()
