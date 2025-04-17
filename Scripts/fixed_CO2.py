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
initial_temperature = 1300
initial_pressure = ct.one_atm
gas.TPX = initial_temperature, initial_pressure, 'O2:2, N2:7.52'

# -----------------------------
# Step 2: Reactor Setup
# -----------------------------
velocity = 1000.0
length = 0.7
dx = 0.001
n_steps = int(length / dx)
dt = dx / velocity
CH3OH_injection_point = 0.05
injection_length = 0.5

reactor = ct.IdealGasConstPressureReactor(gas)
reactor.energy_enabled = True
reservoir = ct.Reservoir(gas)
wall = ct.Wall(left=reactor, right=reservoir)
sim = ct.ReactorNet([reactor])

# Heat loss model (more aggressive cooling)
h = 30000.0  # Heat transfer coefficient
T_wall = 300.0
diameter = 0.04
perimeter = np.pi * diameter

# -----------------------------
# Step 3: Data Storage
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
# Step 4: Main Loop
# -----------------------------
distance = 0.0
for i in range(n_steps):
    distance += dx
    sim.advance(sim.time + dt)

    if distance >= CH3OH_injection_point:
        mix_fraction = min((distance - CH3OH_injection_point) / injection_length, 1.0)
        CH3OH = 1.0 * mix_fraction
        O2 = 1.5 * (1 - mix_fraction)
        N2 = 1.5 * 3.76 * (1 - mix_fraction)
        gas.TPX = reactor.T, reactor.thermo.P, f'CH3OH:{CH3OH}, O2:{O2}, N2:{N2}'
        reactor.syncState()

    # Apply external heat loss using shadow gas
    shadow_gas = ct.Solution('gri30.yaml')
    shadow_gas.TPX = reactor.T, reactor.thermo.P, reactor.thermo.X

    T = shadow_gas.T
    mass = reactor.mass
    qdot = h * perimeter * (T - T_wall)
    delta_h = -qdot * dt  # total heat loss
    delta_h_mass = delta_h / mass
    new_h = shadow_gas.enthalpy_mass + delta_h_mass

    shadow_gas.HP = new_h, shadow_gas.P
    gas.TPX = shadow_gas.T, shadow_gas.P, shadow_gas.X
    reactor.syncState()

    P = reactor.thermo.P
    X = reactor.thermo.X
    M = estimate_mach(reactor.T, velocity, gas)
    mach_numbers.append(M)

    if M > shock_threshold and shock_count < max_shocks:
        M2, T2, P2, velocity = apply_normal_shock_full(M, reactor.T, P, gas)
        gas.TPX = T2, P2, X
        reactor.syncState()
        shock_locations.append(distance)
        shock_count += 1

    positions.append(distance)
    temperatures.append(0.85 * reactor.T)
    CH3OH_X.append(X[gas.species_index('CH3OH')])
    CO_X.append(X[gas.species_index('CO')])
    CO2_X.append(X[gas.species_index('CO2')])
    H2O_X.append(X[gas.species_index('H2O')])
    OH_X.append(X[gas.species_index('OH')])

    if not ignited and X[gas.species_index('CH3OH')] < 0.9 and reactor.T - initial_temperature > 100:
        ignited = True
        ignition_position = distance

    if post_combustion_index is None and X[gas.species_index('CH3OH')] < 1e-4:
        post_combustion_index = i

# -----------------------------
# Step 5: Plotting
# -----------------------------
fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(positions, [0.85 * T for T in temperatures], label='Temperature (scaled)')
axs[0, 0].axvline(CH3OH_injection_point, color='black', linestyle='--', label='CH₃OH Injected')
if ignition_position:
    axs[0, 0].axvline(ignition_position, color='gray', linestyle='--', label='Ignition')
for j, shock_pos in enumerate(shock_locations):
    axs[0, 0].axvline(shock_pos, color='purple', linestyle=':', label=f'Shock {j+1}' if j == 0 else "")
axs[0, 0].set_title('Temperature')
axs[0, 0].legend()
axs[0, 0].grid()

axs[0, 1].plot(positions, CH3OH_X, 'r', label='CH₃OH')
axs[0, 1].set_title('CH₃OH Depletion')
axs[0, 1].legend()
axs[0, 1].grid()

axs[1, 0].plot(positions, CO_X, label='CO')
axs[1, 0].plot(positions, CO2_X, label='CO₂')
axs[1, 0].set_title('Carbon Products')
axs[1, 0].legend()
axs[1, 0].grid()

axs[1, 1].plot(positions, H2O_X, label='H₂O')
axs[1, 1].plot(positions, OH_X, label='OH')
axs[1, 1].set_title('Water and Radicals')
axs[1, 1].legend()
axs[1, 1].grid()

plt.tight_layout()

plt.figure()
plt.plot(positions, mach_numbers, label='Mach Number')
plt.axhline(1.0, color='gray', linestyle='--', label='Sonic Limit')
for j, shock_pos in enumerate(shock_locations):
    plt.axvline(shock_pos, color='purple', linestyle=':', label=f'Shock {j+1}' if j == 0 else "")
plt.axvline(CH3OH_injection_point, color='black', linestyle='--', label='CH₃OH Injected')
plt.title("Mach Number Along Reactor")
plt.xlabel("Distance (m)")
plt.ylabel("Mach")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
