# import statements
from functions_fd import *

# define the boundary and initial conditions
x0 = {'type': 'dirichlet', 'function': lambda x, t: 200.}
x1 = {'type': 'dirichlet', 'function': lambda x, t: 200.}
t0 = {
    'type': 'initial',
    'function': lambda x, t: np.piecewise(
        x,
        [x <= 0, x >= 5, (x > 0) & (x < 5)],  # Modify these conditions
        [200., 200., 30.]
    )
}
# set up the x and t dimensions, as well as desired mesh spacing.
xlim = np.array([0., 5.])
tlim = np.array([0., 4.])


# Task 2.1
# Constants for the materials (in units consistent with ∆x and ∆t, e.g., meters, seconds)
alpha_silver = 1.5  # thermal diffusivity of silver
alpha_copper = 1.25  # thermal diffusivity of copper
alpha_aluminium = 1.0  # thermal diffusivity of gold

# Determine the most stringent condition
alpha_max = max(alpha_silver, alpha_copper, alpha_aluminium)

# Given ∆x
delta_x = 0.1

# Calculate the maximum ∆t for stability
delta_t_max = 0.5 * (delta_x ** 2) / alpha_max

print(f"The largest temporal mesh spacing for stability is: {delta_t_max} seconds")


# Task 2.2
# Parameters for silver
alpha_silver = 1.25
dx = 0.1
dt_small = 0.001
dt_large = 0.005
thetaE = 0
thetaCN = 0.5  # For Crank-Nicolson method

# Create solver instances
solver_explicit_small = SolverHeatXT(xlim, tlim, dx, dt_small, alpha_silver, thetaCN, x0, x1, t0)
solver_explicit_large = SolverHeatXT(xlim, tlim, dx, dt_large, alpha_silver, thetaCN, x0, x1, t0)
solver_implicit_small = SolverHeatXT(xlim, tlim, dx, dt_small, alpha_silver, thetaCN, x0, x1, t0)
solver_implicit_large = SolverHeatXT(xlim, tlim, dx, dt_large, alpha_silver, thetaCN, x0, x1, t0)

# Solve using different methods
solver_explicit_small.solve_explicit()
solver_explicit_large.solve_explicit()
solver_implicit_small.solve_implicit()
solver_implicit_large.solve_implicit()

times_immediately_after = [0.1, 0.2, 0.3, 0.4, 0.5]
times_critical_check = [1, 2, 3, 4]
times_fine_grained_check = [1.8, 1.9, 2, 2.1, 2.2]
time_start_end = [0.1, 1, 2, 3, 4]
times_to_plot = times_immediately_after + times_critical_check + times_fine_grained_check

# Plot solutions
solver_explicit_small.plot_solution(times=time_start_end, title='Explicit Method with Small dt')
solver_explicit_large.plot_solution(times=time_start_end, title='Explicit Method with Large dt')
solver_implicit_small.plot_solution(times=time_start_end, title='Implicit Method with Small dt')
solver_implicit_large.plot_solution(times=time_start_end, title='Implicit Method with Large dt')


# Task 2.3
def solve_and_check_temperature(alpha, material_name):
    """
    Solve the heat equation for a given material and check if the temperature exceeds 172 C.

    Args:
    alpha (float): Thermal diffusivity of the material.
    material_name (str): Name of the material.

    Returns:
    None
    """
    # Set parameters
    dx = 0.1  # spatial step
    dt = 0.001  # time step, determined based on stability criteria
    theta = 0.5  # Crank-Nicolson method parameter

    # Create solver instance
    solver = SolverHeatXT(xlim, tlim, dx, dt, alpha, theta, x0, x1, t0)

    # Solve using Crank-Nicolson method (could choose explicit or implicit as well)
    solver.solve_implicit()  # or solver.solve_explicit()

    # Check the temperature at the middle of the rod across all time steps
    mid_point_index = solver.nx // 2  # Index corresponding to the middle of the rod
    temperature_mid_rod = solver.solution[:, mid_point_index]

    # Check if the temperature exceeded 172 C at any time at the middle of the rod
    if np.any(temperature_mid_rod > 172):
        print(f"The whole section of rod made of {material_name} exceeded 172 C after four seconds in time.")
    else:
        print(f"The whole section of rod made of {material_name} DID NOT exceeded 172 C after four seconds in time.")


# Constants for the materials (thermal diffusivity)
alphas = {
    "silver": 1.5,
    "copper": 1.25,
    "aluminium": 1.0
}

# Solve for each material and check the final temperature
for material, alpha in alphas.items():
    solve_and_check_temperature(alpha, material)