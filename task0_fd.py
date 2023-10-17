# import statements
from functions_fd import *


# define the Poisson function
def poisson(x, y):
    return 6. * x * y * (1. - y) - 2. * x * x * x


# set up the x and y dimensions, as well as desired mesh spacing.
xlim = np.array([0., 1.])
ylim = np.array([0., 1.])
delta = 1. / 10.

# set up the boundary conditions
bc_x0 = {'type': 'dirichlet', 'function': lambda x, y: np.cos(np.pi*y)}
bc_x1 = {'type': 'dirichlet', 'function': lambda x, y: 4.*np.cos(np.pi*y)}
bc_y0 = {'type': 'dirichlet', 'function': lambda x, y: (1.+x)*(1+x)}
bc_y1 = {'type': 'dirichlet', 'function': lambda x, y: -1.*(1.+x)*(1+x)}


# Create an object of SolverPoissonXY and then solve it
solver = SolverPoissonXY(xlim, ylim, delta, bc_x0, bc_x1, bc_y0, bc_y1, poisson)
solver.solve()
solver.plot_solution()