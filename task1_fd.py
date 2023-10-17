# import statements
from functions_fd import *


# define the Poisson function
def poisson(x, y):
    return x - y


def main():
    # set up the x and y dimensions.
    xlim = np.array([-2., 2.])
    ylim = np.array([-3., 3.])

    # set up the boundary conditions
    bc_x0 = {'type': 'neumann', 'function': lambda x, y: x}
    bc_x1 = {'type': 'neumann', 'function': lambda x, y: y}
    bc_y0 = {'type': 'dirichlet', 'function': lambda x, y: x * y}
    bc_y1 = {'type': 'dirichlet', 'function': lambda x, y: x * y - 1.}

    # Set mesh spacings
    deltas = [2, 0.1]  # These are your delta_a and delta_b

    # Create figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    for i, delta in enumerate(deltas):
        # Create a solver instance with the current mesh spacing
        solver = SolverPoissonXY(xlim, ylim, delta, bc_x0, bc_x1, bc_y0, bc_y1, poisson)
        solver.solve()

        # Generate contour plot
        X, Y = np.meshgrid(solver.x, solver.y)
        cp = axs[i].contourf(X, Y, solver.solution, cmap='viridis')
        fig.colorbar(cp, ax=axs[i])  # Add a colorbar to a plot

        axs[i].set_title(f'Solution for mesh spacing = {delta}')
        axs[i].set_xlabel('x')
        axs[i].set_ylabel('y')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
