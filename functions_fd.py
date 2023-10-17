# import statements
import numpy as np
import matplotlib.pyplot as plt


class SolverPoissonXY(object):
    """
    Class containing attributes and methods for solving the Poisson/Laplace PDE in 2D Cartesian coordinates. Assumes
    rectangular model domain and boundary conditions defined along those.

    Attributes:
        nx (int): number of mesh points along the x dimension.
        ny (int): number of mesh points along the y dimension.
        n (int): total number of mesh points.
        x (1D array): mesh coordinates along the x dimension.
        y (1D array): mesh coordinates along the y dimension.
        dx (float): mesh spacing along the x dimension. May differ from input delta.
        dy (float): mesh spacing along the y dimension. May differ from input delta.
        bc_x0 (dict): type and equation for left boundary u(x0,y).
        bc_x1 (dict): type and equation for right boundary u(x1,y).
        bc_y0 (dict): type and equation for bottom boundary u(x,y0).
        bc_y1 (dict): type and equation for top boundary u(x,y1).
        poisson (callable): Poisson function.
        a (2D array): coefficient matrix in system of equations to solve for PDE solution.
        b (1D array): vector of constants in system of equations to solve for PDE solution.
        solution (2D array): PDE solution array on the mesh.
    """

    def __init__(self, xlim, ylim, delta, bc_x0, bc_x1, bc_y0, bc_y1, poisson):
        """
        Arguments:
            xlim (1D array): lower and upper limits in x dimension.
            ylim (1D array): lower and upper limits in y dimension.
            delta (float): desired mesh spacing in x and y dimension i.e. assume uniform mesh spacing
            bc_x0 (dict): type and equation for left boundary u(x0,y).
            bc_x1 (dict): type and equation for right boundary u(x1,y).
            bc_y0 (dict): type and equation for bottom boundary u(x,y0).
            bc_y1 (dict): type and equation for top boundary u(x,y1).
            poisson (callable): Poisson function.
        """
        # Number of points along each dimension (considering boundary points)
        self.nx = int((xlim[1] - xlim[0]) / delta) + 1
        self.ny = int((ylim[1] - ylim[0]) / delta) + 1
        self.n = self.nx * self.ny

        # Coordinates of mesh points
        self.x = np.linspace(xlim[0], xlim[1], self.nx)
        self.y = np.linspace(ylim[0], ylim[1], self.ny)

        # Actual mesh spacing
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]

        # Initialize matrices and solution vector
        self.a = np.zeros((self.n, self.n))
        self.b = np.zeros(self.n)
        self.solution = np.zeros((self.ny, self.nx))

        # store the four boundary conditions
        self.bc_x0 = bc_x0
        self.bc_x1 = bc_x1
        self.bc_y0 = bc_y0
        self.bc_y1 = bc_y1

        # equation corresponding to forcing function
        self.poisson = poisson

    def dirichlet(self):
        """
        Update the corresponding elements of the A matrix and b vector for Dirichlet boundary mesh points.
        """
        # Left boundary (bc_x0):
        if self.bc_x0['type'] == 'dirichlet':
            for j in range(self.ny):
                index = j * self.nx  # Convert 2D index to 1D. This is row-major, so y changes fastest.
                self.a[index, index] = 1
                self.b[index] = self.bc_x0['function'](self.x[0], self.y[j])

        # Right boundary (bc_x1):
        if self.bc_x1['type'] == 'dirichlet':
            for j in range(self.ny):
                index = j * self.nx + (self.nx - 1)
                self.a[index, index] = 1
                self.b[index] = self.bc_x1['function'](self.x[-1], self.y[j])

        # Bottom boundary (bc_y0):
        if self.bc_y0['type'] == 'dirichlet':
            for i in range(self.nx):
                index = i  # The first row of the matrix.
                self.a[index, index] = 1
                self.b[index] = self.bc_y0['function'](self.x[i], self.y[0])

        # Top boundary (bc_y1):
        if self.bc_y1['type'] == 'dirichlet':
            for i in range(self.nx):
                index = (self.ny - 1) * self.nx + i
                self.a[index, index] = 1
                self.b[index] = self.bc_y1['function'](self.x[i], self.y[-1])

    def neumann(self):
        """
        Update the corresponding elements of the A matrix and b vector for Neumann boundary mesh points.
        """
        dx, dy = self.dx, self.dy

        # Left boundary (bc_x0):
        if self.bc_x0['type'] == 'neumann':
            for j in range(self.ny):
                index = j * self.nx
                self.a[index, index] = -1 / dx  # considering the normal is the x-axis
                self.a[index, index + 1] = 1 / dx
                self.b[index] = self.bc_x0['function'](self.x[0], self.y[j])

        # Right boundary (bc_x1):
        if self.bc_x1['type'] == 'neumann':
            for j in range(self.ny):
                index = j * self.nx + self.nx - 1
                self.a[index, index] = 1 / dx
                self.a[index, index - 1] = -1 / dx
                self.b[index] = self.bc_x1['function'](self.x[-1], self.y[j])

    def internal(self):
        """
        Update the corresponding elements of the A matrix and b vector for internal mesh points.
        """
        for j in range(1, self.ny - 1):
            for i in range(1, self.nx - 1):
                index = j * self.nx + i  # Convert 2D index to 1D. This is row-major, so y changes fastest.

                # Set up the five-point stencil in the A matrix
                self.a[index, index] = -4  # Central point
                self.a[index, index - 1] = 1  # Left point
                self.a[index, index + 1] = 1  # Right point
                self.a[index, index - self.nx] = 1  # Bottom point
                self.a[index, index + self.nx] = 1  # Top point

                # For the b vector, we use the Poisson function which gives us the values for the internal points.
                self.b[index] = -self.poisson(self.x[i], self.y[j]) * self.dx * self.dy

    def solve(self):
        """
        Call the dirichlet, neumann and internal methods to form the system of equations, Au=b. Then use built-in
        linear algebra solver to find u and reshape back to self.solution.
        """
        self.dirichlet()
        self.neumann()
        self.internal()

        # Solve the linear system
        u = np.linalg.solve(self.a, self.b)

        # Reshape the solution array to 2D
        self.solution = np.reshape(u, (self.ny, self.nx))

    def plot_solution(self):
        """
        Plot the PDE solution.
        """
        X, Y = np.meshgrid(self.x, self.y)
        plt.contourf(X, Y, self.solution, cmap='viridis')
        plt.colorbar()  # add color bar on the right
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Solution of the 2D Poisson equation')
        plt.show()


class SolverHeatXT(object):
    """
    Class containing attributes and methods for solving the 1D heat equation. Requires Dirichlet boundary conditions.

    Attributes:
        nx (int): number of mesh points along the spatial dimension.
        nt (int): number of mesh points along the time dimension.
        n (int): total number of mesh points.
        x (1D array): mesh coordinates along the x dimension.
        t (1D array): mesh coordinates along the t dimension.
        dx (float): mesh spacing along the x dimension.
        dt (float): mesh spacing along the t dimension.
        alpha (float): thermal diffusivity in the heat equation.
        r (float): equal to alpha*dt/(dx^2), useful for diagnosing numerical stability.
        theta (float): weight applied to spatial derivative at t^(n+1), where 0 < theta <= 1.
        bc_x0 (dict): dictionary storing information for left boundary conditions.
        bc_x1 (dict): dictionary storing information for right boundary conditions.
        ic_t0 (dict): dictionary storing information for initial conditions.
        solution (2D array): solution array corresponding to mesh dimensions (nx, nt).

    Arguments:
        xlim (1D array): lower and upper limits in x dimension.
        tlim (1D array): lower and upper limits in t dimension.
        dx (float): desired mesh spacing in x dimension. May not exactly equal set mesh spacing.
        dt (float): desired mesh spacing in t dimension. May not exactly equal set mesh spacing.
        bc_x0 (dict): boundary conditions along x0.
        bc_x1 (dict): boundary conditions along x1.
        ic_t0 (dict): initial conditions at t0.
        alpha (float): thermal diffusivity in the heat equation.
        theta (float): weight applied to spatial derivative at t^(n+1), where 0 < theta <= 1.
    """

    def __init__(self, xlim, tlim, dx, dt, alpha, theta, bc_x0, bc_x1, ic_t0):
        # Calculate the number of points in each dimension
        self.nx = int((xlim[1] - xlim[0]) / dx) + 1
        self.nt = int((tlim[1] - tlim[0]) / dt) + 1
        self.n = self.nx * self.nt

        # Generate the mesh in space and time
        self.x = np.linspace(xlim[0], xlim[1], self.nx)
        self.t = np.linspace(tlim[0], tlim[1], self.nt)

        # Adjust dx and dt based on the actual generated mesh
        self.dx = self.x[1] - self.x[0]
        self.dt = self.t[1] - self.t[0]

        # Miscellaneous initializations
        self.alpha = alpha
        self.r = self.alpha * self.dt / (self.dx ** 2)
        self.theta = theta

        # Initial and boundary conditions
        self.bc_x0 = bc_x0
        self.bc_x1 = bc_x1
        self.ic_t0 = ic_t0

        # Initialize the solution matrix and apply initial/boundary conditions
        self.solution = np.zeros((self.nt, self.nx))
        self.solution[0, :] = self.ic_t0['function'](self.x, self.t[0])  # Apply initial condition
        self.solution[:, 0] = self.bc_x0['function'](self.x, self.t)  # Apply boundary condition at x0
        self.solution[:, -1] = self.bc_x1['function'](self.x, self.t)  # Apply boundary condition at x1

    def solve_explicit(self):
        """
        Solve the 1D heat equation using an explicit solution method.
        """
        # Loop through time steps
        for n in range(0, self.nt - 1):
            # Loop through space
            for i in range(1, self.nx - 1):
                # Central difference in space, forward difference in time
                self.solution[n + 1, i] = self.r * self.solution[n, i - 1] + (1 - 2 * self.r) * self.solution[
                    n, i] + self.r * self.solution[n, i + 1]

    def implicit_update_a(self):
        """
        Set coefficients in the matrix A, prior to iterative solution. This only needs to be set once i.e. it doesn't
        change with each iteration, unlike the b vector.

        Returns:
            a (2D array): coefficient matrix for implicit method (dimension 2 n_x by 2 n_x)
        """
        # Initialize matrix A
        a = np.diag((1 + 2 * self.r) * np.ones(self.nx))
        np.fill_diagonal(a[1:], -self.r)
        np.fill_diagonal(a[:, 1:], -self.r)
        return a

    def implicit_update_b(self, indx_t):
        """
        Update the b vector for the current time step to be solved, making use of the

        Arguments:
            indx_t (int): time index for the current step being solved.

        Returns:
            b (1D array): vector of constants for implicit method (length of 2 n_x)
        """
        # Initialize vector b from the previous time step, considering boundary conditions
        b = self.solution[indx_t, :].copy()
        b[0] += self.r * self.bc_x0['function'](0, self.t[indx_t + 1])
        b[-1] += self.r * self.bc_x1['function'](0, self.t[indx_t + 1])
        return b

    def solve_implicit(self):
        """
        Solve the 1D heat equation using an implicit solution method.
        """
        # Create the coefficient matrix A
        A_matrix = self.implicit_update_a()

        # Loop through time steps
        for n in range(0, self.nt - 1):
            b_vec = self.implicit_update_b(n)
            # Solve the system of linear equations
            self.solution[n + 1, :] = np.linalg.solve(A_matrix, b_vec)

    def plot_solution(self, times=None, title='Heat Equation Solution', save_to_file=False, save_file_name='fig_HeatXT.png'):
        """
        Plot the solution as a 1D line plot of x(t) at regularly spaced specific times.

        Arguments:
            times (1D array): times (or time indexes) at which to plot the solution.
            save_to_file (boolean): if true, save the plot to a file with pre-determined name.
            save_file_name (string): name of figure to save if save_to_file is true.
            title (string): titles of graphs
        """
        plt.figure()
        for time in times:
            # Find the index of the closest time value
            idx = (np.abs(self.t - time)).argmin()

            # Now use this index to plot
            plt.plot(self.x, self.solution[idx, :], label=f'Time {self.t[idx]:.2f}')

        plt.title(title)
        plt.xlabel('Position along the rod (m)')
        plt.ylabel('Temperature (C)')
        plt.legend()
        plt.grid(True)

        if save_to_file:
            plt.savefig(save_file_name)
        else:
            plt.show()

