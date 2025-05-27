import numpy as np
from scipy.integrate import simpson  # Use 'simpson' as per your SciPy version
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting

class MobiusStrip:
    def __init__(self, R, w, n):
        """
        Initialize the Mobius Strip parameters.

        Parameters:
        R (float): Radius from the center to the strip.
        w (float): Width of the strip.
        n (int): Resolution for the mesh/grid.
        """
        self.R = R
        self.w = w
        self.n = n
        
        # Create parametric grid for parameters u and v
        self.u = np.linspace(0, 2 * np.pi, n)
        self.v = np.linspace(-w/2, w/2, n)
        self.U, self.V = np.meshgrid(self.u, self.v)

        # Compute the 3D mesh points using the parametric equations
        self.X = (R + self.V * np.cos(self.U / 2)) * np.cos(self.U)
        self.Y = (R + self.V * np.cos(self.U / 2)) * np.sin(self.U)
        self.Z = self.V * np.sin(self.U / 2)

    def surface_area(self):
        """
        Numerically approximate the surface area of the Möbius strip.

        Uses the formula for surface area integral over parameters (u,v):
        Area = ∬ ||cross(∂r/∂u, ∂r/∂v)|| dudv

        Returns:
        float: Approximated surface area.
        """
        # Calculate partial derivatives numerically
        du = self.u[1] - self.u[0]
        dv = self.v[1] - self.v[0]

        # Partial derivatives with respect to u
        dX_du = np.gradient(self.X, du, axis=1)
        dY_du = np.gradient(self.Y, du, axis=1)
        dZ_du = np.gradient(self.Z, du, axis=1)

        # Partial derivatives with respect to v
        dX_dv = np.gradient(self.X, dv, axis=0)
        dY_dv = np.gradient(self.Y, dv, axis=0)
        dZ_dv = np.gradient(self.Z, dv, axis=0)

        # Cross product of partial derivatives gives local surface normal vector magnitude
        cross_x = dY_du * dZ_dv - dZ_du * dY_dv
        cross_y = dZ_du * dX_dv - dX_du * dZ_dv
        cross_z = dX_du * dY_dv - dY_du * dX_dv

        integrand = np.sqrt(cross_x**2 + cross_y**2 + cross_z**2)

        # Integrate over v, then u using Simpson's rule
        area_v = simpson(integrand, self.v, axis=0)
        total_area = simpson(area_v, self.u)
        return total_area

    def edge_length(self):
        """
        Approximate the length of the boundary edge of the Möbius strip.

        The boundary is along v = ±w/2 for u in [0, 2π].

        Returns:
        float: Approximated edge length.
        """
        # Edge at v = +w/2
        x_edge1 = (self.R + (self.w/2) * np.cos(self.u / 2)) * np.cos(self.u)
        y_edge1 = (self.R + (self.w/2) * np.cos(self.u / 2)) * np.sin(self.u)
        z_edge1 = (self.w/2) * np.sin(self.u / 2)

        # Edge at v = -w/2
        x_edge2 = (self.R - (self.w/2) * np.cos(self.u / 2)) * np.cos(self.u)
        y_edge2 = (self.R - (self.w/2) * np.cos(self.u / 2)) * np.sin(self.u)
        z_edge2 = -(self.w/2) * np.sin(self.u / 2)

        # Calculate length of each edge curve using Simpson's rule on the distance between consecutive points
        def curve_length(x, y, z):
            dx = np.gradient(x, self.u)
            dy = np.gradient(y, self.u)
            dz = np.gradient(z, self.u)
            ds = np.sqrt(dx**2 + dy**2 + dz**2)
            return simpson(ds, self.u)

        length1 = curve_length(x_edge1, y_edge1, z_edge1)
        length2 = curve_length(x_edge2, y_edge2, z_edge2)

        # Möbius strip has one edge, so the total edge length is sum of both edges
        return length1 + length2

    def plot(self, filename=None):
        """
        Plot the Möbius strip in 3D.

        Parameters:
        filename (str or None): If provided, save the plot to this filename.
                                Otherwise, display the plot interactively.
        """
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        # Plot the surface mesh
        ax.plot_surface(self.X, self.Y, self.Z, cmap='viridis', edgecolor='k', alpha=0.8)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Möbius Strip')

        if filename:
            plt.savefig(filename)
            print(f"Plot saved as '{filename}'")
        else:
            plt.show()


if __name__ == "__main__":
    # Parameters
    R = 5      # Radius of the center circle
    w = 2      # Width of the strip
    n = 100    # Resolution (number of points)

    # Create Mobius strip object
    mobius = MobiusStrip(R, w, n)

    # Compute and print surface area and edge length
    area = mobius.surface_area()
    edge_len = mobius.edge_length()
    print(f"Surface Area: {area}")
    print(f"Edge Length: {edge_len}")

    # Plot and save the figure as PNG
    mobius.plot("mobius_strip.png")
