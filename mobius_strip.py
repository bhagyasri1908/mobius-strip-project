import numpy as np
from scipy.integrate import simpson
import matplotlib.pyplot as plt


class MobiusStrip:
    """
    Class to model a Möbius strip using parametric equations,
    compute its surface mesh, surface area, and edge length,
    and visualize the strip.
    """

    def __init__(self, R=1.0, w=0.3, n=100):
        """
        Initialize the Möbius strip parameters and compute the mesh.

        Parameters:
            R (float): Radius of the Möbius strip center circle.
            w (float): Width of the Möbius strip.
            n (int): Number of points along each parameter (resolution).
        """
        self.R = R
        self.w = w
        self.n = n

        # Parameter grids
        self.u = np.linspace(0, 2 * np.pi, self.n)
        self.v = np.linspace(-self.w / 2, self.w / 2, self.n)
        self.U, self.V = np.meshgrid(self.u, self.v)

        # Compute 3D coordinates on the Möbius strip surface
        self.X, self.Y, self.Z = self._compute_coordinates()

    def _compute_coordinates(self):
        """
        Compute the (x, y, z) coordinates for each (u, v) on the Möbius strip.

        Returns:
            X, Y, Z (np.ndarray): Meshgrid arrays representing 3D surface points.
        """
        # Parametric equations:
        # x(u,v) = (R + v*cos(u/2)) * cos(u)
        # y(u,v) = (R + v*cos(u/2)) * sin(u)
        # z(u,v) = v * sin(u/2)
        half_u = self.U / 2
        x = (self.R + self.V * np.cos(half_u)) * np.cos(self.U)
        y = (self.R + self.V * np.cos(half_u)) * np.sin(self.U)
        z = self.V * np.sin(half_u)
        return x, y, z

    def surface_area(self):
        """
        Numerically approximate the surface area of the Möbius strip
        using the parametric surface integral formula.

        Returns:
            float: Estimated surface area.
        """
        # Calculate partial derivatives
        du_x = np.gradient(self.X, self.u, axis=1)
        du_y = np.gradient(self.Y, self.u, axis=1)
        du_z = np.gradient(self.Z, self.u, axis=1)

        dv_x = np.gradient(self.X, self.v, axis=0)
        dv_y = np.gradient(self.Y, self.v, axis=0)
        dv_z = np.gradient(self.Z, self.v, axis=0)

        # Compute cross product of the partial derivatives (Jacobian)
        cross_x = du_y * dv_z - du_z * dv_y
        cross_y = du_z * dv_x - du_x * dv_z
        cross_z = du_x * dv_y - du_y * dv_x

        # Surface element magnitude
        dS = np.sqrt(cross_x**2 + cross_y**2 + cross_z**2)

        # Numerically integrate over v and u to get total surface area
        area = simpson(simpson(dS, self.v), self.u)
        return area

    def edge_length(self):
        """
        Numerically compute the total length of the Möbius strip's boundary edges.

        Returns:
            float: Estimated edge length.
        """
        # Extract edge curves at v = -w/2 and v = w/2
        edge1 = np.array([self.X[0, :], self.Y[0, :], self.Z[0, :]]).T
        edge2 = np.array([self.X[-1, :], self.Y[-1, :], self.Z[-1, :]]).T

        def curve_length(points):
            # Sum of distances between consecutive points along the edge
            diffs = np.diff(points, axis=0)
            segment_lengths = np.linalg.norm(diffs, axis=1)
            return np.sum(segment_lengths)

        length1 = curve_length(edge1)
        length2 = curve_length(edge2)
        return length1 + length2

    def plot(self, filename=None):
        """
        Visualize the Möbius strip surface in 3D.

        Parameters:
            filename (str, optional): If provided, save the plot to this file.
        """
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        # Plot surface with a colormap for clarity
        ax.plot_surface(self.X, self.Y, self.Z, rstride=1, cstride=1, cmap='viridis', edgecolor='k', alpha=0.8)

        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Möbius Strip')

        # Equal aspect ratio for all axes
        max_range = np.array([self.X.max()-self.X.min(), self.Y.max()-self.Y.min(), self.Z.max()-self.Z.min()]).max() / 2.0
        mid_x = (self.X.max() + self.X.min()) * 0.5
        mid_y = (self.Y.max() + self.Y.min()) * 0.5
        mid_z = (self.Z.max() + self.Z.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        plt.tight_layout()

        if filename:
            plt.savefig(filename, dpi=300)
            print(f"Plot saved as '{filename}'")
        else:
            plt.show()


if __name__ == "__main__":
    # Create Möbius strip with radius=1, width=0.3, resolution=200 (high-res for accuracy)
    mobius = MobiusStrip(R=1.0, w=0.3, n=200)

    # Compute and print surface area and edge length
    area = mobius.surface_area()
    edge_len = mobius.edge_length()
    print(f"Surface Area: {area:.6f}")
    print(f"Edge Length: {edge_len:.6f}")

    # Plot and save the visualization
    mobius.plot(filename="mobius_strip.png")

