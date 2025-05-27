# mobius-strip-project
Python project modeling a Möbius strip with parametric equations, calculating surface area and edge length, and visualizing the 3D shape.

## How It Works

- The Möbius strip is defined by parameters: radius (R), width (w), and resolution (n).
- Uses parametric equations to create a 3D mesh of points on the surface.
- Surface area is approximated by numerically integrating the magnitude of the cross product of partial derivatives.
- Edge length is calculated by summing distances along the two edges of the strip.
- The strip is visualized using Matplotlib and saved as an image file.

## Requirements

- Python 3
- NumPy
- SciPy
- Matplotlib

Install dependencies with:

```bash
pip install numpy scipy matplotlib
