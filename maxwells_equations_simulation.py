#!/usr/bin/env python3
"""
Maxwell's Equations Simulation using FDTD (Finite Difference Time Domain) Method

This script simulates electromagnetic wave propagation in 2D space
using Maxwell's equations in their differential form.

Maxwell's Equations (in differential form):
1. ∇·E = ρ/ε₀ (Gauss's Law)
2. ∇·B = 0 (Gauss's Law for Magnetism)
3. ∇×E = -∂B/∂t (Faraday's Law)
4. ∇×B = μ₀J + μ₀ε₀∂E/∂t (Ampère-Maxwell Law)

Author: Maxwell's Equations Simulator
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')


class MaxwellSimulator:
    """
    2D FDTD Simulator for Maxwell's Equations
    
    Simulates TE (Transverse Electric) or TM (Transverse Magnetic) modes
    in a 2D grid with optional sources and boundaries.
    """
    
    def __init__(self, nx=100, ny=100, dx=0.01, dy=0.01, dt=None, 
                 epsilon_r=1.0, mu_r=1.0, conductivity=0.0):
        """
        Initialize the simulation grid and parameters.
        
        Parameters:
        -----------
        nx, ny : int
            Number of grid points in x and y directions
        dx, dy : float
            Grid spacing in meters
        dt : float, optional
            Time step (if None, calculated from CFL condition)
        epsilon_r : float
            Relative permittivity of the medium
        mu_r : float
            Relative permeability of the medium
        conductivity : float
            Electrical conductivity (S/m)
        """
        # Grid dimensions
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        
        # Physical constants
        self.epsilon_0 = 8.854187817e-12  # Vacuum permittivity (F/m)
        self.mu_0 = 4 * np.pi * 1e-7       # Vacuum permeability (H/m)
        self.c = 299792458                  # Speed of light (m/s)
        
        # Material properties
        self.epsilon = self.epsilon_0 * epsilon_r
        self.mu = self.mu_0 * mu_r
        self.sigma = conductivity
        
        # Time step (CFL condition for stability)
        if dt is None:
            self.dt = 0.99 / (self.c * np.sqrt((1/dx)**2 + (1/dy)**2))
        else:
            self.dt = dt
        
        # Initialize field components (2D TE mode: Ez, Hx, Hy)
        self.Ez = np.zeros((nx, ny))      # Electric field (z-component)
        self.Hx = np.zeros((nx, ny))      # Magnetic field (x-component)
        self.Hy = np.zeros((nx, ny))      # Magnetic field (y-component)
        
        # Field history for visualization
        self.Ez_history = []
        
        # Precompute coefficients for update equations
        self._compute_coefficients()
        
        # Sources
        self.sources = []
        
    def _compute_coefficients(self):
        """Precompute update coefficients for efficiency."""
        # For electric field update
        if self.sigma == 0:
            self.CEz = np.ones((self.nx, self.ny))
            self.CEz_cond = np.zeros((self.nx, self.ny))
        else:
            self.CEz = (1 - self.sigma * self.dt / (2 * self.epsilon)) / \
                       (1 + self.sigma * self.dt / (2 * self.epsilon))
            self.CEz_cond = (self.dt / self.epsilon) / \
                           (1 + self.sigma * self.dt / (2 * self.epsilon))
        
        # For magnetic field updates
        self.CHx = self.dt / (self.mu * self.dy)
        self.CHy = self.dt / (self.mu * self.dx)
        
    def add_source(self, source_type='gaussian', position=None, 
                   frequency=None, amplitude=1.0, polarized='Ez'):
        """
        Add a source to the simulation.
        
        Parameters:
        -----------
        source_type : str
            Type of source: 'gaussian', 'sinusoidal', 'pulse'
        position : tuple
            (x, y) position of the source in grid coordinates
        frequency : float
            Frequency for sinusoidal sources (Hz)
        amplitude : float
            Source amplitude
        polarized : str
            Field component to excite: 'Ez', 'Hx', 'Hy'
        """
        if position is None:
            position = (self.nx // 2, self.ny // 2)
        
        source = {
            'type': source_type,
            'position': position,
            'frequency': frequency,
            'amplitude': amplitude,
            'polarized': polarized,
            'time_step': 0
        }
        self.sources.append(source)
        
    def _update_sources(self, time_step):
        """Update source values at current time step."""
        for source in self.sources:
            source['time_step'] = time_step
            ix, iy = source['position']
            
            if source['type'] == 'gaussian':
                # Gaussian pulse
                t0 = 30 / (self.c * np.sqrt((1/self.dx)**2 + (1/self.dy)**2))
                value = source['amplitude'] * np.exp(-((time_step - t0)**2) / (2 * (t0/3)**2))
            elif source['type'] == 'sinusoidal':
                # Continuous sinusoidal wave
                if source['frequency'] is None:
                    source['frequency'] = self.c / (10 * self.dx)
                omega = 2 * np.pi * source['frequency']
                value = source['amplitude'] * np.sin(omega * time_step * self.dt)
            elif source['type'] == 'pulse':
                # Rectangular pulse
                duration = int(20 / (self.c * np.sqrt((1/self.dx)**2 + (1/self.dy)**2)))
                value = source['amplitude'] if time_step < duration else 0
            else:
                value = source['amplitude']
            
            # Apply source to appropriate field component
            if source['polarized'] == 'Ez':
                if 0 <= ix < self.nx and 0 <= iy < self.ny:
                    self.Ez[ix, iy] += value
            elif source['polarized'] == 'Hx':
                if 0 <= ix < self.nx and 0 <= iy < self.ny:
                    self.Hx[ix, iy] += value
            elif source['polarized'] == 'Hy':
                if 0 <= ix < self.nx and 0 <= iy < self.ny:
                    self.Hy[ix, iy] += value
                    
    def _apply_boundary_conditions(self):
        """Apply Perfectly Matched Layer (PML) or absorbing boundary conditions."""
        # Simple absorbing boundary conditions (Mur's first-order ABC)
        # Update boundaries to absorb outgoing waves
        
        # Left and right boundaries (x-direction)
        if self.nx > 2:
            self.Ez[1, :] = self.Ez[2, :] + (self.c * self.dt - self.dx) / (self.c * self.dt + self.dx) * \
                           (self.Ez[1, :] - self.Ez_old_left if hasattr(self, 'Ez_old_left') else 0)
            self.Ez[-2, :] = self.Ez[-3, :] + (self.c * self.dt - self.dx) / (self.c * self.dt + self.dx) * \
                            (self.Ez[-2, :] - self.Ez_old_right if hasattr(self, 'Ez_old_right') else 0)
        
        # Top and bottom boundaries (y-direction)
        if self.ny > 2:
            self.Ez[:, 1] = self.Ez[:, 2] + (self.c * self.dt - self.dy) / (self.c * self.dt + self.dy) * \
                           (self.Ez[:, 1] - self.Ez_old_bottom if hasattr(self, 'Ez_old_bottom') else 0)
            self.Ez[:, -2] = self.Ez[:, -3] + (self.c * self.dt - self.dy) / (self.c * self.dt + self.dy) * \
                            (self.Ez[:, -2] - self.Ez_old_top if hasattr(self, 'Ez_old_top') else 0)
        
        # Store boundary values for next iteration
        self.Ez_old_left = self.Ez[1, :].copy()
        self.Ez_old_right = self.Ez[-2, :].copy()
        self.Ez_old_bottom = self.Ez[:, 1].copy()
        self.Ez_old_top = self.Ez[:, -2].copy()
        
    def step(self):
        """
        Perform one time step of the FDTD simulation.
        
        Updates all field components using Yee's algorithm.
        """
        # Update sources
        self._update_sources(len(self.Ez_history))
        
        # Update magnetic fields (Hx and Hy) from curl of E
        # Faraday's Law: ∇×E = -∂B/∂t
        
        # Hx update (depends on dEz/dy)
        self.Hx[:, :-1] -= self.CHx * (self.Ez[:, 1:] - self.Ez[:, :-1])
        
        # Hy update (depends on dEz/dx)
        self.Hy[:-1, :] += self.CHy * (self.Ez[1:, :] - self.Ez[:-1, :])
        
        # Update electric field (Ez) from curl of H
        # Ampère-Maxwell Law: ∇×B = μ₀J + μ₀ε₀∂E/∂t
        
        # Ez update (depends on dHy/dx - dHx/dy)
        # Using Yee grid staggered formulation
        # dHy/dx at Ez location: (Hy[i+1,j] - Hy[i,j]) / dx where Hy is at (i+1/2, j)
        # dHx/dy at Ez location: (Hx[i,j+1] - Hx[i,j]) / dy where Hx is at (i, j+1/2)
        
        # For interior points Ez[1:-1, 1:-1], compute the curl properly
        curl_H = np.zeros((self.nx, self.ny))
        curl_H[1:-1, 1:-1] = ((self.Hy[1:-1, 1:-1] - self.Hy[0:-2, 1:-1]) / self.dx -
                              (self.Hx[1:-1, 1:-1] - self.Hx[1:-1, 0:-2]) / self.dy)
        
        self.Ez[1:-1, 1:-1] = (self.CEz[1:-1, 1:-1] * self.Ez[1:-1, 1:-1] +
                               self.CEz_cond[1:-1, 1:-1] * curl_H[1:-1, 1:-1])
        
        # Apply boundary conditions
        self._apply_boundary_conditions()
        
        # Store field history
        self.Ez_history.append(self.Ez.copy())
        
    def run(self, num_steps=200, verbose=True):
        """
        Run the simulation for a specified number of time steps.
        
        Parameters:
        -----------
        num_steps : int
            Number of time steps to simulate
        verbose : bool
            Print progress information
        """
        if verbose:
            print(f"Starting Maxwell's Equations Simulation")
            print(f"Grid size: {self.nx} × {self.ny}")
            print(f"Time step: {self.dt:.2e} s")
            print(f"Running for {num_steps} time steps...")
        
        for step in range(num_steps):
            self.step()
            if verbose and step % (num_steps // 10) == 0:
                print(f"Progress: {step}/{num_steps} ({100*step//num_steps}%)")
        
        if verbose:
            print("Simulation complete!")
            
    def visualize_field(self, field_component='Ez', step_idx=-1, save_path=None):
        """
        Visualize the electromagnetic field at a specific time step.
        
        Parameters:
        -----------
        field_component : str
            Field component to visualize: 'Ez', 'Hx', 'Hy'
        step_idx : int
            Index of time step to visualize (-1 for last step)
        save_path : str, optional
            Path to save the figure
        """
        if step_idx == -1:
            step_idx = len(self.Ez_history) - 1
            
        if step_idx >= len(self.Ez_history):
            print(f"Error: step_idx {step_idx} out of range")
            return
            
        field = self.Ez_history[step_idx]
        
        plt.figure(figsize=(10, 8))
        im = plt.imshow(field.T, origin='lower', cmap='RdBu_r', 
                       extent=[0, self.nx*self.dx, 0, self.ny*self.dy],
                       aspect='auto')
        plt.colorbar(im, label=f'{field_component} Field Strength (V/m)')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.title(f"{field_component} Field at Time Step {step_idx}\n(t = {step_idx*self.dt:.2e} s)")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        plt.show()
        
    def animate(self, field_component='Ez', interval=50, save_path=None):
        """
        Create an animation of the field evolution.
        
        Parameters:
        -----------
        field_component : str
            Field component to animate
        interval : int
            Delay between frames in milliseconds
        save_path : str, optional
            Path to save the animation (requires ffmpeg)
        """
        if len(self.Ez_history) == 0:
            print("No simulation data. Run simulation first.")
            return
            
        fig, ax = plt.subplots(figsize=(10, 8))
        
        def update(frame):
            ax.clear()
            field = self.Ez_history[frame]
            im = ax.imshow(field.T, origin='lower', cmap='RdBu_r',
                          extent=[0, self.nx*self.dx, 0, self.ny*self.dy],
                          aspect='auto', vmin=-np.max(np.abs(field)), 
                          vmax=np.max(np.abs(field)))
            ax.set_xlabel('x (m)')
            ax.set_ylabel('y (m)')
            ax.set_title(f"{field_component} Field - Time Step {frame}\n(t = {frame*self.dt:.2e} s)")
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label(f'{field_component} Field Strength (V/m)')
            return [im]
        
        anim = FuncAnimation(fig, update, frames=len(self.Ez_history),
                           interval=interval, blit=True)
        
        if save_path:
            anim.save(save_path, writer='pillow', fps=20)
            print(f"Animation saved to {save_path}")
        
        plt.show()
        return anim


def demonstrate_plane_wave():
    """Demonstrate plane wave propagation."""
    print("\n" + "="*60)
    print("DEMONSTRATION 1: Plane Wave Propagation")
    print("="*60)
    
    # Create simulator with smaller grid for faster execution
    sim = MaxwellSimulator(nx=100, ny=60, dx=0.01, dy=0.01)
    
    # Add a line source on the left side (plane wave)
    for y in range(15, 45):
        sim.add_source(source_type='sinusoidal', 
                      position=(5, y),
                      frequency=1e8,  # 100 MHz
                      amplitude=1.0,
                      polarized='Ez')
    
    # Run simulation
    sim.run(num_steps=100, verbose=True)
    
    # Visualize
    sim.visualize_field(field_component='Ez', step_idx=-1, 
                       save_path='plane_wave_result.png')
    
    return sim


def demonstrate_point_source():
    """Demonstrate circular wave from point source."""
    print("\n" + "="*60)
    print("DEMONSTRATION 2: Point Source (Circular Waves)")
    print("="*60)
    
    # Create simulator with smaller grid
    sim = MaxwellSimulator(nx=120, ny=120, dx=0.01, dy=0.01)
    
    # Add point source in center
    sim.add_source(source_type='sinusoidal',
                  position=(60, 60),
                  frequency=1.5e8,  # 150 MHz
                  amplitude=2.0,
                  polarized='Ez')
    
    # Run simulation
    sim.run(num_steps=120, verbose=True)
    
    # Visualize
    sim.visualize_field(field_component='Ez', step_idx=-1,
                       save_path='point_source_result.png')
    
    return sim


def demonstrate_pulse_propagation():
    """Demonstrate Gaussian pulse propagation."""
    print("\n" + "="*60)
    print("DEMONSTRATION 3: Gaussian Pulse Propagation")
    print("="*60)
    
    # Create simulator with smaller grid
    sim = MaxwellSimulator(nx=100, ny=100, dx=0.01, dy=0.01)
    
    # Add Gaussian pulse source
    sim.add_source(source_type='gaussian',
                  position=(50, 50),
                  amplitude=5.0,
                  polarized='Ez')
    
    # Run simulation
    sim.run(num_steps=100, verbose=True)
    
    # Visualize final state
    sim.visualize_field(field_component='Ez', step_idx=-1,
                       save_path='pulse_result.png')
    
    # Skip animation for faster execution (optional)
    print("\nSkipping animation to save time...")
    # To create animation, uncomment the following lines:
    # print("\nCreating animation...")
    # try:
    #     sim.animate(field_component='Ez', interval=100, 
    #                save_path='pulse_animation.gif')
    # except Exception as e:
    #     print(f"Could not create animation: {e}")
    
    return sim


def demonstrate_dielectric_interface():
    """Demonstrate wave propagation across dielectric interface."""
    print("\n" + "="*60)
    print("DEMONSTRATION 4: Dielectric Interface (Refraction)")
    print("="*60)
    
    # Create simulator with different materials (smaller grid)
    sim = MaxwellSimulator(nx=120, ny=60, dx=0.01, dy=0.01,
                          epsilon_r=1.0)  # Start with vacuum
    
    # Modify permittivity for right half (dielectric material)
    epsilon_r_map = np.ones((sim.nx, sim.ny))
    epsilon_r_map[sim.nx//2:, :] = 4.0  # εr = 4 for right half
    
    sim.epsilon = sim.epsilon_0 * epsilon_r_map
    sim._compute_coefficients()
    
    # Add source on left side
    for y in range(20, 40):
        sim.add_source(source_type='sinusoidal',
                      position=(10, y),
                      frequency=1e8,
                      amplitude=1.0,
                      polarized='Ez')
    
    # Run simulation
    sim.run(num_steps=150, verbose=True)
    
    # Visualize
    sim.visualize_field(field_component='Ez', step_idx=-1,
                       save_path='dielectric_interface_result.png')
    
    return sim


def plot_maxwell_equations():
    """Create educational visualization of Maxwell's equations."""
    print("\n" + "="*60)
    print("EDUCATIONAL: Maxwell's Equations Visualization")
    print("="*60)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("Maxwell's Equations - Fundamental Laws of Electromagnetism", 
                fontsize=16, fontweight='bold')
    
    equations = [
        ("Gauss's Law", r"$\nabla \cdot \mathbf{E} = \frac{\rho}{\varepsilon_0}$",
         "Electric charges produce electric fields"),
        ("Gauss's Law for Magnetism", r"$\nabla \cdot \mathbf{B} = 0$",
         "No magnetic monopoles exist"),
        ("Faraday's Law", r"$\nabla \times \mathbf{E} = -\frac{\partial \mathbf{B}}{\partial t}$",
         "Changing magnetic fields induce electric fields"),
        ("Ampère-Maxwell Law", r"$\nabla \times \mathbf{B} = \mu_0\mathbf{J} + \mu_0\varepsilon_0\frac{\partial \mathbf{E}}{\partial t}$",
         "Currents and changing electric fields produce magnetic fields")
    ]
    
    for idx, (title, equation, description) in enumerate(equations):
        ax = axes[idx // 2, idx % 2]
        ax.text(0.5, 0.7, equation, ha='center', va='center', 
               fontsize=18, family='serif', transform=ax.transAxes)
        ax.text(0.5, 0.3, title, ha='center', va='center',
               fontsize=14, fontweight='bold', transform=ax.transAxes)
        ax.text(0.5, 0.1, description, ha='center', va='center',
               fontsize=11, style='italic', wrap=True, transform=ax.transAxes)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('maxwell_equations_summary.png', dpi=150, bbox_inches='tight')
    print("Educational summary saved to maxwell_equations_summary.png")
    plt.show()


def main():
    """Main function to run demonstrations."""
    print("\n" + "="*60)
    print("MAXWELL'S EQUATIONS SIMULATION")
    print("Using Finite Difference Time Domain (FDTD) Method")
    print("="*60)
    
    # Show educational summary
    plot_maxwell_equations()
    
    # Run demonstrations
    sim1 = demonstrate_plane_wave()
    sim2 = demonstrate_point_source()
    sim3 = demonstrate_pulse_propagation()
    sim4 = demonstrate_dielectric_interface()
    
    print("\n" + "="*60)
    print("All demonstrations complete!")
    print("Generated files:")
    print("  - plane_wave_result.png")
    print("  - point_source_result.png")
    print("  - pulse_result.png")
    print("  - pulse_animation.gif (if successful)")
    print("  - dielectric_interface_result.png")
    print("  - maxwell_equations_summary.png")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
