# BasicPhysics

All about Physic

## Features:

FDTD (Finite Difference Time Domain) Method: Implements Yee's algorithm to solve Maxwell's equations numerically in 2D
All Four Maxwell's Equations implemented:
Gauss's Law: ∇·E = ρ/ε₀
Gauss's Law for Magnetism: ∇·B = 0
Faraday's Law: ∇×E = -∂B/∂t
Ampère-Maxwell Law: ∇×B = μ₀J + μ₀ε₀∂E/∂t

## Key Components:

MaxwellSimulator Class: Main simulation engine with configurable grid size, material properties, and time steps
Multiple Source Types: Gaussian pulses, sinusoidal waves, and rectangular pulses
Absorbing Boundary Conditions: Mur's first-order ABC to minimize reflections
Visualization Tools: Static field plots and animation capabilities
Four Demonstrations Included:
Plane Wave Propagation - Linear array of sources creating planar wavefronts
Point Source - Circular waves radiating from a central point
Gaussian Pulse - Short electromagnetic pulse propagation
Dielectric Interface - Wave refraction at material boundary (εr=1 to εr=4)

## Generated Output Files:
maxwell_equations_summary.png - Educational visualization of all four equations
plane_wave_result.png - Plane wave simulation result
point_source_result.png - Point source circular waves
pulse_result.png - Gaussian pulse propagation
dielectric_interface_result.png - Refraction at dielectric boundary
