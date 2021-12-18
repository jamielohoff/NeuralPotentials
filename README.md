# NeuralPotentials

This repository contains the work I did on the topic on neural potentials. It contains a Jupyter notebook with an introduction on neural potentials using harmonic oscillators as well as all the code that was used to analyze various astrophysical and cosmological problems. The repository also contains the Master's thesis on which the project is based. THe thesis serves as a reference and handbook for the project.

## Setup of the Project

ALL the code is written in Julia, therefore it is imperative to install it on your machine. Follow the instructions under https://julialang.org/downloads/platform/ to install the latest version on Windows/Linux/Mac. The package was tested with Julia version 1.6.1.
The program requires several packages from the Julia package repository. To learn how to install packages, look at https://docs.julialang.org/en/v1/stdlib/Pkg/.
The relevant Julia packages are:
* DifferentialEquations
* Flux
* FluxDiffEq
* Zygote
* CSV
* Plots
* LateXStrings
* Measures
* DataFrames

To use the Jupyter notebook, it is necessary to have a Julia version and packages listed above installed on your system. To understand how to deploy a Julia notebook on Jupyter, look at https://datatofish.com/add-julia-to-jupyter/. 

## The Notebook

Run the notebook by cloning the package and navigating to the root folder of the package. Then run the command `jupyter notebook` and a directory will open in your browser. Select the notebook and run it. It gives an overview about the method of neural potentials, its advantages and disadvantages and how to implement it in Julia.

## Harmonic Oscillator

The notebook teaches your the basics about neural potentials using the example of a simple harmonic oscillator. The folder called `HarmonicOscillator` containes further exmpales where techniques like bootstrap-sampling are introduced and more elaborate systems like a double-well potential are analyzed.

## Sagittarius A* - The Milkyway's Supermassive Black Hole

The folder named `KeplerianOrbits` contains the example usecase of analyzing the infrared data of the positions of S-stars around Sagittarius A* recorded by the GRAVITY collaboration. It is sligthly more elaborate than the harmonic oscillator but still pretty simple as we again work in the Newtonian framework where we try to learn Newton's gravitational law and its relativistic corrections. The datasets can be obtained from https://vizier.cds.unistra.fr/viz-bin/VizieR-3?-source=J/ApJ/837/30/table5

## Redshift-Luminosity Data of Supernovae Ia

## Project Status

The project is no longer maintained.

