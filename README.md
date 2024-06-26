#  Assimilating mean flow data around an airfoil using Physics-informed Neural Networks
This repository contains the Python scripts used for the master thesis of Timo Vogel of TU Berlin. The thesis objective is to assimilate the flow field around a NACA-0012 airfoil from the sparse experimental dataset shown in this [figure](https://github.com/timovogel/PINN_NACA0012/blob/main/EXP.pdf).


## Abstract
The characterization of the turbulent flow around objects faces several challenges. Commonly used experimental or numerical techniques compromise on either accuracy or resolution. A recently proposed assimilation method, called physics-informed neural networks, integrates governing equations describing the flow into the training process of a deep neural network. The residuals of the model are added to the loss function defined for the network training. The network is thus optimized to predict the observed data and to obey the implemented equations. The result, unlike from numerical methods, is a continuous and therefore mesh-free prediction of the flow. 

In this work, a physics-informed neural network approach is used to assimilate the mean flow around a NACA-0012 airfoil from sparse experimental data. Velocity data is mainly available in the wake of the airfoil and pressure data is recorded on the airfoil surface. The Reynolds-Averaged Navier-Stokes equations are used as the governing equations for network training. The included stress tensor is modeled by the Spalart-Allmaras turbulence model by adding a transport equation of the introduced closure variable, the eddy viscosity.
Several sub-steps are taken towards this assimilation task to incrementally increase the complexity. In the beginning, dense numerical data from the RANS simulation is used and the assimilation domain is restricted to the airfoil wake. The sparsity of the numerical data is increased to match the experimental data locations. In the last step, the experimental data is used for the assimilation. This progression is repeated on a domain, which includes the airfoil.
 
The assimilated flow field shows an accurate prediction of the flow in large parts of the domain. The PINN is most accurate in the wake, where small velocity gradients are present. In the boundary layer and trailing edge regions, where the flow exhibits larger velocity gradients, small assimilation errors occur. This is most evident in the eddy viscosity, the model variable of the turbulence model. These deviations from the expected results are found to be related to the inaccurate prediction of the first and second-order derivatives of the velocity. Improving the prediction of these derivatives should be part of future research to further increase the accuracy of the assimilation outcomes.

## Files
[MA_TV_ModelAB.py](https://github.com/timovogel/PINN_NACA0012/blob/main/MA_TV_ModelAB.py) contains the script for using dense and sparse numerical data for the network training. Within the script, the domain can be chosen between the entire considered domain or only the wake. 

[MA_TV_ModelC.py](https://github.com/timovogel/PINN_NACA0012/blob/main/MA_TV_ModelC.py) contains the script for using sparse numerical data for the network training. Within the script, the domain can be chosen between the entire considered domain or only the wake. 

## Licence
[GNU GENERAL PUBLIC LICENSE](https://github.com/timovogel/PINN_NACA0012/blob/main/LICENCE)
