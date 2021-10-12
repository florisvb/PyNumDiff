---
title: 'PyNumDiff: A Python package for numerical differentiation of noisy time-series data'
tags:
  - Python
  - numerical differentiation
  - denoising
  - dynamics
  - time series
  - machine learning
authors:
  - name: Floris Van Breugel^[corresponding author]
    affiliation: 1
  - name: Yuying Liu
    affiliation: 2
  - name: Bingni W. Brunton
    affiliation: 3
  - name: J. Nathan Kutz
    affiliation: 2
affiliations:
 - name: Department of Mechanical Engineering, University of Nevada at Reno
   index: 1
 - name: Department of Applied Mathematics, University of Washington
   index: 2
- name: Department of Biology, University of Washington
   index: 3
date: 10 July 2021
bibliography: paper.bib
---

# Statement of need

The numerical computation of derivatives is ubiquitous in every scientific discipline and engineering application because derivatives express fundamental relationships among many quantities of interest. As a result, a large number of diverse algorithms have been developed to differentiate numerical data.  These efforts are challenging because, in reality, practitioners  often have sparse and noisy measurements and data, which undermine the ability to estimate accurate derivatives.  Among the diversity of mathematical approaches that have been formulated, many are ad hoc in nature and require significant bespoke tuning of multiple parameters to produce reasonable results. Thus, at a practical level, it is often unclear which method should be used, how to choose parameters, and how to compare results from different methods.

The ability to assess the most appropriate derivative estimation algorithm to use based upon features of the data is of paramount importance.  Currently, practitioners must often implement a number of methods, relying on disparate software packages, before selecting one that is appropriate for their application.  Whereas high-quality data can leverage simple and computationally efficient algorithms (e.g. finite-differences), data that are sparse and noisy requires more sophisticated algorithms that are also more computationally expensive (e.g. total variation regularization).  Regardless of application domain, scientists of various levels of mathematical expertise would benefit from a unified toolbox for differentiation techniques and parameter tuning. To address these needs, we built the open-source package `PyNumDiff`, with two primary goals in mind: (1) to develop a unified source for a diversity of differentiation methods using a common API, and (2) to provide an objective approach for choosing optimal parameters with a single universal hyperparameter (`gamma`) that functions similarly for all differentiation methods[@van2020numerical]. By filling these needs, `PyNumdiff` facilitates easy computations of derivatives on diverse time-series data sets.
	

# Summary

`PyNumDiff` is a Python package that implements methods for computing numerical derivatives of noisy data. 
In this package, we implement four commonly used families of differentiation methods whose mathematical formulations have different 
underlying assumptions, including both global and local methods [@ahnert2007numerical]. The first family of methods usually start by 
applying a smoothing filter to the data, followed by a finite difference calculation[@butterworth1930theory]. 
The second family relies on building a local model of the data through linear regression, and then analytically 
calculating the derivative based on the model[@belytschko1996meshless; @schafer2011savitzky; @savitzky1964smoothing]. 
The third family we consider is the Kalman filter[@kalman1960new; @henderson2010fundamentals; @aravkin2017generalized; @crassidis2004optimal], 
with unknown noise and process characteristics.   The last family is an optimization approach based on total variation 
regularization (TVR) method [@rudin1992nonlinear; @chartrand2011numerical]. For more technical details, 
refer to [@van2020numerical]. Individual methods under each family are accessed through the API as `pynumdiff.family.method`. 

Applying `PyNumDiff` usually 
takes three steps: (i) pick a differentiation method, (ii) obtain optimized parameters and (iii) apply the differentiation. 
Step (ii) can be skipped if one wants to manually assign the parameters, which is recommended when computation time is limited and the timeseries is long. Alternatively for long timeseries, optimal parameters can be chosen using a short but representative subset of the data. This optimization routine is provided as a sub-module (pynumdiff.optimize) with the same structure of differentiation families (i.e. `pynumdiff.optimize.family.method`). By default, the package performs the optimization using the open source CVXOPT package. Faster solutions can be achieved by using proprietary solvers such as MOSEK. 

The software package includes tutorials in the form of Jupyter notebooks. These tutorials demonstrate the usage of the aforementioned
features. For more detailed information, there is a more comprehensive Sphinx documentation associated with the repository.

# Acknowledgements

The work of J. Nathan Kutz was supported by the Air Force Office of Scientific Research under Grant FA9550-19-1-0011 and FA9550-19-1-0386. The work of F. van Breugel was supported by NIH grant P20GM103650, Air Force Research Lab award FA8651-20-1-0002 Airforce Office of Scientific Research FA9550-21-0122. BWB acknowledges support from the Air Force Office of Scientific Research award FA9550-19-1-0386.

# References

