---
title: 'PyNumDiff: A Python package for Numerical Differentiation of noisy time-series data'
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
  - name: J. Nathan Kutz
    affiliation: 2
affiliations:
 - name: Department of Mechanical Engineering, University of Nevada at Reno
   index: 1
 - name: Department of Applied Mathematics, University of Washington
   index: 2
date: 10 July 2021
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
aas-doi: 10.1109/ACCESS.2020.3034077
aas-journal: IEEE Access
---

# Summary

`PyNumdiff` is a Python package that implements various methods for computing numerical derivatives of noisy data, 
which can be a critical step in developing dynamic models or designing control. There are four different families of 
methods implemented in this repository: smoothing followed by finite difference calculation[@author:2001], 
local approximation with linear models[@author2:2002], Kalman filtering based methods[@author3:2002] and 
total variation regularization methods[@author4:2002]. Most of these methods have multiple parameters involved to tune. 
We take a principled approach and propose a multi-objective optimization framework for choosing parameters that minimize 
a loss function to balance the faithfulness and smoothness of the derivative estimate. 


# Statement of need

Current growth of measurement data has popularized the use of data-driven modeling. Unfortunately,
such measurements are usually polluted by noise, making pattern extraction difficult. On the other hand, 
computing numerical derivatives is ubiquitous in the fields of physical, biological and engineering sciences.
And these computations are even more sensitive to the noise. (ToDo: Add more)

`PyNumDiff` is designed to be used by engineering science researchers who work with sensors and are interested in 
aquiring numerical derivative information from the noisy measurement data. (ToDo: Add more)


# Mathematics

Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

You can also use plain \LaTeX for equations
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation}
and refer to \autoref{eq:fourier} from text.


# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

Fundings ??

# References