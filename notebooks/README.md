# Code Usage and Experiments

| Notebook | What's in it |
| --- | --- |
| [Basic Tutorial](https://github.com/florisvb/PyNumDiff/blob/master/notebooks/1_basic_tutorial.ipynb) | Demo to show invocations of all the major methods in this library on 1D data. |
| [Optimizing Hyperparameters](https://github.com/florisvb/PyNumDiff/blob/master/notebooks/2_optimizing_hyperparameters.ipynb) | All methods' answers are affected by choices of hyperparameters, which complicates differentiation if the true derivative is not known. Here we briefly cover metrics we'd like to optimize and show how to use our `optimize` function to find good hyperparameter choices. |
| [Automatic Method Suggestion](https://github.com/florisvb/PyNumDiff/blob/master/notebooks/3_automatic_method_suggestion.ipynb) | A short demo of how to allow `pynumdiff` to choose a differentiation method for your data. |
| [Performance Analysis](https://github.com/florisvb/PyNumDiff/blob/master/notebooks/4_performance_analyssi.ipynb) | Experiments to compare methods' accuracy and bias across simulations. |
| [Robustness to Outliers Demo](https://github.com/florisvb/PyNumDiff/blob/master/notebooks/5_robust_outliers_demo.ipynb) | This notebook shows a head-to-head of `RTSDiff`'s and `RobustDiff`'s minimum-RMSE performances on simulations with outliers, to illustrate the value of using a Huber loss in the Kalman MAP problem. |
| [Multidimensionality Demo](https://github.com/florisvb/PyNumDiff/blob/master/notebooks/6_multidimensionality_demo.ipynb) | Demonstration of differentating multidimensional data along particular axes. |