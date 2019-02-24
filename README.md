# Machine-Learning-Under-Optimization-Lens-Fall2018
Work from class 15.095 Machine Learning Under a Modern Optimization Lens at MIT taught by Dimitris Bertsimas and Martin Copenhaver

## Course Content and Objectives: 

The majority of the central problems of regression, classification, and estimation have been addressed using heuristic methods even though they can be formulated as formal optimization problems. While continuous optimization approaches have had a significant impact in Machine Learning (ML)/Statistics (S), mixed integer optimization (MIO) has played a very limited role, primarily based on the belief that MIO models are computationally intractable. The last three decades have witnessed (a) algorithmic advances in MIO, which coupled
with hardware improvements have resulted in an astonishing over 2 trillion factor speedup in solving MIO problems, (b) significant advances in our ability to model and solve very high dimensional robust and convex optimization models. The objective in this course is to revisit some of the classical problems in ML/S and demonstrate that they can greatly benefitt from a modern optimization treatment. The optimization lenses we use in this course include convex, robust, and mixed integer optimization. In all cases we demonstrate that optimal solutions to large scale instances (a) can be found in seconds, (b) can be certified to be optimal/near-optimal in minutes and (c) outperform classical heuristic approaches in out of sample experiments involving real and synthetic data.

The problems we address in this course include:
* variable selection in linear and logistic regression,
* convex, robust, and median regression,
* an algorithmic framework to construct linear and logistic regression models that satisfy properties of sparsity, robustness, significance, absence of multi-collinearity in an optimal way,
* clustering,
* deep learning,
* how to transform predictive algorithms to prescriptive algorithms,
* optimal prescriptive trees,
* the design of experiments via optimization ,
* missing data imputation

## Homework / assignments
The course had 4 assignments and one final project. The 4 assignments were:
* 1: Solve a linear optimization problem and build a robust linear regression model
* 2: Build an algorithmic framework for linear regression that satisfies properties of sparsity, robustness, significance, absence of multi-collinearity in an optimal way
* 3: Construct state-of-the-art optimal tree models
* 4: Perform prescriptive analysis to decide how much stock to have in a store to optimize revenue

## Details
* _Language_: Python and Julia/JuMP over Jupyter Notebooks.
* _Libraries_ in Python: numpy, pandas, matplotlib, scitkit-learn
* _Libraries_ in Julia: JuMP, DataFrames, Gurobi, Plots, MLDataUtils, OptimalTrees, OptImpute

## Authors 
* Kim-Anh-Nhi Nguyen @kanguyn

## Sources and acknowledgments
* Prof.: [Dimitris Bertsimas](https://web-cert.mit.edu/dbertsim/www/) and [Martin Copenhaver](http://www.mit.edu/~mcopen/)
* Teaching assistants: [Colin Pawlowski](cpawlows@mit.edu) and [Yuchen Wang](yuchenw@mit.edu)
