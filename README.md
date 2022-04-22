# Learning Categorical distributions (The die problem) 

This project implements a model for learning an unknown Categorical
distribution given a sample. If D is a die with d faces, the
goal is to approximate the probability of each face of the
die. This is accomplished by optimizing the likelihood function
of a sample of m throws of the die. 

The model employed is the natural parametrization of a 
exponential family, since categorical distributions 
belong to that family. Algorithms implemented are : SGD,
SNGD, AdaGrad, MOD, MEGD, CSNGD and MAP (check the thesis
for more information about the algorithms and their 
definitions). 

The experiments results can be found in my Ph.D Thesis 
with title: _Efficient and convergent natural gradient
based optimization algorithms for machine learning_ 
 
### Running the default experiments
1. Clone the project and access to the directory in your command line

2. Open the coding file with name _dual-die-prob.py_

3. At the end of the document, uncomment the experiments 
desired to run, among:
    - three_entropy_graphs 
    - mod_surfaces
    - three_entropy_three_dimensions_graph
   
4. Execute the _dual-die-prob.py_ coding file:
```bash
python3 dual-die-prob.py
```

### Experiment description
#### three_entropy_graphs(algs)
variable _algs_ stands for a list of algorithms. Give any list
of algorithms containing following algorithms:
  - sgd 
  - sngd 
  - megd 
  - csngd
  - mod1000
  - map

These algorithms are already initialized, so for example, 
at the end of the document _dual-die-prob.py_, we can write:
```python
three_entropy_graphs([sngd, sgd])
```
This experiment solves the die problem in 3 different scenarios
of entropy. Algorithms' performance is plotted at the end, 
by means of their Kullback-leibler divergence to the 
solution, in 3 graphs separating different entropy scenarios.
Some variables can be changed for a customized
experiment. After the algs variable, the function accepts 
the following named arguments:
  - sample_length= (default 5000. Length of the sample)
  - dimension= (default 199. Dimension of the model)
  - n_repeats= (default 100. Amount of instances for each scenario)

Hence, for example we can write at the end of _dual-die-prob.py_
the following experiment:
```python
three_entropy_graphs([sngd, sgd], sample_length=1000, 
                     dimension= 400,
                     n_repeats= 150)
```
#### three_entropy_three_dimensions_graph(algs)
The only difference with experiment _three_entropy_graph_ is that 
in addition 3 different dimensions are contemplated for the
3 different scenarios of entropy. The result is a 9 graphs
output for the 3 scenarios of entropy and 3 dimensions.
As an example, the 
instruction line at the end of  _dual-die-prob.py_
to run this experiment could be:
```python
three_entropy_three_dimensions_graph([sngd, sgd], 
                                     sample_length=5000,
                                     dimensions=(50, 200, 500),
                                     n_repeats=100)
```
Observe that in this case, the modifiable variable _dimensions_
is a tuple of 3 elements.
#### mod_surfaces()
This experiment runs the 3 scenarios mentioned and solves 
the die problem with several versions of _MOD_ algorithm
with different _eras_ parameter. Then it plot for every
scenario a surface that allows to observe the variability
between the all the _MOD_ algorithms. This experiment 
does not have named arguments. For example, at the end of  
_dual-die-prob.py_ write
```python
mod_surfaces()
```
