# mmTDA
Code for paper "A topological data analysis based classification method for multiple measurements.

This code will run a multiple measurements topological data analysis based 
classifier, and return the relevant cross-validated prediction accuracy. Please
 see the accompanying paper for full details of the methodology.
 
This code uses the filter function PCA, ie. the first principle component of 
the point cloud. The metric used is Euclidean.

# Requirements
* Python 3.X
* numpy
* pandas
* scipy
* sklearn
* fastcluster
    
An easy installation of the environment can be accomplished by first installing
the Anaconda python distribution. Then, pip can be used to install fastcluster.
    
Data should be prepared in a .csv file, of an nxm matrix of data points. Each 
column should have a feature label. There must also exist a column 'id' to 
denote the individual sample that each observation belongs to, and well as a 
column 'label' for the class label. Classes may be either binary or multi-class 
(integers).

The default sample file (pp6.csv) gives an example of a 6 class problem, where each class represents a different point process. From each class, 400 individual samplings of each of the point processes are grouped together with a common id. Please see the paper for further details.

# Options
* -f filename: the filename of the .csv (without .csv). The default is 'pp6' to use the sample file.
* -g graphs: the number of graphs to build. The default is 10.
* -r runs: the number of runs for each graph. The default is 10.
* -c continuous: whether the data should be sampled randomly (0) or continuously (1). The default is 0 for random sampling.
