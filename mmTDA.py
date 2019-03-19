# -*- coding: utf-8 -*-
"""
This code will run a multiple measurements topological data analysis based 
classifier, and return the relevant cross-validated prediction accuracy. Please
 see the accompanying paper for full details of the methodology.

This code uses the filter function PCA, ie. the first principle component of 
the point cloud. The metric used is Euclidean.

Requirements:
    Python 3.X
    numpy
    pandas
    scipy
    sklearn
    fastcluster
    
An easy installation of the environment can be accomplished by first installing
the Anaconda python distribution. Then, pip can be used to install fastcluster.
    
Data should be prepared in a .csv file, of an nxm matrix of data points. Each 
column should have a feature label. There must also exist a column 'id' to 
denote the individual sample that each observation belongs to, and well as a 
column 'label' for the class label. Classes may be either binary or multi-class 
(integers).

@author: Ryan Ramanujam
@email: ryan@ramanujam.org
"""

import pandas as pd
import numpy as np
import fastcluster
from optparse import OptionParser
from sklearn.decomposition import PCA
import scipy.cluster.hierarchy as hcluster
from sklearn import linear_model
from sklearn.model_selection import cross_val_predict
from sklearn import svm
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier

parser = OptionParser()

# file name if not the default
parser.add_option("-f", action="store", type="string", default="pp6", dest="filename")
# the number of graphs
parser.add_option("-g", type="int", default=10, dest="graphs")
# the number of runs to average
parser.add_option("-r", type="int", default=10, dest="runs")
# if the data should be sampled continuously
parser.add_option("-c", type="int", default=0, dest="cont")

(options, args) = parser.parse_args()


def load_file(datafile):
    """Loads the file with given filename
    """
    
    # load the datafiles, stores in data
    print('Loading datafile:', datafile)
    
    data=pd.read_csv(datafile + '.csv')
    data = data.sample(frac=1).reset_index(drop=True)
  
    return data


def sample_data(data, points, continuous=0):
    """Takes the data, and samples according to number of points passed. The
    output is the data input dataframe and the label column as a dataframe
    """
    
    # randomly sample the data based on number of points (options.points)
    num_id = np.unique(data['id'])
    data_array = np.zeros((np.shape(num_id)[0],3))
    data_array[:,0] = num_id
    data_loc = np.array(())
    for i in range(np.shape(data_array)[0]):
        data_array[i,1] = np.shape(np.where(data['id']==data_array[i,0])[0])[0]
        data_array[i,2] = int(points)
        points_location = np.where(np.in1d(data['id'], data_array[i,0]) == True)[0]
        if (continuous == 0):
            # if datapoints are to be sampled randomly
            data_loc = np.hstack((data_loc, np.random.choice(np.where(np.in1d(data['id'], data_array[i,0]) == True)[0],int(data_array[i,2]), replace=False)))
        else:
            # if datapoints are to be sampled continuously
            # start_index is the index of points_location to begin at
            start_index = np.random.randint(points_location[0],points_location[-1])
            # correct if chosen index is too close to the end of data
            if (points_location[-1] - start_index) < points:
                start_index = points_location[-1] - points
            data_loc = np.hstack((data_loc, np.arange(start_index, start_index + points)))
 
    data_loc.flatten()
    data = data.iloc[data_loc]
    
    # move the label to another dataframe (which also has ids)
    label_col = data[['id','label']]
    data = data.drop('label', axis=1)

    return (data, label_col)


def first_pca(data):
    """Computes the first principle component and returns it for given data
    """

    clf = PCA(n_components=1)
    clf.fit(data)
    
    return (clf.transform(data)[:,0])


def make_dendrogram(data):
    """Runs fastcluster to make a dendrogram and then returns it
    """
    
    # make a dendrogram using fastcluster
    dendro = fastcluster.linkage(data, method='complete', metric='euclidean')
    
    return dendro


def create_nodes(inputs, outputs, filter_fxn, interval_num=10):
    """Uses the inputs and outputs (classes) to build nodes representative of 
    a TDA graph. The output is a matrix of individuals x nodes, where the number 
    of an individual's samples which occured in that node are given. 
    """
    
    # create intervals
    intervals = create_intervals(inputs, filter_fxn, interval_num)
    
    # index of nodes
    curr_node = 0
    
    # find unique samples
    num_id = np.unique(outputs['id'])
    # create a dataframe for the results, with id as the index
    samples_nodes = pd.DataFrame(index=num_id, columns=['label'])
    # create a dataframe for the average values
    avg_vals = pd.DataFrame(index=inputs.columns, columns=['0'])
    # create a dataframe for size and purity
    size_purity = ['size','purity']
    size_purity_df = pd.DataFrame(index=size_purity, columns=['0'])
    
    # map the label column from outputs
    samples_nodes['label'] = samples_nodes.index.map(outputs.drop_duplicates().set_index('id')['label'])

    # create nodes for each interval
    for i in range(interval_num):
        
        # find the data subset (indexes) in this interval
        data_subset_indexes = map_to_interval(filter_fxn, intervals[i], intervals[i+1])
        
        # check that at least one sample maps to this interval
        if (np.shape(data_subset_indexes)[0] >= 1):
        
            # determine which samples map to this interval
            # first get subsets for inputs and outputs
            inputs_subset = inputs.iloc[data_subset_indexes]
            outputs_subset = outputs.iloc[data_subset_indexes].copy()
           
            # check that at least 3 samples map here, otherwise no reason to
            # use local clustering
            if (np.shape(data_subset_indexes)[0] >= 3):
    
                # do complete linkage to determine nodes
                dendrogram = make_dendrogram(inputs_subset)
                
                # determine the epsilon to split on
                epsilon = determine_epsilon(dendrogram, interval_num)
                            
                # determine the clusters based on epsilon and add to outputs_subset
                outputs_subset['clusters'] = hcluster.fcluster(dendrogram, epsilon, criterion='distance')
       
            else:
                # assign them all the same cluster
                outputs_subset['clusters'] = 1
        
            # for each node in this interval, add node columns and sample numbers
            for node in np.unique(outputs_subset['clusters']):
                
                # subset input/output subsets that are in this node
                node_subset_in = inputs_subset[outputs_subset['clusters']==node]
                node_subset_out = outputs_subset[outputs_subset['clusters']==node]
                
                # determine the number of each id that are in this node
                id_counts = pd.DataFrame(node_subset_out['id'].value_counts())
                # and the labels as well
                label_counts = pd.DataFrame(node_subset_out['label'].value_counts())
                
                # find the average feature values for this node
                feature_avg = node_subset_in.mean()
               
                # add to a new column to samples_nodes, with numbers in the node
                samples_nodes[str(curr_node)] = samples_nodes.index.map(id_counts['id'])
                # add average values
                avg_vals[str(curr_node)] = avg_vals.index.map(feature_avg)
                # add size and purity
                size_purity_df.loc['size',str(curr_node)] = label_counts['label'].sum()
                # determine the purity, defined as number of most frequent label / total
                size_purity_df.loc['purity',str(curr_node)] = (label_counts['label'].max())/(label_counts['label'].sum())
                                
                # update node number
                curr_node += 1
                
            # endif a sample maps to this node
            
    # fillna with 0
    samples_nodes.fillna(0, inplace=True)
    
    # concat dataframes and fillna with 0
    avg_vals = pd.concat([avg_vals, size_purity_df])
    avg_vals.fillna(0, inplace=True)
    
    # save the dataframe of additional information
    avg_vals.to_csv('average_node_values.csv')

    # return node_matrix
    return (samples_nodes, avg_vals)



def determine_epsilon(dendro, resolution=20):
    """ This function determines the epsilon value to use for cluster construction.
    """ 
 
    # create array of differences for each range
    diff_array = np.diff(dendro[:,2])
    
    # the size of a bin
    binsize = (dendro[-1,2]-dendro[0,2])/resolution
    
    # find the index in the difference array where gaps occur that are greater than the binsize
    gaps_occur = np.where(diff_array >= binsize)

    # if there is no gap of sufficient size, return max
    if (np.shape(gaps_occur)[1] == 0):
        return (dendro[-1,2])
        
    else:
    
        # return the epsilon for the first suitable gap
        return dendro[gaps_occur[0][0],2]


def map_to_interval(data, interval1, interval2):
    """ Return the indexes which map to a certain interval. Both sides of the
    inequality contain '=' to account for the starting/ending points.
    """
   
    return np.where((data >= interval1) & (data <= interval2))[0]
    


def create_intervals(data, fxn, step_size=5):
    """ Creates a number of non-overlapping intervals based on the limits of the 
        filter function. Intervals do not overlap since we are interested only in 
        the nodes generated, and not the graph per se.
    """
    
    # sort the filter function
    filter_ascending = np.sort(fxn)
    # make an output array
    output_array = np.zeros((step_size+1))
    
    # create the first/last entries
    output_array[0] = filter_ascending[0]
    output_array[-1] = filter_ascending[-1]
   
    # interval size is the range divided by interval count
    interval_size = (filter_ascending[-1] - filter_ascending[0]) / (step_size)
    
    # iterate through and add interval cutoffs
    for i in range(1,step_size+1):
        output_array[i] = output_array[i-1] + interval_size

    return(output_array)


def run_cross_validation(nxm, runs):
    """Takes input and output dataframes and a filter fxn. Based on the number
    of runs, creates nodes and conducts cross-validation on the nodes. The
    output is a weighted average of the accuracy obtained over the bootstrap.
    """
    
    # create a model for training, in this case a sparse logistic regression model
    model = linear_model.LogisticRegressionCV(max_iter=1000, penalty='l1', solver='liblinear', Cs=10, cv=3, multi_class='ovr')
    
    # number correct across all runs
    number_correct = 0
    
    for i in range(runs):

        # make cross-validated model and predict
        model_predictions = cross_val_predict(model, nxm.iloc[:,1:], nxm['label'], cv=5)
        
        # add the number correct for this run
        number_correct = number_correct + (model_predictions == nxm.iloc[:,0]).sum()

    # return overall accuracy
    model_accuracy = number_correct / (np.shape(nxm)[0] * runs)
    
    return model_accuracy


def run_grid_search(input, opt):
    """This controls running sampling of different numbers of datapoints and
    determining results. For each sampling, a TDA graph (nodes) is constructed a 
    number of times based on the options given (default=10). Then, the 
    procedure of sampling,  determining nodes, and feeding to the machine learning 
    algorithm and alternative model are conducted. The     results are printed to screen.
    """
    
    # some lists of datapoints, results
    points_list = []
    accuracy_list = []
    alternate_list = []
    
    # loop the search
    for datapoints in [10,20,30,40,50,60,70,80,90,100,150,200,300,400,500,
              1000,2000,3000,4000,5000]:
    
        # find the minimum number of samples
        sample_counts =  np.bincount(input['id'])[np.where(np.bincount(input['id'])>0)]
        
        # set overall and alternative accuracy for this number of datapoints
        overall_accuracy = 0
        alternative_accuracy = 0
        
        #check that the maximum limit of samples has not been exceeded
        if (datapoints <= np.min(sample_counts)):
            
            print('Creating model with sampling using datapoints:' ,datapoints)
            
            # loop as this must be repeated
            for current_graph in range(opt.graphs):
                
                print('Making graph number:', current_graph+1)
    
                # sample the data points
                (data_in, data_out) = sample_data(input, datapoints, opt.cont)
            
                # create filter function
                data_in = data_in.drop('id', axis=1)
                filter_fxn = first_pca(data_in)
            
                # make the graph
                (node_matrix, avg_values) = create_nodes(data_in, data_out, filter_fxn)
                
                # do the entire procedure for this run, finding accuracy
                overall_accuracy = overall_accuracy + run_cross_validation(node_matrix, opt.runs)
                alternative_accuracy = alternative_accuracy + run_alt_accuracy(data_in, data_out, node_matrix, opt.runs)
                
                # if you want to monitor output
                print('Points:',datapoints,'Graph number:',current_graph+1,
                      'Accuracy:',overall_accuracy/(current_graph+1),
                      'Alternate Acc:',alternative_accuracy/(current_graph+1))
            
            # add the accuracy results to the lists
            points_list.append(datapoints)
            accuracy_list.append(overall_accuracy/opt.graphs)
            alternate_list.append(alternative_accuracy/opt.graphs)
            
    # create a DataFrame of results lists and print
    data_tuples = list(zip(points_list, accuracy_list, alternate_list))
    results_df = pd.DataFrame(data_tuples, columns=['Datapoints','Accuracy','Alternate_Acc'])
    print(results_df)
    
    return 0

def run_alt_accuracy(inputs, outputs, nxm, runs):
    """Finds the alternative accuracy, being a SVM voting classifier. Inputs
    are the original inputs, outputs are the original outputs. nxm is the
    nodes_matrix, in n x m form. runs are passed and given in options.
    
    Data is transformed to allow the SVM to converge to a solution. Kernel can
    be chosed based on the data.
    """

    # use an SVM model
    training_model = svm.SVC(kernel='rbf',gamma='auto')
    
    min_max_scaler = preprocessing.MinMaxScaler()
    inputs_transformed = min_max_scaler.fit_transform(inputs.astype(float))
    
    # number correct across all runs
    number_correct = 0
    
    for i in range(runs):

        # make cross-validated model and predict
        model_predict = cross_val_predict(training_model, inputs_transformed, outputs['label'], cv=5)
        
        # check the voting per id
        for index, row in nxm.iterrows():
            # find the svm predicted label by vote
            svm_prediction = np.bincount(model_predict[np.where(outputs['id']==index)[0]]).argmax()
            
            # compare with actual label
            if (svm_prediction == row['label']):
                number_correct += 1

    # return overall accuracy
    svm_accuracy = number_correct / (np.shape(nxm)[0] * runs)
    
    return svm_accuracy


if __name__ == '__main__':
    
    # load data file and related tasks
    data = load_file(options.filename)
        
    # run a grid search of different samplings
    run_grid_search(data, options)
    
    print('Done!')
