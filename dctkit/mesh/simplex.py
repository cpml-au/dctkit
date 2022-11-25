import numpy as np 

def compute_face_to_edge_connectivity(edge):
    #take in input an array of arrays edge in which 
    #any rows is an edge
    #returns the matrix in which any element
    #is the label of an edge
    vals, idx, count = np.unique(edge, axis = 0, return_index = True, return_counts = True) 
    big = np.c_[vals, count, idx] #create the matrix
    big = big[big[:, -1].argsort()] #sort to preserve the initial order
    count = big[:,2] #update count w.r.t the original order
    vals = big[:, :2] #update vals w.r.t the original order
    rep_edges = vals[count > 1] #save the edge repeated
    position = np.array([np.where((edge == i).all(axis=1))[0] for i in rep_edges]) #index position of rep_edges in the original array edge
    position = np.concatenate(position)
    edge_label = np.array(range(len(edge))) #create the vectors of label
    edge_label[position[1::2]] = position[::2] #eliminate duplicate labels
    edge_label = edge_label.reshape(len(edge)//3, 3) #build edge_label matrix
    return edge_label

