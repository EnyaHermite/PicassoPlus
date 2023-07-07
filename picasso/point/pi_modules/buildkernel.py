import torch
import point_buildkernel as module


def SPH3Dkernel_(database, query, nn_index, nn_dist, radius, kernel=[8,2,3]):
    '''
    Input:
        database: (concat_Np, 3+) float32 array, database points (x,y,z,...)
        query:    (concat_Mp, 3+) float32 array, query points (x,y,z,...)
        nn_index: (Nout, 2) int32 array, neighbor indices
        nn_dist:  (Nout) float32, sqrt distance array
        radius:   float32, range search radius
        kernel:   list of 3 int32, spherical kernel size
    Output:
        filt_index: (Nout) int32 array, filter bin indices
    '''
    database = database[:,:3].contiguous()  #(x,y,z)
    query = query[:,:3].contiguous()        #(x,y,z)
    filt_index = module.SPH3D(database, query, nn_index, nn_dist, radius, *kernel)
    return filt_index



def fuzzySPH3Dkernel_(database, query, nn_index, nn_dist, radius, kernel=[8,4,1]):
    '''
    Input:
        database: (concat_Np, 3+) float32 array, database points (x,y,z,...)
        query:    (concat_Mp, 3+) float32 array, query points (x,y,z,...)
        nn_index: (Nout, 2) int32 array, neighbor indices
        nn_dist:  (Nout) float32, sqrt distance array
        radius:   float32, range search radius
        kernel:   list of 3 int32, spherical kernel size
    Output:
        filt_index: (Nout, 3) int32 array, fuzzy filter indices,
        filt_coeff: (Nout, 3) float32 array, fuzzy filter weights,
                    kernelsize=prod(kernel)+1
    '''
    database = database[:,:3].contiguous()   #(x,y,z)
    query = query[:,:3].contiguous()         #(x,y,z)
    filt_index, filt_coeff = module.fuzzySPH3D(database, query, nn_index, nn_dist, radius, *kernel)
    return filt_index, filt_coeff








