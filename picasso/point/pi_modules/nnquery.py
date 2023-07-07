import torch
import point_nnquery as module


def range_search_(database, query, nvDatabase, nvQuery,
                 radius=0.1, nnsample=None):
    '''
    Input:
        database: (concat_Np, 3+x) float32 array, database points
        query:    (concat_Mp, 3) float32 array, query points
        radius:   float32, range search radius
        dilation_rate: float32, dilation rate of range search
        nnsample: int32, maximum number of neighbors to be sampled
    Output:
        nn_count: (concat_Mp) int32 array, number of neighbors
        nn_index: (Nout, 2) int32 array, neighbor indices
        nn_dist:  (Nout) float32, sqrt distance array
    '''
    database = database[...,:3].contiguous()
    query = query[...,:3].contiguous()
    nvDatabase = torch.cumsum(nvDatabase, dim=-1, dtype=torch.int32)
    nvQuery = torch.cumsum(nvQuery, dim=-1, dtype=torch.int32)

    if nnsample is None:
        nnsample = 2147483647 # int32 maximum value
    # print('nnSample:', nnsample)

    cntInfo, nnIndex, nnDist = module.range(database, query, nvDatabase,
                                            nvQuery, radius, nnsample)
    nnCount = torch.cat([cntInfo[:1], cntInfo[1:]-cntInfo[:-1]],dim=0)
    nnCount = nnCount.contiguous()
    nnCount = nnCount.to(torch.int)
    return cntInfo, nnCount, nnIndex, nnDist


def cube_search_(database, query, nvDatabase, nvQuery,
                length=0.1, nnsample=None, gridsize=4):
    '''
    Input:
        database: (concat_Np, 3) float32 array, database points
        query:    (concat_Mp, 3) float32 array, query points
        length:   float32, cube search length
        dilation_rate: float32, dilation rate of cube search
        nnsample: int32, maximum number of neighbors to be sampled
        gridsize: int32 , cubical kernel size
    Output:
        nn_count: (oncat_Mp) int32 array, number of neighbors
        nn_index: (Nout, 3) int32 array, neighbor and filter bin indices
    '''
    database = database[...,:3].contiguous()
    query = query[...,:3].contiguous()
    nvDatabase = torch.cumsum(nvDatabase, dim=-1, dtype=torch.int32)
    nvQuery = torch.cumsum(nvQuery, dim=-1, dtype=torch.int32)

    if nnsample is None:
        nnsample = 2147483647 # int32 maximum value
    # print('nnSample:', nnsample)

    cntInfo, nnIndex = module.cube(database, query, nvDatabase, nvQuery,
                                   length, nnsample, gridsize)
    nnCount = torch.cat([cntInfo[:1], cntInfo[1:]-cntInfo[:-1]], dim=0)
    nnCount = nnCount.contiguous()
    nnCount = nnCount.to(torch.int)
    return cntInfo, nnCount, nnIndex


def knn3_search_(database, query, nvDatabase, nvQuery):
    '''
    Input:
        database: (concat_Np, 3) float32 array, database points
        query:    (concat_Mp, 3) float32 array, query points
    Output:
        nn_index: (concat_Mp, 3) int32 array, neighbor indices
        nn_dist:  (concat_Mp, 3) float32, sqrt distance array
    '''
    # Return the 3 nearest neighbors of each point
    database = database[...,:3].contiguous()
    query = query[...,:3].contiguous()
    nvDatabase = torch.cumsum(nvDatabase, dim=-1, dtype=torch.int32)
    nvQuery = torch.cumsum(nvQuery, dim=-1, dtype=torch.int32)

    nnIndex, nnDist = module.knn3(database, query, nvDatabase, nvQuery)
    return nnIndex, nnDist