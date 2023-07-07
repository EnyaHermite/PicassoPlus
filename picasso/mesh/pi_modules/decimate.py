import torch
import mesh_decimation as module


def mesh_decimation_(vertexIn, faceIn, geometryIn, nvIn, mfIn, nvOut, useArea=True, wgtBnd=5):
    '''
    inputs:
        vertexIn:   (batch_npoints, 3+) float32 array, concatenated points with/without features
        faceIn:     (batch_nfaces, 3) int32 array, concatenated triangular faces
        geometryIn: (batch_nfaces, 5) float32 array, geometrics of each face which is
                                      composed of [normal=[nx,ny,nz],intercept=d,area]
        nvIn:       (batch,) int32 vector, point/vertex number of each sample in the batch
        mfIn:       (batch,) int32 vector, face number of each sample in the batch
        nvOut:      (batch,) int32 vector, expected number of points/vertices to output of each sample
        wgtBnd:      float scalar, weight boundary quadric error, (>1) preserves boundary edges
    returns:
        vertexOut:   (batch_mpoints, 3) float32 array, concatenated points with/without features
        faceOut:     (batch_mfaces, 3) int32 array, concatenated triangular faces
        geometryOut: (batch_mfaces, 5) float32 array, geometrics of each output face which is
                                       composed of [normal=[nx,ny,nz],intercept=d,area]
        nvOut:       (batch,) int32 vector, point/vertex number of each sample in the batch
        mfOut:       (batch,) int32 vector, face number of each sample in the batch
        vtReplace:   (batch_npoints,) int32 array, negative values (remove minus '-') for vertex to be
                                      contracted in each cluster; zero for vertex no change because it
                                      forms a cluster by itself; positive values recording the cluster
                                      size excluding the vertex itself
        vtMap:       (batch_npoints,) int32 array, contracted/degenerated vertices got mapping to -1,
                                      the valid vertices got mapping start from 0
    '''
    nv2Remove = nvIn - nvOut
    repIn = torch.zeros([vertexIn.shape[0]], dtype=torch.int32, device=nvIn.get_device())
    mapIn = torch.arange(repIn.shape[0], dtype=torch.int32, device=nvIn.get_device())

    useArea = torch.tensor(useArea)
    wgtBnd = torch.tensor(wgtBnd)

    while torch.any(nv2Remove>0):
        nvIn_cumsum = torch.cumsum(nvIn, dim=-1, dtype=torch.int32)
        mfIn_cumsum = torch.cumsum(mfIn, dim=-1, dtype=torch.int32)

        vertexOut, faceOut, isDegenerate, repOut, mapOut, \
        nvOut, mfOut = module.simplify(vertexIn, faceIn, geometryIn, \
                                       nvIn_cumsum, mfIn_cumsum, nv2Remove, \
                                       useArea, wgtBnd)
        faceIn = faceOut[~isDegenerate,:]
        vertexIn = vertexOut[mapOut>=0,:]
        geometryIn = compute_triangle_geometry_(vertexIn, faceIn)
        nv2Remove = nv2Remove - (nvIn - nvOut)
        repIn, mapIn = combine_clusters_(repIn, mapIn, repOut, mapOut)
        nvIn, mfIn = nvOut, mfOut

    return vertexIn, faceIn, geometryIn, nvIn, mfIn, repIn, mapIn


def combine_clusters_(repA, mapA, repB, mapB):
    '''
       inputs:
            repA: (batch_points,) int32 array, vertex clustering information of LARGE input
            mapA: (batch_points,) int32 array, vertex mappinging information of LARGE input
            repB: (batch_points,) int32 array, vertex clustering information of SMALL/decimated input
            mapB: (batch_points,) int32 array, vertex mappinging information of SMALL/decimated input
       returns:
            repComb: (batch_points,) int32 array, vertex clustering information after merging LARGE/SMALL input
            mapComb: (batch_points,) int32 array, vertex mappinging information after merging LARGE/SMALL input
    '''
    repComb, mapComb = module.combine_clusters(repA, mapA, repB, mapB)
    return repComb, mapComb


def compute_triangle_geometry_(vertex, face):
    '''
    Compute normals of the facets (0-order: no interpolation)
    '''
    vertex = vertex[:,:3]
    face = face.to(torch.long)
    vec10 = vertex[face[:,1],:] - vertex[face[:,0],:]
    vec20 = vertex[face[:,2],:] - vertex[face[:,0],:]
    raw_normal = torch.cross(vec10,vec20,-1)
    l2norm = torch.sqrt(torch.sum(raw_normal**2,dim=-1,keepdim=True))
    area = l2norm/2
    normal = raw_normal/(l2norm+1e-10)  # unit length normal vectors
    v1 = vertex[face[:,0],:]
    d = -torch.sum(normal*v1,dim=-1,keepdim=True)  # intercept of the triangle plane
    geometry = torch.cat([normal,d,area],dim=-1)
    return geometry


def count_vertex_adjface_(face, vtMap, vertexOut):
    '''
        Count the number of adjacent faces for output vertices, i.e. {vtMap[i] | vtMap[i]>=0},
    '''
    nfCount = module.count_vertex_adjface(face, vtMap, vertexOut[:,:3].contiguous())
    return nfCount
