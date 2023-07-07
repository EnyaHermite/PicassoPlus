#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>
#include <thrust/mismatch.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cmath> // sqrtf
#include <algorithm>    // std::random_shuffle
#include <cstdlib>      // std::rand, std::srand
#include <cassert>

#define myEPS 2.220446049250313e-16F
#define rcondEPS 1e-6F
#define minSinTheta 0.001F
#define areaMAGNITUDE 100.0F    // assume input in metre, change Area scale to decimetre

// reference appreciation: https://stackoverflow.com/questions/34697937/unique-rows-from-linearized-matrix-cuda
// edge_sort_func<T> sort the edge array in ascending order
template <typename T>
struct edge_sort_func
{
    int cols;
    T* data;
    edge_sort_func(int _cols, T* _data) : cols(_cols),data(_data) {};
    __host__ __device__
    bool operator()(int c1, int c2){
        for (int i = 0; i < 2; i++){
            if (data[c1+i*cols] < data[c2+i*cols])
                return true;
            else if (data[c1+i*cols] > data[c2+i*cols])
                return false;}
        return false;
    }
};

// reference appreciation: https://stackoverflow.com/questions/34697937/unique-rows-from-linearized-matrix-cuda
// edge_unique_func<T> unique the edge array
template <typename T>
struct edge_unique_func
{
    int cols;
    T* data;
    edge_unique_func(int _cols, T* _data) : cols(_cols),data(_data) {};
    __device__
    bool operator()(int c1, int c2){
        thrust::pair<T*, T*> res1 = thrust::mismatch(thrust::seq, data+c1, data+c1+1, data+c2);
        if (res1.first!=data+c1+1)
            return false;
        else
        {
            thrust::pair<T*, T*> res2 = thrust::mismatch(thrust::seq, data+cols+c1, data+cols+c1+1, data+cols+c2);
            return (res2.first==data+cols+c1+1);
        }
    }
};

__device__ void NormalizeCrossProductDim3(const float* vecA, const float* vecB, float* vecCross)
{
    vecCross[0] = vecA[1]*vecB[2] - vecA[2]*vecB[1];
    vecCross[1] = vecA[2]*vecB[0] - vecA[0]*vecB[2],
    vecCross[2] = vecA[0]*vecB[1] - vecA[1]*vecB[0];
    float length = std::sqrt(vecCross[0]*vecCross[0] + vecCross[1]*vecCross[1] + vecCross[2]*vecCross[2]);
    length = length>1e-20f?length:1e-20f;
    vecCross[0] /= length; vecCross[1] /= length; vecCross[2] /= length;
}

__device__ void CrossProductDim3(const float* vecA, const float* vecB, float* vecCross)
{
    vecCross[0] = vecA[1]*vecB[2] - vecA[2]*vecB[1];
    vecCross[1] = vecA[2]*vecB[0] - vecA[0]*vecB[2],
    vecCross[2] = vecA[0]*vecB[1] - vecA[1]*vecB[0];
}

__device__ float DotProductDim3(const float* vecA, const float* vecB)
{
    return (vecA[0]*vecB[0] + vecA[1]*vecB[1] + vecA[2]*vecB[2]);
}

__device__ float RrefDim3(const float A[3][3], const float b[3], float invA[3][3], float xyz[3])
{
    float Mat[3][7];
    for(int row=0;row<3;row++)
    {
        for (int col=0;col<3;col++)
        {
            Mat[row][col] = A[row][col];
            if (row==col)
                Mat[row][col+4] = 1;
            else
                Mat[row][col+4] = 0;
        }
    }
    Mat[0][3] = b[0]; Mat[1][3] = b[1]; Mat[2][3] = b[2];

    // compute infinity norm of A
    float normA=0;
    for(int row=0;row<3;row++)
    {
        float rowSum = 0;
        for(int col=0;col<3;col++)
            rowSum += std::abs(Mat[row][col]);
        if (normA<rowSum)
            normA = rowSum;
    }

    // matlab EPS of 'single' datatype is 1.1920929e-07, of 'double' is 2.220446049250313e-16
    const int m=3, n=7;
    float tol = myEPS*n*normA;

    int i=0, j=0;
    while ((i<m) && (j<n))
    {
        // Find value and index of largest element in the remainder of column j.
        float p=std::abs(Mat[i][j]);
        int k=i;
        for(int row=i+1;row<m;row++)
        {
            if (p<std::abs(Mat[row][j]))
            {
                p=std::abs(Mat[row][j]);
                k = row;
            }
        }

        if (p <= tol)
        {
            // The column is negligible, zero it out.
            for(int row=i;row<m;row++)
                Mat[row][j] = 0;
            j = j + 1;
        }
        else
        {
            // Swap i-th and k-th rows.
            float temp[n] = {0};
            for(int col=j;col<n;col++)
                temp[col] = Mat[i][col];
            for(int col=j;col<n;col++)
            {
                Mat[i][col] = Mat[k][col];
                Mat[k][col] = temp[col];
            }

            // Divide the pivot row by the pivot element.
            float pivot = Mat[i][j];
            for(int col=j;col<n;col++)
            {
                Mat[i][col] = Mat[i][col]/pivot;
            }

            // Subtract multiples of the pivot row from all the other rows.
            for(int row=0;row<i;row++)
            {
                const float value = Mat[row][j];
                for(int col=j;col<n;col++)
                    Mat[row][col] -= (value*Mat[i][col]);
            }
            for(int row=i+1;row<m;row++)
            {
                const float value = Mat[row][j];
                for(int col=j;col<n;col++)
                    Mat[row][col] -= (value*Mat[i][col]);
            }
            i++;
            j++;
        }
    }

    xyz[0] = -Mat[0][3]; xyz[1] = -Mat[1][3]; xyz[2] = -Mat[2][3];
    for(int row=0;row<3;row++)
        for(int col=0;col<3;col++)
            invA[row][col] = Mat[row][col+4];

    // infinity norm of the inverse of A
    float normInvA=0;
    for(int row=0;row<3;row++)
    {
        float rowSum = 0;
        for(int col=0;col<3;col++)
            rowSum += std::abs(invA[row][col]);
        if (normInvA<rowSum)
            normInvA = rowSum;
    }

    float rcond = 1/(normA*normInvA);
    return rcond;
}

__device__ float ComputeError(const float xyz[3], const float A[3][3], const float b[3], const float& c)
{
    float cost =   ((xyz[0]*A[0][0] + xyz[1]*A[1][0] + xyz[2]*A[2][0])*xyz[0]
                 +  (xyz[0]*A[0][1] + xyz[1]*A[1][1] + xyz[2]*A[2][1])*xyz[1]
                 +  (xyz[0]*A[0][2] + xyz[1]*A[1][2] + xyz[2]*A[2][2])*xyz[2])
                 +  (xyz[0]*b[0] + xyz[1]*b[1] + xyz[2]*b[2])*2
                 +   c;
    return cost;
}

// compute Quadric = (A, b, c) = [ q11, q12, q13, q14;
//                                 q12, q22, q23, q24;
//                                 q13, q23, q33, q34;
//                                 q14, q24, q34, q44; ]
__device__ void GenerateQuadrics(const float* normal, const float& d, float* Q)
{

    Q[0] = normal[0]*normal[0]; Q[1] = normal[0]*normal[1]; Q[2] = normal[0]*normal[2]; // q11, q12, q13
    Q[3] = normal[1]*normal[1]; Q[4] = normal[1]*normal[2]; Q[5] = normal[2]*normal[2]; // q22, q23, q33
    Q[6] = normal[0]*d;         Q[7] = normal[1]*d;         Q[8] = normal[2]*d;         // q14, q24, q34
    Q[9] = d*d; // q44
}

__device__ void AddQuadrics(const float* Q1, const float* Q2, float* Q)
{
    for(int i=0;i<10;i++)
        Q[i] = Q1[i] + Q2[i];
}


__global__ void initVertexQuadrics(int Nf, const bool useArea, const int* faceIn,
                                   const float* planeIn, float* vertexQuadric)
{
    int fcIdx = blockIdx.x*blockDim.x + threadIdx.x; // global face index in the batch

    if (fcIdx<Nf) // index must be in the legal range
    {
        // geometric information of the triangular face
        const float* normal = &planeIn[5*fcIdx]; // [0,1,2]
        float d       =  planeIn[5*fcIdx+3];
        float area    =  planeIn[5*fcIdx+4]*areaMAGNITUDE;
        float Q[10];
        GenerateQuadrics(normal, d, Q);

        // weighting value for the Quadric computation
        float wgtArea = 1.0f;
        if (useArea)  {wgtArea = area/3;}

        // accumulate Quadrics for each vertex
        for (int k=0; k<3; k++)
        {
            int vtIdx = faceIn[3*fcIdx+k];  // assume: already global vertex index in the batch
            for(int it=0;it<10;it++)
                atomicAdd(&vertexQuadric[10*vtIdx+it],  Q[it]*wgtArea);
        }
    }
}


// one column for one edge
__global__ void extractEdges(int Nf, const int* faceIn, int* edgeOut)
{
    int fcIdx = blockIdx.x*blockDim.x + threadIdx.x; // global face index in the batch

    if (fcIdx<Nf) // index must be in the legal range
    {
        // we store global vertex index for the edge such that applying thrust to them at once
        // One face for 3 edges, therefore the length of edges will be 3 times of faces'.
        // 3 edges, 9 integers including face index to be stored
        int Ne = 3*Nf; // the total number of edges in the batch
        int v1 = faceIn[3*fcIdx];
        int v2 = faceIn[3*fcIdx+1];
        int v3 = faceIn[3*fcIdx+2];
        if (v1<=v2)
        {
            edgeOut[3*fcIdx]      = v1;    // in each edge, the vertex has its global index in the batch
            edgeOut[3*fcIdx+Ne]   = v2;
            edgeOut[3*fcIdx+Ne*2] = fcIdx; // record mapping between face index in the batch
        } else{
            edgeOut[3*fcIdx]      = v2;
            edgeOut[3*fcIdx+Ne]   = v1;
            edgeOut[3*fcIdx+Ne*2] = fcIdx;
        }
        if (v1<=v3)
        {
            edgeOut[3*fcIdx+1]      = v1;
            edgeOut[3*fcIdx+1+Ne]   = v3;
            edgeOut[3*fcIdx+1+Ne*2] = fcIdx;
        } else{
            edgeOut[3*fcIdx+1]      = v3;
            edgeOut[3*fcIdx+1+Ne]   = v1;
            edgeOut[3*fcIdx+1+Ne*2] = fcIdx;
        }
        if (v2<=v3)
        {
            edgeOut[3*fcIdx+2]      = v2;
            edgeOut[3*fcIdx+2+Ne]   = v3;
            edgeOut[3*fcIdx+2+Ne*2] = fcIdx;
        } else{
            edgeOut[3*fcIdx+2]      = v3;
            edgeOut[3*fcIdx+2+Ne]   = v2;
            edgeOut[3*fcIdx+2+Ne*2] = fcIdx;
        }
    }
}


//  Ne is the total number of edges in the batch
__global__ void addBoundaryQuadrics(const int Ne, const int D, const bool useArea, const float wgtBnd,
                                    const float* vertexIn, const float* planeIn, const int* edgeOut,
                                    const int* sortedEdgeIdx, float* vertexQuadric)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x; // global index of the current thread

    if (idx<Ne) // edge index must be within [0,Ne-1]
    {
        int curr_vi = edgeOut[sortedEdgeIdx[idx]];
        int curr_vj = edgeOut[sortedEdgeIdx[idx]+Ne];

        // boundary edge initialized as true, if the same edge exist in its neighboring
        // update it to be false
        bool isBndEdge = true;
        if (idx>0 && isBndEdge)
        {
            int prev_vi = edgeOut[sortedEdgeIdx[idx-1]];
            int prev_vj = edgeOut[sortedEdgeIdx[idx-1]+Ne];
            if (curr_vi==prev_vi && curr_vj==prev_vj)
            {
                isBndEdge = false; // not boundary edge
            }
        }
        if (idx<(Ne-1) && isBndEdge)
        {
            int next_vi = edgeOut[sortedEdgeIdx[idx+1]];
            int next_vj = edgeOut[sortedEdgeIdx[idx+1]+Ne];
            if (curr_vi==next_vi && curr_vj==next_vj)
            {
                isBndEdge = false; // not boundary edge
            }
        }

        if (isBndEdge)  // if the current edge is boundary edge
        {
            int fcIdx = edgeOut[sortedEdgeIdx[idx]+Ne*2]; // get the corresponding global face index

            // geometric information of the triangular face
            const float* normal = &planeIn[5*fcIdx];    //[0,1,2]=[nx,ny,nz]
            const float* xyz_vi = &vertexIn[curr_vi*D]; //[0,1,2]=[ x, y, z]
            const float* xyz_vj = &vertexIn[curr_vj*D]; //[0,1,2]=[ x, y, z]
            float Dxyz[3] = {xyz_vj[0]-xyz_vi[0], xyz_vj[1]-xyz_vi[1], xyz_vj[2]-xyz_vi[2]};

            float bndNormal[3];
            NormalizeCrossProductDim3(normal, Dxyz, bndNormal);
            float  d         = -DotProductDim3(bndNormal, xyz_vi);
            float  bndArea   =  DotProductDim3(Dxyz, Dxyz)*areaMAGNITUDE;
            if (useArea)
                bndArea *= wgtBnd;
            else
                bndArea = wgtBnd;

            float Q[10];
            GenerateQuadrics(bndNormal, d, Q);

            // accumulate Quadrics for edge vertices vi, vj
            for(int it=0;it<10;it++)
            {
                atomicAdd(&vertexQuadric[10*curr_vi+it],  Q[it]*bndArea);
                atomicAdd(&vertexQuadric[10*curr_vj+it],  Q[it]*bndArea);
            }
        }
    }
}


// note the candidate edges are unique: arrangement of `edgeCost' corresponds to `uniqueEdgeIdx'
__global__ void computeEdgeCost(const int uniqueNe, const int Ne, const int D,
                                const float* vertexIn, const int* edgeOut, const int* uniqueEdgeIdx,
                                const float* vertexQuadric, float* edgeCost)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;  // global edge (cost) index in the batch

    if (idx<uniqueNe) // in the range [0,numUniqueEdges-1]
    {
        int vi = edgeOut[uniqueEdgeIdx[idx]];
        int vj = edgeOut[uniqueEdgeIdx[idx]+Ne];

        // add Quadrics of vi and vj
        float Q[10];   //(A,b,c)
        AddQuadrics(&vertexQuadric[10*vi], &vertexQuadric[10*vj], Q);

        // redundant copying for EXPLICITY
        float A[3][3] = {Q[0], Q[1], Q[2], Q[1], Q[3], Q[4], Q[2], Q[4], Q[5]};
        float b[3] = {Q[6], Q[7], Q[8]};
        float c    =  Q[9];

        float invA[3][3];
        float opt_xyz[3];
        float rcondA = RrefDim3(A, b, invA, opt_xyz);

        if (rcondA>rcondEPS) // A is well invertible
        {
            edgeCost[idx] =  ComputeError(opt_xyz, A, b, c);
        }
        else  // A is a singular matrix
        {
            float errI = ComputeError(&vertexIn[vi*D], A, b, c);
            float errJ = ComputeError(&vertexIn[vj*D], A, b, c);
            edgeCost[idx] = errI<errJ?errI:errJ;
        }
    }
}


// accumulate cluster Quadrics,
__global__ void VertexClusterQuadrics(const int B, const int D, const int* nvIn, const int* vtReplace,
                                      const float* vertexIn, float* vertexQuadric, float* vertexOut)
{
    int vj = blockIdx.x*blockDim.x + threadIdx.x; // old (global) vertex index in the batch

    int Nv = nvIn[B-1];
    if (vj<Nv && vtReplace[vj]<0) // in the legal vertex range
    {
        int vi = -vtReplace[vj];  // new-replaced (global) vertex index in the batch
        for(int it=0;it<10;it++)  // accumulate vertex Quadrics
            atomicAdd(&vertexQuadric[10*vi+it], vertexQuadric[10*vj+it]);

        for(int it=0;it<D;it++)   // accumulate vertex XYZ
            atomicAdd(&vertexOut[D*vi+it], vertexIn[D*vj+it]);
    }
}


// compute optimal contracted location of output vertex in each cluster
__global__ void VertexClusterContraction(const int B, const int D, const int* nvIn, const int* vtReplace,
                                         const float* vertexIn, float* vertexQuadric, float* vertexOut)
{
    int vj = blockIdx.x*blockDim.x + threadIdx.x; // old (global) vertex index in the batch

    int Nv = nvIn[B-1];
    if (vj<Nv) // in the legal vertex range
    {
        // for numerical stable of the optimal location, we use average of xyz
        if (vtReplace[vj]>0) // vertex to be contracted to
        {
            for(int it=0;it<D;it++) // accumulate vertex XYZ
                vertexOut[D*vj+it] = (vertexOut[D*vj+it]+vertexIn[D*vj+it])/(vtReplace[vj]+1);
        }
        if (vtReplace[vj]==0) // left out vertex, forms singular cluster, copy from original
        {
            for(int it=0;it<D;it++)
                vertexOut[D*vj+it] = vertexIn[D*vj+it];
        }
    }
}


__global__ void labelDegenerateTriangles(const int B, const int D, const int* mfIn, const int* vtReplace,
                                         const float* vertexOut, const int* faceIn, int* faceOut,
                                         bool* isDegenerate, int* isKept, int* mfOut)
{
    int fcIdx = blockIdx.x*blockDim.x + threadIdx.x; // global face index in the batch

    int Nf = mfIn[B-1];
    if (fcIdx<Nf) // index must be in the legal range
    {
        // get sample index in the batch accessing by the current thread
        int batIdx;
        for(int it=0;it<B;it++)
        {
            if (fcIdx < mfIn[it])
            {
                batIdx = it;
                break;
            }
        }

        // old v1, v2, v3
        int v1 = faceIn[3*fcIdx];
        int v2 = faceIn[3*fcIdx+1];
        int v3 = faceIn[3*fcIdx+2];

        // new v1, v2, v3
        if (vtReplace[v1]<0) v1 = -vtReplace[v1];
        if (vtReplace[v2]<0) v2 = -vtReplace[v2];
        if (vtReplace[v3]<0) v3 = -vtReplace[v3];

        // update face list: vtReplace[.] is global vertex index in the batch BEFORE decimation
        // vtMap[.] is global vertex index in the batch AFTER decimation
        faceOut[3*fcIdx]   = v1; //vtMap[v1];
        faceOut[3*fcIdx+1] = v2; //vtMap[v2];
        faceOut[3*fcIdx+2] = v3; //vtMap[v3];

        if (v1==v2 || v1==v3 || v2==v3)
        {
            isDegenerate[fcIdx] = true;
            //isKept[v1] = 0; isKept[v2] = 0; isKept[v3] = 0;
        }
        else
        {
            atomicAdd(&mfOut[batIdx],1);
            isKept[v1] = 1; isKept[v2] = 1; isKept[v3] = 1;
        }

//        const float* xyz_v1 = &vertexOut[D*v1];
//        const float* xyz_v2 = &vertexOut[D*v2];
//        const float* xyz_v3 = &vertexOut[D*v3];
//        float D21[3] = {xyz_v2[0]-xyz_v1[0], xyz_v2[1]-xyz_v1[1], xyz_v2[2]-xyz_v1[2]};
//        float D31[3] = {xyz_v3[0]-xyz_v1[0], xyz_v3[1]-xyz_v1[1], xyz_v3[2]-xyz_v1[2]};
//        float D32[3] = {xyz_v3[0]-xyz_v2[0], xyz_v3[1]-xyz_v2[1], xyz_v3[2]-xyz_v2[2]};
//
//        float new_raw_normal[3]; // un-normalized normal
//        CrossProductDim3(D21, D31, new_raw_normal);
//        float Ln = sqrt(DotProductDim3(new_raw_normal, new_raw_normal)); // new_area = Ln/2;
//        float L[3] = { sqrt(DotProductDim3(D21, D21)),
//                       sqrt(DotProductDim3(D31, D31)),
//                       sqrt(DotProductDim3(D32, D32)) };
//
//        float temp = max(max(L[0]*L[1], max(L[0]*L[2], L[1]*L[2])), 1e-20f);
//        float min_sin_theta = Ln/temp;
//
//        if (min_sin_theta < minSinTheta) // minSinTheta=0.1392, e.g. 8 degree as threshold
//            isDegenerate[fcIdx] = true;
//        else
//            atomicAdd(&mfOut[batIdx],1);
    }
}

__global__ void sizeofOutput(const int B, const int* nvIn, const int* vtMap, int* nvOut)
{
    for(int it=0;it<B;it++)
    {
        if (it>0)
            nvOut[it] = vtMap[nvIn[it]-1] - vtMap[nvIn[it-1]-1];
        else
            nvOut[it] = vtMap[nvIn[it]-1];
    }
}

__global__ void getIOmap(int Nv, const int* isKept, int* vtMap)

{
    int vi = blockIdx.x*blockDim.x + threadIdx.x;

    if (vi < Nv)
    {
        if(isKept[vi]==1)
            vtMap[vi] -= 1; // index start from 0
        else
            vtMap[vi] = -1; // NOTE: vtReplace[vi]>=0 may also get vtMap[vi]=-1, because they are deleted in the vertex contraction
    }
}

__global__ void updateFaces(int Nf, const int* vtMap, const bool* isDegenerate, int* faceOut)
{
    int fcIdx = blockIdx.x*blockDim.x + threadIdx.x; // global face index in the batch

    if (fcIdx<Nf && !isDegenerate[fcIdx]) // index must be in the legal range
    {
        // old (v1, v2, v3): indices in the input
        int v1 = faceOut[3*fcIdx];
        int v2 = faceOut[3*fcIdx+1];
        int v3 = faceOut[3*fcIdx+2];

        // new (v1, v2, v3): indices in the output
        faceOut[3*fcIdx]   = vtMap[v1];
        faceOut[3*fcIdx+1] = vtMap[v2];
        faceOut[3*fcIdx+2] = vtMap[v3];
    }
}


__host__ void clusterVertices(const int Ne, const int uniqueNe, const int vtNum, const int startIdx,
                              const int* edgeOut, const int* edgeIdx, const int nv2Remove, int* vtReplace)
{
    int vtRemoved = 0;
    std::vector<char> Covered(vtNum,0);

    // vertex clustering of time complexity O(n)+O(n)
    for(int i=0;i<uniqueNe;i++)
    {
        int vi = edgeOut[edgeIdx[i]];
        int vj = edgeOut[edgeIdx[i]+Ne];

        if (vtRemoved>=nv2Remove)
            continue;

        if(Covered[vi-startIdx]==0 && Covered[vj-startIdx]==0) // both vertices of the edge are not covered
        {
            vtReplace[vi] = -vj; // negative contracted vertex index
            vtReplace[vj]++;     // cluster size except the point itself
            Covered[vi-startIdx] = 1;    // 1 for seed vertex
            Covered[vj-startIdx] = 1;    // 1 for seed vertex
            vtRemoved++;
        }
    }

    for(int i=0;i<uniqueNe;i++)
    {
        int vk1 = edgeOut[edgeIdx[i]];
        int vk2 = edgeOut[edgeIdx[i]+Ne];

        if (vtRemoved>=nv2Remove)
            continue;

        if(Covered[vk1-startIdx]>0 && Covered[vk2-startIdx]>0) // both vertices of the edge are covered
            continue;

        if (Covered[vk2-startIdx]==1)  // only `vk2' is in seed vertex pair
        {
            // swap `vk1' and `vk2'
            int temp = vk1;
            vk1 = vk2;
            vk2 = temp;
        }
        if (vtReplace[vk1]<0)
            vk1 = -vtReplace[vk1];

        vtReplace[vk2] = -vk1;
        vtReplace[vk1]++;
        if (vtReplace[vk1]<0)
            std::cout<<"error clustering.\n";
        Covered[vk2-startIdx] = 2; // 2 for non-seed vertex
        vtRemoved++;
    }
}


// NOTE: each row of array 'planIn' is composed of [normal=[nx,ny,nz],intercept=d,area]
void meshDecimationLauncher(const bool useArea, const float wgtBnd,     //hyperparams
                            const int B, const int D, const int Nv, const int Nf, const int* nvIn, const int* mfIn,  //inputs
                            const int* nv2Remove, const float* vertexIn, const int* faceIn, const float* planeIn,    //inputs
                            int* nvOut, int* mfOut, float* vertexOut, int* faceOut, int* vtReplace, int* vtMap,      //ouputs
                            bool* isDegenerate)
{
    // copy data from devide to host
    int* h_nvIn = new int[B];
    int* h_mfIn = new int[B];
    int* h_nv2Remove = new int[B];
    cudaMemcpy(h_nvIn, nvIn, B*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_mfIn, mfIn, B*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_nv2Remove, nv2Remove, B*sizeof(int), cudaMemcpyDeviceToHost);

    // Initialize per-vertex Quadric on GPU in parallel
    float* vertexQuadric;
    int numGrid = int(Nf/1024) + 1;  // Nf is the total number of faces in the batch
    cudaMalloc(&vertexQuadric, (10*Nv)*sizeof(float));
    cudaMemset(vertexQuadric, 0, (10*Nv)*sizeof(float)); // initialize all to zeros
    initVertexQuadrics<<<numGrid,1024>>>(Nf, useArea, faceIn, planeIn, vertexQuadric);
    //cudaDeviceSynchronize();

    // Extract edges from face list: resulted shape (3,Ne)
    int nRows = 3; // edge(v1,v2) + faceIdx: 3 integers
    int Ne = 3*Nf; // total number of edges in the batch
    int* edgeOut;
    cudaMalloc(&edgeOut, (nRows*Ne)*sizeof(int));
    extractEdges<<<numGrid,1024>>>(Nf, faceIn, edgeOut);
    //cudaDeviceSynchronize();

    // sort the edges
    thrust::device_vector<int> edgeIdx(Ne);
    thrust::sequence(edgeIdx.begin(), edgeIdx.end());
    thrust::sort(edgeIdx.begin(), edgeIdx.end(), edge_sort_func<int>(Ne, edgeOut));

    //add additional boundary Quadric
    if (wgtBnd>0)
    {
        numGrid = int(Ne/1024) + 1;
        addBoundaryQuadrics<<<numGrid,1024>>>(Ne, D, useArea, wgtBnd, vertexIn, planeIn, edgeOut,
                                              thrust::raw_pointer_cast(edgeIdx.data()), vertexQuadric);
        //cudaDeviceSynchronize();
    }

    int h_vtReplace[Nv] = {0};
    int* h_edgeOut = new int[2*Ne];
    cudaMemcpy(h_edgeOut, edgeOut, 2*Ne*sizeof(int), cudaMemcpyDeviceToHost);

    srand(time(0));  // setting seed of random_shuffle as current time

    int beginIdx=0, endIdx;
    for(int b=0;b<B;b++)
    {
        endIdx = 3*h_mfIn[b];

        // get unique edges using thrust:unique on GPU, which forms the candidate vertex pairs
        // IMPORTANT: edgeOut is still in size (3,Ne), it's just the first uniquNe columns
        // forms the unique edges
        int uniqueNe = thrust::unique(thrust::device, edgeIdx.data()+beginIdx, edgeIdx.data()+endIdx,
                                      edge_unique_func<int>(Ne, edgeOut)) - (edgeIdx.data()+beginIdx);

        // Compute Quadric cost of each unique edge
        float* edgeCost;
        numGrid = int(uniqueNe/1024) + 1;
        cudaMalloc(&edgeCost, uniqueNe*sizeof(float));
        computeEdgeCost<<<numGrid,1024>>>(uniqueNe, Ne, D, vertexIn, edgeOut,
                                          thrust::raw_pointer_cast(edgeIdx.data()+beginIdx),
                                          vertexQuadric, edgeCost);
        //cudaDeviceSynchronize();

        // sorting unique edges based on their cost using thrust::sort_by_key
        // edges of different samples will be mixedly ordered
        thrust::sort_by_key(thrust::device, edgeCost, edgeCost+uniqueNe, edgeIdx.data()+beginIdx);

        // copy from device to host h_edgeIdx
        int* h_edgeIdx = new int[uniqueNe];
        cudaMemcpy(h_edgeIdx, thrust::raw_pointer_cast(edgeIdx.data()+beginIdx),
                   uniqueNe*sizeof(int), cudaMemcpyDeviceToHost);

        // Vertex cluster: seed/disjoint pair generation on CPU with conditional-loops
        int prevNum = 0;
        if(b>0)   prevNum = h_nvIn[b-1];
        const int vtNum = h_nvIn[b] - prevNum;
        clusterVertices(Ne, uniqueNe, vtNum, prevNum, h_edgeOut, &h_edgeIdx[0],
                        h_nv2Remove[b], h_vtReplace);

        beginIdx = endIdx;

        cudaFree(edgeCost);
        delete[] h_edgeIdx;
    }
    cudaMemcpy(vtReplace, h_vtReplace, Nv*sizeof(int), cudaMemcpyHostToDevice);

    // Vertex cluster contraction on GPU in parallel, and compute nvOut
    numGrid = int(Nv/1024) + 1;
    VertexClusterQuadrics<<<numGrid,1024>>>(B, D, nvIn, vtReplace, vertexIn, vertexQuadric, vertexOut);
    VertexClusterContraction<<<numGrid,1024>>>(B, D, nvIn, vtReplace, vertexIn, vertexQuadric, vertexOut);
    //cudaDeviceSynchronize();

    // Label degenerate faces(w/o silver triangles), and compute mfOut
    int* isKept;
    cudaMalloc(&isKept, Nv*sizeof(int));
    cudaMemset(isKept,0,Nv*sizeof(int));
    numGrid = int(Nf/1024) + 1;
    labelDegenerateTriangles<<<numGrid,1024>>>(B, D, mfIn, vtReplace, vertexOut, faceIn, faceOut,
                                               isDegenerate, isKept, mfOut);
    //cudaDeviceSynchronize();

    // update vertex indices(e.g. vtMap), some of them are not existing any more because of degenerate faces
    cudaMemcpy(vtMap, isKept, Nv*sizeof(int), cudaMemcpyDeviceToDevice);
    thrust::inclusive_scan(thrust::device, vtMap, vtMap+Nv, vtMap);
    sizeofOutput<<<1,1>>>(B, nvIn, vtMap, nvOut);
    numGrid = int(Nv/1024) + 1;
    getIOmap<<<numGrid,1024>>>(Nv, isKept, vtMap);
    //cudaDeviceSynchronize();

    // update vertex indices of the triangles
    numGrid = int(Nf/1024) + 1;
    updateFaces<<<numGrid,1024>>>(Nf, vtMap, isDegenerate, faceOut);
    //cudaDeviceSynchronize();

    // free the cpu and gpu memory
    cudaFree(vertexQuadric);
    cudaFree(edgeOut);
    cudaFree(isKept);
    delete[] h_nvIn;
    delete[] h_mfIn;
    delete[] h_nv2Remove;
    delete[] h_edgeOut;
}


__global__  void update_repB(const int nvA, int* repA, int* mapA, int* repB)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx<nvA)
    {
        int vi = idx;
        int vo = mapA[vi];
        if (vo>=0 && repB[vo]>=0)
            repB[vo] = vi;
    }
}
__global__ void update_repA(const int nvA, int* repA, int* mapA, int* repB)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx<nvA)
    {
        int vi = idx;
        if (repA[vi]<0)
            vi = -repA[vi];

        int vo = mapA[vi];
        if (vo>=0 && repB[vo]<0)
        {
            vo = -repB[vo];
            repA[idx] = -repB[vo];
            if (repB[vo]<0)
                printf("update_repA Error: repB[vo]=%d<0!\n", repB[vo]);
            atomicAdd(&repA[repB[vo]],1);
        }
    }
}
__global__ void update_mapA(const int nvA, int* repA, int* mapA, int* repB, const int* mapB)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx<nvA)
    {
        int vi = idx;
        int vo = mapA[vi];
        if (vo>=0)
            mapA[vi] = mapB[vo];
        else
            mapA[vi] = -1;
    }
}
void combineClustersLauncher(const int nvA, const int nvB, const int* repA, const int* mapA,
                             const int* repB, const int* mapB, int* repOut, int* mapOut)
{
    cudaMemcpy(repOut, repA, nvA*sizeof(int), cudaMemcpyDeviceToDevice);
    cudaMemcpy(mapOut, mapA, nvA*sizeof(int), cudaMemcpyDeviceToDevice);

    int* repBcopy;
    cudaMalloc(&repBcopy, nvB*sizeof(int));
    cudaMemcpy(repBcopy, repB, nvB*sizeof(int), cudaMemcpyDeviceToDevice);

    int numGrid = int(nvA/1024) + 1;
    update_repB<<<numGrid,1024>>>(nvA, repOut, mapOut, repBcopy);
    update_repA<<<numGrid,1024>>>(nvA, repOut, mapOut, repBcopy);
    update_mapA<<<numGrid,1024>>>(nvA, repOut, mapOut, repBcopy, mapB);

    cudaFree(repBcopy);
}


__global__ void count_vertex_adjface(int NfIn, const int* face, const int* vtMap, int* nfCount)
{
    int fcIdx = blockIdx.x*blockDim.x + threadIdx.x;

    if (fcIdx < NfIn)
    {
        int v[3] = {face[fcIdx*3], face[fcIdx*3+1], face[fcIdx*3+2]};

        for(int k=0;k<3;k++)
        {
            int vi = v[k];
            int vo = vtMap[vi];
            if (vo>=0)
            {
                atomicAdd(&nfCount[vo],1);
            }
        }
    }
}
void countVertexAdjfaceLauncher(int NfIn, const int* face, const int* vtMap, int* nfCount)
{
    int numGrid = int(NfIn/1024) + 1;
    count_vertex_adjface<<<numGrid,1024>>>(NfIn, face, vtMap, nfCount);
}



