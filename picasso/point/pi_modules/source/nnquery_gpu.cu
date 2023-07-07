#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <thrust/device_vector.h>

#ifndef M_PI
#define M_PI           3.14159265358979323846F  /* pi */
#endif

typedef long int LLint;

struct point3d
{
    float x=0, y=0, z=0;
};

// database:   Np*3
// query:      Mp*3
// nvDatabase: B
// nvQuery:    B
// cntInfo:    Mp
__global__ void count_sphereNN(int B, int Np, int Mp, float radius, int nnSample, const float* database,
                               const float* query, const int* nvDatabase, const int* nvQuery, LLint* cntInfo)
{
    // get the neighbor indices
    point3d ptQuery, pt, delta;
    for(int i=blockIdx.x;i<B;i+=gridDim.x)
    {
        int startM=0, M=nvQuery[i];
        int startN=0, N=nvDatabase[i];
        if (i>0)
        {
            startM = nvQuery[i-1];
            M = nvQuery[i] - nvQuery[i-1];

            startN = nvDatabase[i-1];
            N = nvDatabase[i] - nvDatabase[i-1];
        }

        for(int j=threadIdx.x;j<M;j+=blockDim.x)
        {
            ptQuery.x = query[(startM+j)*3];
            ptQuery.y = query[(startM+j)*3+1];
            ptQuery.z = query[(startM+j)*3+2];

            int cnt=0; // to count the number of neighbors
            for(int k=0;k<N;k++)
            {
                pt.x = database[(startN+k)*3];
                pt.y = database[(startN+k)*3+1];
                pt.z = database[(startN+k)*3+2];

                delta.x = pt.x - ptQuery.x;
                delta.y = pt.y - ptQuery.y;
                delta.z = pt.z - ptQuery.z;

                float dist3D = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z; // squared 3D
                dist3D = sqrtf(dist3D); //sqrt

                if (dist3D<radius && cnt<nnSample) // find a neighbor in range
                    cnt++;
            }
            cntInfo[startM+j] = cnt;
        }
    }
}
// database:   Np*3
// query:      Mp*3
// nvDatabase: B
// nvQuery:    B
// cntInfo:    Mp
// nnIndex:    Nout*2
// nnDist:     Nout
__global__ void build_sphereNN_idx(int B, int Np, int Mp, float radius, const float* database,
                                   const float* query, const int* nvDatabase, const int* nvQuery,
                                   const LLint* cntInfo, int* nnIndex, float* nnDist)
{
    // get the neighbor indices
    point3d ptQuery, pt, delta;
    for(int i=blockIdx.x;i<B;i+=gridDim.x)
    {
        int startM=0, M=nvQuery[i];
        int startN=0, N=nvDatabase[i];
        if (i>0)
        {
            startM = nvQuery[i-1];
            M = nvQuery[i] - nvQuery[i-1];

            startN = nvDatabase[i-1];
            N = nvDatabase[i] - nvDatabase[i-1];
        }

        for(int j=threadIdx.x;j<M;j+=blockDim.x)
        {
            ptQuery.x = query[(startM+j)*3];
            ptQuery.y = query[(startM+j)*3+1];
            ptQuery.z = query[(startM+j)*3+2];

            LLint startCnt = 0;
            if ((startM+j)>0)
            {
                startCnt = cntInfo[startM+j-1]; // starting index of neighbors for (startM+j)
            }
            int MAX_CNT = int(cntInfo[startM+j] - startCnt);

            int cnt=0; // to count the number of neighbors
            for(int k=0;k<N;k++)
            {
                pt.x = database[(startN+k)*3];
                pt.y = database[(startN+k)*3+1];
                pt.z = database[(startN+k)*3+2];

                delta.x = pt.x - ptQuery.x;
                delta.y = pt.y - ptQuery.y;
                delta.z = pt.z - ptQuery.z;

                float dist3D = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z; // squared 3D
                dist3D = sqrtf(dist3D); //sqrt

                if (dist3D<radius && cnt<MAX_CNT) // find a neighbor in range
                {
                    nnIndex[(startCnt+cnt)*2] = (startM+j);
                    nnIndex[(startCnt+cnt)*2+1] = (startN+k);
                    nnDist[startCnt+cnt] = dist3D; // sqrt, not the squared one
                    cnt++;
                }
            }
        }
    }
}


// database:   Np*3
// query:      Mp*3
// nvDatabase: B
// nvQuery:    B
// nnCount:    Mp
__global__ void count_cubeNN(int B, int Np, int Mp, float length, int nnSample, const float* database,
                             const float* query, const int* nvDatabase, const int* nvQuery, LLint* cntInfo)
{
    // get the neighbor indices, and compute their indices in the filter/kernel bins
    point3d ptQuery, pt, delta;
    for(int i=blockIdx.x;i<B;i+=gridDim.x)
    {
        int startM=0, M=nvQuery[i];
        int startN=0, N=nvDatabase[i];
        if (i>0)
        {
            startM = nvQuery[i-1];
            M = nvQuery[i] - nvQuery[i-1];

            startN = nvDatabase[i-1];
            N = nvDatabase[i] - nvDatabase[i-1];
        }

        for(int j=threadIdx.x;j<M;j+=blockDim.x)
        {
            ptQuery.x = query[(startM+j)*3];
            ptQuery.y = query[(startM+j)*3+1];
            ptQuery.z = query[(startM+j)*3+2];

            int cnt=0; // to count the number of neighbors
            for(int k=0;k<N;k++)
            {
                pt.x = database[(startN+k)*3];
                pt.y = database[(startN+k)*3+1];
                pt.z = database[(startN+k)*3+2];

                delta.x = pt.x - ptQuery.x;
                delta.y = pt.y - ptQuery.y;
                delta.z = pt.z - ptQuery.z;

                if (abs(delta.x)<length/2 && abs(delta.y)<length/2 && abs(delta.z)<length/2 && cnt<nnSample)
                    cnt++;
            }
            cntInfo[startM+j] = cnt;
        }
    }
}
// database:   Np*3
// query:      Mp*3
// nvDatabase: B
// nvQuery:    B
// cntInfo:    Mp
// nnIndex:    Nout*2*2
__global__ void build_cubeNN_idx(int B, int Np, int Mp, float length, int gridSize,
                                 const float* database, const float* query, const int* nvDatabase,
                                 const int* nvQuery, const LLint* cntInfo, int* nnIndex)
{
    // get the neighbor indices, and compute their indices in the filter/kernel bins
    point3d ptQuery, pt, delta;
    for(int i=blockIdx.x;i<B;i+=gridDim.x)
    {
        int startM=0, M=nvQuery[i];
        int startN=0, N=nvDatabase[i];
        if (i>0)
        {
            startM = nvQuery[i-1];
            M = nvQuery[i] - nvQuery[i-1];

            startN = nvDatabase[i-1];
            N = nvDatabase[i] - nvDatabase[i-1];
        }

        for(int j=threadIdx.x;j<M;j+=blockDim.x)
        {
            ptQuery.x = query[(startM+j)*3];
            ptQuery.y = query[(startM+j)*3+1];
            ptQuery.z = query[(startM+j)*3+2];

            LLint startCnt = 0;
            if ((startM+j)>0)
            {
                startCnt = cntInfo[startM+j-1]; // starting index of neighbors for (startM+j)
            }
            int MAX_CNT = int(cntInfo[startM+j] - startCnt);

            int cnt=0; // to count the number of neighbors
            for(int k=0;k<N;k++)
            {
                pt.x = database[(startN+k)*3];
                pt.y = database[(startN+k)*3+1];
                pt.z = database[(startN+k)*3+2];

                delta.x = pt.x - ptQuery.x;
                delta.y = pt.y - ptQuery.y;
                delta.z = pt.z - ptQuery.z;

                if (abs(delta.x)<length/2 && abs(delta.y)<length/2 && abs(delta.z)<length/2 && cnt<MAX_CNT)
                {
                    // calculate bin index in the cubic filter/kernel
                    int xId = (delta.x + length/2)/(length/gridSize); //[0, gridSize)
                    int yId = (delta.y + length/2)/(length/gridSize); //[0, gridSize)
                    int zId = (delta.z + length/2)/(length/gridSize); //[0, gridSize)
                    int binID = xId*gridSize*gridSize + yId*gridSize + zId;

                    nnIndex[(startCnt+cnt)*3] = (startM+j);
                    nnIndex[(startCnt+cnt)*3+1] = (startN+k);
                    nnIndex[(startCnt+cnt)*3+2] = binID;

                    cnt++;
                }
            }
        }
    }
}

/* This part's code is based on the 3nn weighted interpolation of PointNet++
   of Charles R. Qi.
*/
// database:   Np*3
// query:      Mp*3
// nvDatabase: B
// nvQuery:    B
// nnIndex:    Mp*nnOut
// nnDist:     Mp*nnOut
__global__ void build_NN3(int B, int Np, int Mp, const float* database, const float* query,
                          const int* nvDatabase, const int* nvQuery, int* nnIndex, float* nnDist)
{
    // get the neighbor indices
    point3d ptQuery, pt, delta;
    for(int i=blockIdx.x;i<B;i+=gridDim.x)
    {
        int startM=0, M=nvQuery[i];
        int startN=0, N=nvDatabase[i];
        if (i>0)
        {
            startM = nvQuery[i-1];
            M = nvQuery[i] - nvQuery[i-1];

            startN = nvDatabase[i-1];
            N = nvDatabase[i] - nvDatabase[i-1];
        }

        for(int j=threadIdx.x;j<M;j+=blockDim.x)
        {
            ptQuery.x = query[(startM+j)*3];
            ptQuery.y = query[(startM+j)*3+1];
            ptQuery.z = query[(startM+j)*3+2];

            float best1=1e40, best2=1e40, best3=1e40;
            int   besti1=0, besti2=0, besti3=0;

            for(int k=0;k<N;k++)
            {
                pt.x = database[(startN+k)*3];
                pt.y = database[(startN+k)*3+1];
                pt.z = database[(startN+k)*3+2];

                delta.x = pt.x - ptQuery.x;
                delta.y = pt.y - ptQuery.y;
                delta.z = pt.z - ptQuery.z;

                float dist3D = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z; // squared 3D

                if (dist3D<best1) { // This is from PointNet++
                    best3=best2;
                    besti3=besti2;
                    best2=best1;
                    besti2=besti1;
                    best1=dist3D;
                    besti1=k;
                } else if (dist3D<best2) {
                    best3=best2;
                    besti3=besti2;
                    best2=dist3D;
                    besti2=k;
                } else if (dist3D<best3) {
                    best3=dist3D;
                    besti3=k;
                }

                nnIndex[(startM+j)*3]   = (startN+besti1);
                nnIndex[(startM+j)*3+1] = (startN+besti2);
                nnIndex[(startM+j)*3+2] = (startN+besti3);
                nnDist[(startM+j)*3]    = sqrtf(best1); // sqrt, not the squared one
                nnDist[(startM+j)*3+1]  = sqrtf(best2); // sqrt, not the squared one
                nnDist[(startM+j)*3+2]  = sqrtf(best3); // sqrt, not the squared one
            }
        }
    }
}



LLint countSphereNeighborLauncher(int B, int Np, int Mp, float radius, int nnSample, const float* database,
                                const float* query, const int* nvDatabase, const int* nvQuery, LLint* cntInfo)
{
    count_sphereNN<<<B,1024>>>(B, Np, Mp, radius, nnSample, database, query, nvDatabase, nvQuery, cntInfo);
    thrust::device_ptr<LLint> dev_ptr(cntInfo);
    thrust::inclusive_scan(dev_ptr, dev_ptr+Mp, dev_ptr); // in-place scan

    LLint Nout;
    cudaMemcpy(&Nout, cntInfo+(Mp-1), sizeof(LLint), cudaMemcpyDeviceToHost);
    return Nout;
}
void buildSphereNeighborLauncher(int B, int Np, int Mp, float radius, const float* database,
                                 const float* query, const int* nvDatabase, const int* nvQuery,
                                 const LLint* cntInfo, int* nnIndex, float* nnDist)
{
    build_sphereNN_idx<<<B,1024>>>(B, Np, Mp, radius, database, query, nvDatabase, nvQuery, cntInfo, nnIndex, nnDist);
}


LLint countCubeNeighborLauncher(int B, int Np, int Mp, float length, int nnSample, const float* database,
                              const float* query, const int* nvDatabase, const int* nvQuery, LLint* cntInfo)
{
    count_cubeNN<<<B,1024>>>(B, Np, Mp, length, nnSample, database, query, nvDatabase, nvQuery, cntInfo);
    thrust::device_ptr<LLint> dev_ptr(cntInfo);
    thrust::inclusive_scan(dev_ptr, dev_ptr+Mp, dev_ptr); // in-place scan

    LLint Nout;
    cudaMemcpy(&Nout, cntInfo+(Mp-1), sizeof(LLint), cudaMemcpyDeviceToHost);
    return Nout;
}
void buildCubeNeighborLauncher(int B, int Np, int Mp, float length, int gridSize, const float* database,
                               const float* query, const int* nvDatabase, const int* nvQuery,
                               const LLint* cntInfo, int* nnIndex)
{
    build_cubeNN_idx<<<B,1024>>>(B, Np, Mp, length, gridSize, database, query, nvDatabase, nvQuery, cntInfo, nnIndex);
    return;
}

void buildNearestNeighborLauncher(int B, int Np, int Mp, const float* database, const float* query,
                                  const int* nvDatabase, const int* nvQuery, int* nnIndex, float* nnDist)
{
    build_NN3<<<B,1024>>>(B, Np, Mp, database, query, nvDatabase, nvQuery, nnIndex, nnDist);
    return;
}