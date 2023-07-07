/* Furthest point sampling GPU implementation
 * Original author: Haoqiang Fan
 * Modified by Charles R. Qi
 * All Rights Reserved. 2017. 
 */

//#include <thrust/scan.h>
//#include <thrust/device_vector.h>
//#include <thrust/host_vector.h>

__global__ void farthest_point_sample(int B, int Np, const int* nvIn, const int* nvOut,
                                      const float* xyzIn, float* temp, int* indexOut)
{
    const int BlockSize=1024;
    __shared__ float dists[BlockSize];
    __shared__ int dists_i[BlockSize];
    const int BufferSize=3072;
    __shared__ float buf[BufferSize*3];

    for (int i=blockIdx.x;i<B;i+=gridDim.x)
    {
        // get (startM, M) and (startN, N);
        int startM=0, M=nvOut[i];
        int startN=0, N=nvIn[i];
        if (i>0)
        {
            startM = nvOut[i-1];
            M = nvOut[i] - nvOut[i-1];

            startN = nvIn[i-1];
            N = nvIn[i] - nvIn[i-1];
        }

        int old = startN;
        if (threadIdx.x==0)
            indexOut[startM] = old;
        for (int j=threadIdx.x;j<N;j+=blockDim.x)
        {
            temp[startN+j]=1e38;
        }
        for (int j=threadIdx.x;j<min(BufferSize,N)*3;j+=blockDim.x)
        {
            buf[j] = xyzIn[startN*3+j]; // copy as many as xyz_in to buffer
        }
        __syncthreads();

        for (int j=1;j<M;j++){
            int besti=0;
            float best=-1;
            float x1,y1,z1;
            x1=xyzIn[old*3+0];
            y1=xyzIn[old*3+1];
            z1=xyzIn[old*3+2];
            for (int k=threadIdx.x;k<N;k+=blockDim.x)
            {
                float td=temp[startN+k];
                float x2,y2,z2;
                if (k<BufferSize){
                    x2=buf[k*3+0];
                    y2=buf[k*3+1];
                    z2=buf[k*3+2];
                }else{
                    x2=xyzIn[startN*3+k*3+0];
                    y2=xyzIn[startN*3+k*3+1];
                    z2=xyzIn[startN*3+k*3+2];
                }
                float d=(x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1);
                float d2=min(d,td);
                if (d2!=td)
                    temp[startN+k]=d2;
                if (d2>best){
                    best=d2;
                    besti=startN+k;
                }
            }

            dists[threadIdx.x]=best;
            dists_i[threadIdx.x]=besti;
            for (int u=0;(1<<u)<blockDim.x;u++)
            {
                __syncthreads();
                if (threadIdx.x<(blockDim.x>>(u+1)))
                {
                    int i1=(threadIdx.x*2)<<u;
                    int i2=(threadIdx.x*2+1)<<u;
                    if (dists[i1]<dists[i2])
                    {
                        dists[i1]=dists[i2];
                        dists_i[i1]=dists_i[i2];
                    }
                }
            }
            __syncthreads();
            old=dists_i[0];
            if (threadIdx.x==0)
                indexOut[startM+j]=old;
        }
    }
}


int computeOutputSize(int B, const int* nvOut)
{
    int Mp;
    cudaMemcpy(&Mp, nvOut+(B-1), sizeof(int), cudaMemcpyDeviceToHost);

    return Mp;
}

void farthestPointSampleLauncher(int B, int Np, const int* nvIn, const int* nvOut,
                                 const float* xyzIn, float* temp, int* indexOut)
{
    farthest_point_sample<<<B,1024>>>(B, Np, nvIn, nvOut, xyzIn, temp, indexOut);
}


