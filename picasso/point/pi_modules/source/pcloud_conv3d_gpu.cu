//#include <thrust/host_vector.h>

// nnCount: concat_Mp;
// nnIndex: Nout*2;
// input:   concat_Np*C;
// filter:  filter_size*C*r;
// output:  concat_Mp*(C*r);
__global__ void depthwise_conv3d_forward(int Np, int Mp, int Nout, int F, int C, int r, const int* nnCount,
                                         const int* nnIndex, const int* binIndex, const float* input,
                                         const float* filter, float* output)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x; // global  global index
    int pairIdx = idx/(C*r); // extract neighbor pair ID
    if (pairIdx<Nout) // index must be in the legal range
    {
        int cout   = idx%(C*r);    // output channel ID
        int cin    = cout/r;       // input channel ID
        int outIdx = nnIndex[pairIdx*2];
        int inIdx  = nnIndex[pairIdx*2+1];
        int nnSize = nnCount[outIdx];
        int f      = binIndex[pairIdx];

        atomicAdd(&output[outIdx*C*r+cout], input[inIdx*C+cin]*filter[f*C*r+cout]/nnSize);
    }
}


__global__ void depthwise_input_backward(int Np, int Mp, int Nout, int F, int C, int r, const int* nnCount,
                                         const int* nnIndex, const int* binIndex, const float* input,
                                         const float* filter, const float* gradOutput, float* gradInput)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x; // global index
    int pairIdx = idx/(C*r); // extract neighbor pair ID
    if (pairIdx<Nout) // index must be in the legal range
    {
        int cout   = idx%(C*r);    // output channel ID
        int cin    = cout/r;       // input channel ID
        int outIdx = nnIndex[pairIdx*2];
        int inIdx  = nnIndex[pairIdx*2+1];
        int nnSize = nnCount[outIdx];
        int f      = binIndex[pairIdx];

        float derIn = gradOutput[outIdx*C*r+cout]*filter[f*C*r+cout]/nnSize;
        atomicAdd(&gradInput[inIdx*C+cin],derIn);
    }
}


__global__ void depthwise_filter_backward(int Np, int Mp, int Nout, int F, int C, int r, const int* nnCount,
                                          const int* nnIndex, const int* binIndex, const float* input,
                                          const float* gradOutput, float* gradFilter, int sharedMemSize,
                                          int startIdx)
{
    extern __shared__ float gradPerBlock[]; // the gradient on each block
    for (int i=threadIdx.x;i<sharedMemSize;i+=blockDim.x)
    {
        gradPerBlock[i] = 0; // for 1D block
    }
    __syncthreads();

    int endIdx = sharedMemSize+startIdx;
    int idx = blockIdx.x*blockDim.x + threadIdx.x; //  global index
    int pairIdx = idx/(C*r); // extract neighbor pair ID
    if (pairIdx<Nout) // index must be in the legal range
    {
        int cout   = idx%(C*r);    // output channel ID
        int cin    = cout/r;       // input channel ID
        int outIdx = nnIndex[pairIdx*2];
        int inIdx  = nnIndex[pairIdx*2+1];
        int nnSize = nnCount[outIdx];
        int f      = binIndex[pairIdx];

        int currIdx = f*C*r+cout;
        float derFilt = gradOutput[outIdx*C*r+cout]*input[inIdx*C+cin]/nnSize;
        if((currIdx>=startIdx) && (currIdx<endIdx)) // within the shared memory
        {
            atomicAdd(&gradPerBlock[currIdx-startIdx],derFilt);
        }
    }
    __syncthreads();

    for (int i=threadIdx.x;i<sharedMemSize;i+=blockDim.x)
    {
        atomicAdd(&gradFilter[i+startIdx],gradPerBlock[i]); // for 1D block
    }
}


// nnIndex: B*M*K;
// nnCount: B*M;
// input:   B*N*C;
// filter:  filter_size*C*r;
// output:  B*M*(C*r)
__global__ void fuzzy_depthwise_conv3d_forward(int Np, int Mp, int Nout, int F, int C, int r, const int T, const int* nnCount,
                                               const int* nnIndex,const int* binIndex, const float* binCoeff,
                                               const float* input, const float* filter, float* output)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x; // global index
    int pairIdx = idx/(C*r); // extract neighbor pair ID
    if (pairIdx<Nout) // index must be in the legal range
    {
        int cout   = idx%(C*r);    // output channel ID
        int cin    = cout/r;       // input channel ID
        int outIdx = nnIndex[pairIdx*2];
        int inIdx  = nnIndex[pairIdx*2+1];
        int nnSize = nnCount[outIdx];

        float weight = 0;
        for(int fid=0;fid<T;fid++)
        {
            int f       = binIndex[pairIdx*T+fid];
            float coeff = binCoeff[pairIdx*T+fid];

            weight += coeff*filter[f*C*r+cout];
        }
        atomicAdd(&output[outIdx*C*r+cout], input[inIdx*C+cin]*weight/nnSize);
    }
}


__global__ void fuzzy_depthwise_input_backward(int Np, int Mp, int Nout, int F, int C, int r, const int T,
                                               const int* nnCount, const int* nnIndex, const int* binIndex,
                                               const float* binCoeff, const float* input, const float* filter,
                                               const float* gradOutput, float* gradInput)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x; // global index
    int pairIdx = idx/(C*r); // extract neighbor pair ID

    if (pairIdx<Nout) // index must be in the legal range
    {
        int cout   = idx%(C*r);    // output channel ID
        int cin    = cout/r;       // input channel ID
        int outIdx = nnIndex[pairIdx*2];
        int inIdx  = nnIndex[pairIdx*2+1];
        int nnSize = nnCount[outIdx];

        float weight = 0;
        for(int fid=0;fid<T;fid++)
        {
            int f       = binIndex[pairIdx*T+fid];
            float coeff = binCoeff[pairIdx*T+fid];

            weight += coeff*filter[f*C*r+cout];
        }
        float derIn = gradOutput[outIdx*C*r+cout]*weight/nnSize;
        atomicAdd(&gradInput[inIdx*C+cin],derIn);
    }
}


__global__ void fuzzy_depthwise_filter_backward(int Np, int Mp, int Nout, int F, int C, int r, const int T,
                                                const int* nnCount, const int* nnIndex, const int* binIndex,
                                                const float* binCoeff, const float* input, const float* gradOutput,
                                                float* gradFilter, int sharedMemSize, int startIdx)
{
    extern __shared__ float gradPerBlock[]; // the gradient on each block
    for (int i=threadIdx.x;i<sharedMemSize;i+=blockDim.x)
    {
        gradPerBlock[i] = 0; // for 1D block
    }
    __syncthreads();

    int endIdx = sharedMemSize+startIdx;
    int idx = blockIdx.x*blockDim.x + threadIdx.x; // global index
    int pairIdx = idx/(C*r); // extract neighbor pair ID
    if (pairIdx<Nout) // index must be in the legal range
    {
        int cout   = idx%(C*r);    // output channel ID
        int cin    = cout/r;       // input channel ID
        int outIdx = nnIndex[pairIdx*2];
        int inIdx  = nnIndex[pairIdx*2+1];
        int nnSize = nnCount[outIdx];

        float derFilt = gradOutput[outIdx*C*r+cout]*input[inIdx*C+cin]/nnSize;
        for(int fid=0;fid<T;fid++)
        {
            int f       = binIndex[pairIdx*T+fid];
            float coeff = binCoeff[pairIdx*T+fid];

            int currIdx = f*C*r+cout;
            if((currIdx>=startIdx) && (currIdx<endIdx)) // within the shared memory
            {
                atomicAdd(&gradPerBlock[currIdx-startIdx],coeff*derFilt);
            }
        }
    }
    __syncthreads();

    for (int i=threadIdx.x;i<sharedMemSize;i+=blockDim.x)
    {
        atomicAdd(&gradFilter[i+startIdx],gradPerBlock[i]); // for 1D block
    }
}




void depthwiseConv3dLauncher(int Np, int Mp, int Nout, int F, int C, int r, const int* nnCount,
                             const int* nnIndex, const int* binIndex, const float* input,
                             const float* filter, float* output)
{
    int numGrid = int(Nout*C*r/1024) + 1;
    depthwise_conv3d_forward<<<numGrid,1024>>>(Np, Mp, Nout, F, C, r, nnCount, nnIndex, binIndex,
                                               input, filter, output);
}

void depthwiseConv3dGradLauncher(int Np, int Mp, int Nout, int F, int C, int r,
                                 const int* nnCount, const int* nnIndex, const int* binIndex,
                                 const float* input, const float* filter, const float* gradOutput,
                                 float* gradInput, float* gradFilter)
{
    int numGrid = int(Nout*C*r/1024) + 1;

    // titan xp has shared memory of 49152 bytes, each float value takes 4 bytes in the memory
    int maxSharedMemSize = int(49152/sizeof(float));

    depthwise_input_backward<<<numGrid,1024>>>(Np, Mp, Nout, F, C, r, nnCount, nnIndex, binIndex,
                                               input, filter, gradOutput, gradInput);

    int maxIter = (F*C*r)/maxSharedMemSize;
    int remainder = (F*C*r)%maxSharedMemSize;
    for(int iter=0;iter<maxIter;iter++)
    {
        depthwise_filter_backward<<<numGrid,1024,sizeof(float)*maxSharedMemSize>>>(Np, Mp, Nout, F, C, r, nnCount, nnIndex,
                                                                                   binIndex, input, gradOutput, gradFilter,
                                                                                   maxSharedMemSize, maxSharedMemSize*iter);
    }
    if(remainder>0) // fill the remainder
    {
        depthwise_filter_backward<<<numGrid,1024,sizeof(float)*remainder>>>(Np, Mp, Nout, F, C, r, nnCount, nnIndex,
                                                                            binIndex, input, gradOutput, gradFilter,
                                                                            remainder, maxSharedMemSize*maxIter);
    }
}


void fuzzyDepthwiseConv3dLauncher(int Np, int Mp, int Nout, int F, int C, int r, int T, const int* nnCount,
                                  const int* nnIndex, const int* binIndex, const float* binCoeff,
                                  const float* input, const float* filter, float* output)
{
    int numGrid = int(Nout*C*r/1024) + 1;
    fuzzy_depthwise_conv3d_forward<<<numGrid,1024>>>(Np, Mp, Nout, F, C, r, T, nnCount, nnIndex, binIndex,
                                                     binCoeff, input, filter, output);
}

void fuzzyDepthwiseConv3dGradLauncher(int Np, int Mp, int Nout, int F, int C, int r, int T, const int* nnCount,
                                      const int* nnIndex, const int* binIndex, const float* binCoeff,
                                      const float* input,  const float* filter, const float* gradOutput,
                                      float* gradInput, float* gradFilter)
{
    int numGrid = int(Nout*C*r/1024) + 1;

    // titan xp has shared memory of 49152 bytes, each float value takes 4 bytes in the memory
    int maxSharedMemSize = int(49152/sizeof(float));

    fuzzy_depthwise_input_backward<<<numGrid,1024>>>(Np, Mp, Nout, F, C, r, T, nnCount, nnIndex, binIndex,
                                                     binCoeff, input, filter, gradOutput, gradInput);

    int maxIter = (F*C*r)/maxSharedMemSize;
    int remainder = (F*C*r)%maxSharedMemSize;
    for(int iter=0;iter<maxIter;iter++)
    {
        fuzzy_depthwise_filter_backward<<<numGrid,1024,sizeof(float)*maxSharedMemSize>>>(Np, Mp, Nout, F, C, r, T,
                                                                           nnCount, nnIndex, binIndex, binCoeff,
                                                                           input, gradOutput, gradFilter,
                                                                           maxSharedMemSize, maxSharedMemSize*iter);

    }
    if(remainder>0) // fill the remainder
    {
        fuzzy_depthwise_filter_backward<<<numGrid,1024,sizeof(float)*remainder>>>(Np, Mp, Nout, F, C, r, T, nnCount,
                                                                              nnIndex, binIndex, binCoeff, input,
                                                                              gradOutput, gradFilter, remainder,
                                                                              maxSharedMemSize*maxIter);
    }
}