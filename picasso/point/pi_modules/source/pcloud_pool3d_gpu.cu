#define myEPS 1e-16F

// reference: https://stackoverflow.com/questions/17399119/cant-we-use-atomic-operations-for-floating-point-variables-in-cuda
__device__ static float atomicMax(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

// nnCount: Mp;
// nnIndex: Nout*2;
// input:   Np*C;
// output:  Mp*C (M<N)
__global__ void max_pool3d_init(int Nout, int C, const int* nnIndex, const float* input, float* output)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x; // global index
    int pairIdx = idx/C; // extract neighbor pair ID

    if (pairIdx<Nout) // index must be in the legal range
    {
        int ch     = idx%C;    // channel ID
        int outIdx = nnIndex[pairIdx*2];
        int inIdx  = nnIndex[pairIdx*2+1];

        if (abs(output[outIdx*C+ch])<myEPS)
        {
            output[outIdx*C+ch] = input[inIdx*C+ch];
        }
    }
}

// nnCount: Mp;
// nnIndex: Nout*2;
// input:   Np*C;
// output:  Mp*C (M<N)
__global__ void max_pool3d_forward(int Nout, int C, const int* nnIndex, const float* input, float* output)

{
    int idx = blockIdx.x*blockDim.x + threadIdx.x; // global index
    int pairIdx = idx/C; // extract neighbor pair ID

    if (pairIdx<Nout) // index must be in the legal range
    {
        int ch     = idx%C;    // channel ID
        int outIdx = nnIndex[pairIdx*2];
        int inIdx  = nnIndex[pairIdx*2+1];

        atomicMax(&output[outIdx*C+ch],input[inIdx*C+ch]);    // max pooling
    }
}


// nnCount: Mp;
// nnIndex: Nout*2;
// input:   Np*C;
// output:  Mp*C (M<N)
__global__ void max_pool3d_index(int Nout, int C, const int* nnIndex,
                                 const float* input, const float* output, int* maxIndex)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x; // global index
    int pairIdx = idx/C; // extract neighbor pair ID

    if (pairIdx<Nout) // index must be in the legal range
    {
        int ch     = idx%C;    // channel ID
        int outIdx = nnIndex[pairIdx*2];
        int inIdx  = nnIndex[pairIdx*2+1];

        if (abs(input[inIdx*C+ch]-output[outIdx*C+ch])<myEPS) // get argmax
            maxIndex[outIdx*C+ch] = inIdx;
    }
}


// maxIndex: Mp*C, indices of the maximum feature point
__global__ void max_pool3d_backward(int Mp, int C, const int* maxIndex,
                                    const float* gradOutput, float* gradInput)

{
    int idx = blockIdx.x*blockDim.x + threadIdx.x; // global index
    int outIdx = idx/C; // extract neighbor pair ID

    if (outIdx < Mp) // index must be in the legal range
    {
        int ch    = idx%C;    // channel ID
        int inIdx = maxIndex[outIdx*C+ch];

        atomicAdd(&gradInput[inIdx*C+ch],gradOutput[outIdx*C+ch]);
    }
}


__global__ void avg_pool3d_forward(int Nout, int C, const int* nnCount,
                                   const int* nnIndex, const float* input, float* output)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x; // global index
    int pairIdx = idx/C; // extract neighbor pair ID

    if (pairIdx<Nout) // index must be in the legal range
    {
        int ch     = idx%C;    // channel ID
        int outIdx = nnIndex[pairIdx*2];
        int inIdx  = nnIndex[pairIdx*2+1];
        int nnSize = nnCount[outIdx];

        atomicAdd(&output[outIdx*C+ch], input[inIdx*C+ch]/nnSize);
    }
}


__global__ void avg_pool3d_backward(int Nout, int C, const int* nnCount, const int* nnIndex,
                                    const float* gradOutput, float* gradInput)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x; // global index
    int pairIdx = idx/C; // extract neighbor pair ID

    if (pairIdx<Nout) // index must be in the legal range
    {
        int ch     = idx%C;    // channel ID
        int outIdx = nnIndex[pairIdx*2];
        int inIdx  = nnIndex[pairIdx*2+1];
        int nnSize = nnCount[outIdx];

        atomicAdd(&gradInput[inIdx*C+ch],gradOutput[outIdx*C+ch]/nnSize);
    }
}


void maxPool3dLauncher(int Np, int Mp, int Nout, int C, const int* nnCount, const int* nnIndex,
                       const float* input, float* output, int* maxIndex)
{
    int numGrid = int(Nout*C/1024) + 1;
    max_pool3d_init   <<<numGrid,1024>>>(Nout, C, nnIndex, input, output);
    max_pool3d_forward<<<numGrid,1024>>>(Nout, C, nnIndex, input, output);
    max_pool3d_index  <<<numGrid,1024>>>(Nout, C, nnIndex, input, output, maxIndex);
}

void maxPool3dGradLauncher(int Np, int Mp, int C, const int* maxIndex, const float* gradOutput, float* gradInput)
{
    int numGrid = int(Mp*C/1024) + 1;
    max_pool3d_backward<<<numGrid,1024>>>(Mp, C, maxIndex, gradOutput, gradInput);
}

void avgPool3dLauncher(int Np, int Mp, int Nout, int C, const int* nnCount, const int* nnIndex,
                       const float* input, float* output)
{
    int numGrid = int(Nout*C/1024) + 1;
    avg_pool3d_forward<<<numGrid,1024>>>(Nout, C, nnCount, nnIndex, input, output);
}

void avgPool3dGradLauncher(int Np, int Mp, int Nout, int C, const int* nnCount, const int* nnIndex,
                           const float* gradOutput, float* gradInput)
{
    int numGrid = int(Nout*C/1024) + 1;
    avg_pool3d_backward<<<numGrid,1024>>>(Nout, C, nnCount, nnIndex, gradOutput, gradInput);
}