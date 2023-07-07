__global__ void weighted_interpolate_forward(int Np, int Mp, int C, int K, const int* nnIndex,
                                             const float* input, const float* weight, float* output)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x; // global index
    int outIdx = idx/C; // extract neighbor pair ID

    if (outIdx<Np) // index must be in the legal range
    {
        int ch     = idx%C;    // output channel ID

        for(int k=0;k<K;k++)
        {
            int inIdx = nnIndex[outIdx*K+k];
            float wgt = weight[outIdx*K+k];
            output[outIdx*C+ch] += input[inIdx*C+ch]*wgt;
        }
    }
}


__global__ void weighted_interpolate_backward(int Np, int Mp, int C, int K, const int* nnIndex,
                                              const float* gradOutput, const float* weight, float* gradInput)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x; // global index
    int outIdx = idx/C; // extract neighbor pair ID

    if (outIdx<Np) // index must be in the legal range
    {
        int ch     = idx%C;    // output channel ID

        for(int k=0;k<K;k++)
        {
            int inIdx = nnIndex[outIdx*K+k];
            float wgt = weight[outIdx*K+k];
            atomicAdd(&gradInput[inIdx*C+ch],gradOutput[outIdx*C+ch]*wgt);
        }
    }
}


void weightedInterpolateLauncher(int Np, int Mp, int C, int K, const int* nnIndex,
                                 const float* input, const float* weight, float* output)
{
    int numGrid = int(Np*C/1024)+1;
    weighted_interpolate_forward<<<numGrid,1024>>>(Np, Mp, C, K, nnIndex, input, weight, output);
}

void weightedInterpolateGradLauncher(int Np, int Mp, int C, int K, const int* nnIndex,
                                     const float* gradOutput, const float* weight, float* gradInput)
{
    int numGrid = int(Np*C/1024)+1;
    weighted_interpolate_backward<<<numGrid,1024>>>(Np, Mp, C, K, nnIndex, gradOutput, weight, gradInput);
}