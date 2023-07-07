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

// vtReplace: nvIn;
// vtReplace: nvIn;
// input:     nvIn*C;
// output:    nvOut*C;
__global__ void max_pool3d_init(const int nvIn, const int C, const int* vtReplace, const int* vtMap,
                                      const float* input, float* output)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x; // thread index
    int vi  = idx/C; // global vertex index
    int ch  = idx%C; // which feature channel

    if(vi<nvIn && vtMap[vi]>=0) // within the legal input/output range
    {
        int vo = vtMap[vi]; // mapping from input index to the output index
        output[C*vo+ch] = input[C*vi+ch];
    }
}

// vtReplace: nvIn;
// vtReplace: nvIn;
// input:     nvIn*C;
// output:    nvOut*C;
// maxIndex:  nvOut*C;
__global__ void max_pool3d_forward(const int nvIn, const int C, const int* vtReplace, const int* vtMap,
                                   const float* input, float* output)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x; // thread index
    int vi  = idx/C; // global vertex index
    int ch  = idx%C; // which feature channel

    if(vi < nvIn) // within the legal range
    {
        int mapid = vi;
        if (vtReplace[vi]<0)
            mapid = -vtReplace[vi];

        if (vtMap[mapid]>=0)  // valid output
        {
            int vo = vtMap[mapid]; // mapping from input index to the output index
            atomicMax(&output[C*vo+ch],input[C*vi+ch]); // max pooling
        }
    }
}

// vtReplace: nvIn;
// vtReplace: nvIn;
// input:     nvIn*C;
// output:    nvOut*C;
// maxIndex:  nvOut*C;
__global__ void max_pool3d_index(const int nvIn, const int C, const int* vtReplace, const int* vtMap,
                                 const float* input, float* output, int* maxIndex)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x; // thread index
    int vi  = idx/C; // global vertex index
    int ch  = idx%C; // which feature channel

    if(vi < nvIn) // within the legal range
    {
        int mapid = vi;
        if (vtReplace[vi]<0)
            mapid = -vtReplace[vi];

        if (vtMap[mapid]>=0)  // valid output
        {
            int vo = vtMap[mapid]; // mapping from input index to the output index
            if (abs(input[C*vi+ch]-output[C*vo+ch])<myEPS) // get argmax
                maxIndex[C*vo+ch] = vi;
        }
    }
}

// maxIndex:   nvOut*C;
// gradOutput: nvOut*C;
// gradInput:  nvIn*C;
__global__ void max_pool3d_backward(const int nvOut, const int C, const int* maxIndex,
                                    const float* gradOutput, float* gradInput)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x; // thread index
    int vo  = idx/C; // global vertex index
    int ch  = idx%C; // which feature channel

    if(vo < nvOut) // within the legal range
    {
        int vi = maxIndex[C*vo+ch];
        gradInput[C*vi+ch] = gradOutput[C*vo+ch];
    }
}


// vtReplace: nvIn;
// vtReplace: nvIn;
// input:     nvIn*C;
// output:    nvOut*C;
__global__ void avg_pool3d_forward(const int nvIn, const int C, const int* vtReplace,
                                   const int* vtMap, const float* input, float* output)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x; // thread index
    int vi  = idx/C; // global vertex index
    int ch  = idx%C; // which feature channel

    if(vi < nvIn)
    {
        int mapid = vi;
        if (vtReplace[vi]<0)
            mapid = -vtReplace[vi];

        if (vtMap[mapid]>=0)
        {
            int vo = vtMap[mapid]; // mapping from input index to the output index
            int count = vtReplace[mapid]+1; // check for `decimation' folder for the details about vtReplace
            atomicAdd(&output[C*vo+ch],input[C*vi+ch]/count); // output should be initialized to zeros
        }
    }
}

// vtReplace:  nvIn;
// vtReplace:  nvIn;
// gradOutput: nvOut*C;
// gradInput:  nvIn*C;
__global__ void avg_pool3d_backward(const int nvIn, const int C, const int* vtReplace,
                                    const int* vtMap, const float* gradOutput, float* gradInput)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x; // thread index
    int vi  = idx/C; // global vertex index of the input
    int ch  = idx%C; // which feature channel

    if(vi < nvIn)
    {
        int mapid = vi;
        if (vtReplace[vi]<0)
            mapid = -vtReplace[vi];

        if (vtMap[mapid]>=0)
        {
            int vo = vtMap[mapid]; // mapping from input index to the output index
            int count = vtReplace[mapid]+1; // check for `decimation' folder for the details about vtReplace
            gradInput[C*vi+ch] = gradOutput[C*vo+ch]/count;
        }
    }
}



void maxPool3dLauncher(const int nvIn, const int C, const int* vtReplace,
                       const int* vtMap, const float* input, float* output, int* maxIndex)
{
    int numGrid = int(nvIn*C/1024) + 1;
    max_pool3d_init   <<<numGrid,1024>>>(nvIn, C, vtReplace, vtMap, input, output);
    max_pool3d_forward<<<numGrid,1024>>>(nvIn, C, vtReplace, vtMap, input, output);
    max_pool3d_index  <<<numGrid,1024>>>(nvIn, C, vtReplace, vtMap, input, output, maxIndex);
}

void maxPool3dGradLauncher(const int nvOut, const int C, const int* maxIndex,
                           const float* gradOutput, float* gradInput)
{
    int numGrid = int(nvOut*C/1024) + 1;
    max_pool3d_backward<<<numGrid,1024>>>(nvOut, C, maxIndex, gradOutput, gradInput);
    //cudaDeviceSynchronize();
}

void avgPool3dLauncher(const int nvIn, const int C, const int* vtReplace,
                       const int* vtMap, const float* input, float* output)
{
    int numGrid = int(nvIn*C/1024) + 1;
    avg_pool3d_forward<<<numGrid,1024>>>(nvIn, C, vtReplace, vtMap, input, output);
    //cudaDeviceSynchronize();
}

void avgPool3dGradLauncher(const int nvIn, const int C, const int* vtReplace,
                           const int* vtMap, const float* gradOutput, float* gradInput)
{
    int numGrid = int(nvIn*C/1024) + 1;
    avg_pool3d_backward<<<numGrid,1024>>>(nvIn, C, vtReplace, vtMap, gradOutput, gradInput);
    //cudaDeviceSynchronize();
}

