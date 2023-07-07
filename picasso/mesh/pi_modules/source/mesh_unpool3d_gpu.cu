// vtReplace: nvOut;
// vtMap:     nvOut;
// input:     nvIn*C;
// output:    nvOut*C (nvOut>nvIn)
__global__ void interpolate_forward(int nvOut, int nvIn, int C, const int* vtReplace, const int* vtMap,
                                    const float* input, float* output)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x; // thread index
    int vo  = idx/C; // global vertex index
    int ch  = idx%C; // which feature channel

    if(vo < nvOut)
    {
        int mapid = vo;
        if (vtReplace[vo]<0)
            mapid = -vtReplace[vo];

        if (vtMap[mapid]>=0)  // make sure `vi' is valid
        {
            int vi = vtMap[mapid]; // mapping from output index to the input index
            output[vo*C+ch] = input[vi*C+ch]; // copy input to output because we have isolated vertex clusters
        }
    }
}


__global__ void interpolate_backward(int nvOut, int nvIn, int C, const int* vtReplace, const int* vtMap,
                                     const float* gradOutput, float* gradInput)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x; // thread index
    int vo  = idx/C; // global vertex index of the input
    int ch  = idx%C; // which feature channel

    if(vo < nvOut)
    {
        int mapid = vo;
        if (vtReplace[vo]<0)
            mapid = -vtReplace[vo];

        if (vtMap[mapid]>=0)  // make sure `vi' is valid
        {
            int vi = vtMap[mapid]; // mapping from input index to the output index
            atomicAdd(&gradInput[C*vi+ch],gradOutput[C*vo+ch]);
        }
    }
}


void interpolateLauncher(int nvIn, int C, int nvOut, const int* vtReplace, const int* vtMap,
                             const float* input, float* output)
{
    int numGrid = int(nvOut*C/1024) + 1;
    interpolate_forward<<<numGrid,1024>>>(nvOut, nvIn, C, vtReplace, vtMap, input, output);
    //cudaDeviceSynchronize();
}

void interpolateGradLauncher(int nvIn, int C, int nvOut, const int* vtReplace, const int* vtMap,
                           const float* gradOutput, float* gradInput)
{
    int numGrid = int(nvOut*C/1024) + 1;
    interpolate_backward<<<numGrid,1024>>>(nvOut, nvIn, C, vtReplace, vtMap, gradOutput, gradInput);
    //cudaDeviceSynchronize();
}


