// numInterior: (NfIn)
// baryCoeff:   (NiK, F)
// input:       (NiK, Cin)
// filter:      (Cout, Cin, F)
// output:      (NfIn, Cout)
__global__ void facet2facet_conv3d_forward(int NfIn, int F, int Cin, int Cout, const int* numInterior,
                                           const float* baryCoeff, const float* input,
                                           const float* filter, float* output)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x; // thread index
    int fcIdx = idx/Cout;  // global face index in the batch
    int ch_out = idx%Cout;  // output channel ID

    if(fcIdx<NfIn)
    {
        int kStart = 0;
        int kEnd = numInterior[fcIdx];
        if (fcIdx>0) kStart = numInterior[fcIdx-1];

        // convolution
        int K = kEnd - kStart;
        for(int ch_in=0;ch_in<Cin;ch_in++)
        {
            int ch = ch_out*Cin*F + ch_in*F;
            for(int k=kStart;k<kEnd;k++)
            {
                // get interpolation weights (w1,w2,w3) related to (v1->v2->v3) of the face
                float weight = 0.0;
                for (int d=0; d<F; d++)
                {
                    weight += baryCoeff[k*F+d]*filter[ch+d];
                }
                output[fcIdx*Cout+ch_out] += input[k*Cin+ch_in]*weight/K;
            }
        }
    }
}


// numInterior: (NfIn)
// baryCoeff:   (NiK, F)
// filter:      (Cout, Cin, F)
// gradOutput:  (NfIn, Cout)
// gradInput:   (NiK, Cin)
__global__ void facet2facet_input_backward(int NfIn, int F, int Cin, int Cout, const int* numInterior,
                                           const float* baryCoeff, const float* filter,
                                           const float* gradOutput, float* gradInput)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x; // thread index
    int fcIdx = idx/Cout;   // global face index in the batch
    int ch_out = idx%Cout;  // output channel ID

    if(fcIdx<NfIn)
    {
        int kStart = 0;
        int kEnd = numInterior[fcIdx];
        if (fcIdx>0) kStart = numInterior[fcIdx-1];

        // convolution
        int K = kEnd - kStart;
        for(int ch_in=0;ch_in<Cin;ch_in++)
        {
            int ch = ch_out*Cin*F + ch_in*F;
            for(int k=kStart;k<kEnd;k++)
            {
                // get interpolation weights (w1,w2,w3) related to (v1->v2->v3) of the face
                float weight = 0.0;
                for (int d=0; d<F; d++)
                {
                    weight += baryCoeff[k*F+d]*filter[ch+d];
                }
                float derIn = gradOutput[fcIdx*Cout+ch_out]*weight/K;
                atomicAdd(&gradInput[k*Cin+ch_in], derIn);
            }
        }
    }
}


// numInterior: (NfIn)
// baryCoeff:   (NiK, F)
// input:       (NiK, Cin)
// gradOutput:  (NfIn, Cout)
// gradFilter:  (Cout, Cin, F)
__global__ void facet2facet_filter_backward(int NfIn, int F, int Cin, int Cout, const int* numInterior,
                                            const float* baryCoeff, const float* input, const float* gradOutput,
                                            float* gradFilter, int sharedMemSize, int startIdx)
{
    extern __shared__ float gradPerBlock[]; // the gradient on each block
    for (int i=threadIdx.x;i<sharedMemSize;i+=blockDim.x)
    {
        gradPerBlock[i] = 0; // for 1D block
    }
    __syncthreads();

    int idx = blockIdx.x*blockDim.x + threadIdx.x; // thread index
    int fcIdx = idx/Cout;  // global face index in the batch
    int ch_out = idx%Cout;  // output channel ID

    int endIdx = sharedMemSize+startIdx;
    if(fcIdx<NfIn)
    {
        int kStart = 0;
        int kEnd = numInterior[fcIdx];
        if (fcIdx>0) kStart = numInterior[fcIdx-1];

        // convolution
        int K = kEnd - kStart;
        for(int ch_in=0;ch_in<Cin;ch_in++)
        {
            int ch = ch_out*Cin*F + ch_in*F;
            for(int d=0;d<F;d++)
            {
                int currIdx = ch + d;
                if((currIdx>=startIdx) && (currIdx<endIdx)) // within the shared memory
                {
                    float derFilt = 0.0;
                    for(int k=kStart;k<kEnd;k++)
                    {
                        // get interpolation weights (w1,w2,w3) related to (v1->v2->v3) of the face
                        float temp = gradOutput[fcIdx*Cout+ch_out]*input[k*Cin+ch_in]/K;
                        derFilt += temp*baryCoeff[k*F+d];
                    }
                   atomicAdd(&gradPerBlock[currIdx-startIdx],derFilt);
                }
            }
        }
    }
    __syncthreads();

    for (int i=threadIdx.x;i<sharedMemSize;i+=blockDim.x)
    {
        atomicAdd(&gradFilter[i+startIdx],gradPerBlock[i]); // for 1D block
    }
}
void facet2facetConv3dLauncher(int NfIn, int F, int Cin, int Cout, const int* numInterior, const float* baryCoeff,
                               const float* input, const float* filter, float* output)
{
    int numGrid = NfIn*Cout/1024 + 1;
    facet2facet_conv3d_forward<<<numGrid,1024>>>(NfIn, F, Cin, Cout, numInterior, baryCoeff,
                                                 input, filter, output);

}
void facet2facetConv3dGradLauncher(int NfIn, int F, int Cin, int Cout, const int* numInterior, const float* baryCoeff,
                                   const float* input, const float* filter, const float* gradOutput,
                                   float* gradInput, float* gradFilter)
{
    int numGrid = NfIn*Cout/1024 + 1;
    facet2facet_input_backward<<<numGrid,1024>>>(NfIn, F, Cin, Cout, numInterior, baryCoeff, filter,
                                                 gradOutput, gradInput);

    // titan xp has shared memory of 49152 bytes, each float value takes 4 bytes in the memory
    int maxSharedMemSize = int(49152/sizeof(float));
    int maxIter = (F*Cout*Cin)/maxSharedMemSize;
    int remainder = (F*Cout*Cin)%maxSharedMemSize;
    for(int iter=0;iter<maxIter;iter++)
    {
        facet2facet_filter_backward<<<numGrid,1024,sizeof(float)*maxSharedMemSize>>>(NfIn, F, Cin, Cout, numInterior,
                                                                            baryCoeff, input, gradOutput, gradFilter,
                                                                            maxSharedMemSize, maxSharedMemSize*iter);
    }
    if(remainder>0) // fill the remainder
    {
        facet2facet_filter_backward<<<numGrid,1024,sizeof(float)*remainder>>>(NfIn, F, Cin, Cout, numInterior,
                                                                        baryCoeff, input, gradOutput, gradFilter,
                                                                        remainder, maxSharedMemSize*maxIter);
    }
}



// face:    (NfIn, 3)
// input:   (NvIn, C)
// filter:  (3, C, r)
// output:  (NfIn, C*r)
__global__ void vertex2facet_conv3d_forward(int NfIn, int C, int r, const int* face,
                                            const float* input, const float* filter, float* output)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x; // thread index
    int fcIdx = idx/(C*r);  // global face index in the batch
    int cout = idx%(C*r);   // output channel ID
    int cin = cout/r;       // input channel ID

    if(fcIdx<NfIn)
    {
        int v[3] = {face[3*fcIdx], face[3*fcIdx+1], face[3*fcIdx+2]};

        float feat = 0.0;
        for (int k=0;k<3;k++)
        {
            feat += filter[cout+k*C*r]*input[v[k]*C+cin];
        }
        output[fcIdx*C*r+cout] = feat;
    }
}

// face:       (NfIn, 3)
// filter:     (3, C, r)
// gradOutput: (NfIn, C*r)
// gradInput:  (NvIn, C)
__global__ void vertex2facet_input_backward(int NfIn, int C, int r, const int* face, const float* filter,
                                            const float* gradOutput, float* gradInput)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x; // thread index
    int fcIdx = idx/(C*r);  // global face index in the batch
    int cout = idx%(C*r);  // output channel ID
    int cin = cout/r;     // input channel ID

    if(fcIdx<NfIn)
    {
        int v[3] = {face[3*fcIdx], face[3*fcIdx+1], face[3*fcIdx+2]};

        for (int k=0;k<3;k++)
        {
            atomicAdd(&gradInput[v[k]*C+cin], gradOutput[fcIdx*C*r+cout]*filter[cout+k*C*r]);
        }
    }
}

// face:       (NfIn, 3)
// input:      (NvIn, C)
// gradOutput: (NfIn, C*r)
// gradFilter: (3, C, r)
__global__ void vertex2facet_filter_backward(int NfIn, int C, int r, const int* face, const float* input,
                                             const float* gradOutput, float* gradFilter,
                                             int sharedMemSize, int startIdx)
{
    extern __shared__ float gradPerBlock[]; // the gradient on each block
    for (int i=threadIdx.x;i<sharedMemSize;i+=blockDim.x)
    {
        gradPerBlock[i] = 0; // for 1D block
    }
    __syncthreads();

    int idx   = blockIdx.x*blockDim.x + threadIdx.x; // thread index
    int fcIdx = idx/(C*r);  // global face index in the batch
    int cout  = idx%(C*r);  // output channel ID
    int cin   = cout/r;     // input channel ID

    int endIdx = sharedMemSize+startIdx;
    if(fcIdx<NfIn)
    {
        int v[3] = {face[3*fcIdx], face[3*fcIdx+1], face[3*fcIdx+2]};

        for(int k=0;k<3;k++)
        {
            int currIdx = k*C*r+cout;
            if((currIdx>=startIdx) && (currIdx<endIdx)) // within the shared memory
            {
               atomicAdd(&gradPerBlock[currIdx-startIdx],gradOutput[fcIdx*C*r+cout]*input[v[k]*C+cin]);
            }
        }
    }
    __syncthreads();

    for (int i=threadIdx.x;i<sharedMemSize;i+=blockDim.x)
    {
        atomicAdd(&gradFilter[i+startIdx],gradPerBlock[i]); // for 1D block
    }
}
void vertex2facetConv3dLauncher(int NfIn, int C, int r, const int* face,
                                const float* input, const float* filter, float* output)
{
    int numGrid = NfIn*C*r/1024 + 1;
    vertex2facet_conv3d_forward<<<numGrid,1024>>>(NfIn, C, r, face, input, filter, output);
}
void vertex2facetConv3dGradLauncher(int NfIn, int C, int r, const int* face, const float* input,
                                    const float* filter, const float* gradOutput,
                                    float* gradInput, float* gradFilter)
{
    int numGrid = NfIn*C*r/1024 + 1;
    vertex2facet_input_backward<<<numGrid,1024>>>(NfIn, C, r, face, filter, gradOutput, gradInput);

    // titan xp has shared memory of 49152 bytes, each float value takes 4 bytes in the memory
    int maxSharedMemSize = int(49152/sizeof(float));
    int maxIter = (3*C*r)/maxSharedMemSize;
    int remainder = (3*C*r)%maxSharedMemSize;
    for(int iter=0;iter<maxIter;iter++)
    {
        vertex2facet_filter_backward<<<numGrid,1024,sizeof(float)*maxSharedMemSize>>>(NfIn, C, r, face,
                                                                                input, gradOutput, gradFilter,
                                                                                maxSharedMemSize, maxSharedMemSize*iter);
    }
    if(remainder>0) // fill the remainder
    {
        vertex2facet_filter_backward<<<numGrid,1024,sizeof(float)*remainder>>>(NfIn, C, r, face,
                                                                         input, gradOutput, gradFilter,
                                                                         remainder, maxSharedMemSize*maxIter);
    }
}

// vtMap:   NvIn; // only non-negative mapid got output features
// nfCount: NvIn;
// face:    NfIn*3;
// coeff:   NfIn*K;
// input:   NfIn*C;
// filter:  K*C*r;
// output:  NvOut*(C*r);
__global__ void facet2vertex_conv3d_forward(int NfIn, int C, int r, const int K, const int* vtMap,
                                    const int* nfCount, const int* face, const float* coeff,
                                    const float* input, const float* filter, float* output)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x; // thread index
    int fcIdx = idx/(C*r);  // global face index in the batch
    int cout = idx%(C*r);   // output channel ID
    int cin = cout/r;       // input channel ID

    if (fcIdx<NfIn) // index must be in the legal range
    {
        // a fuzzy combined weights
        float weight = 0;
        for(int k=0;k<K;k++)
        {
            float xi_k = coeff[fcIdx*K+k];
            weight += xi_k*filter[k*C*r+cout];
        }
        float out_feat = weight*input[fcIdx*C+cin];

        int v[3] = {face[3*fcIdx], face[3*fcIdx+1], face[3*fcIdx+2]};
        for(int k=0;k<3;k++)          // aggregate context of vertex from adjacent faces
        {
            int vi = v[k];
            int vo = vtMap[vi];       // for non-strided convolution, we have vtMap[vi]=vi.
            int nfSize = nfCount[vo]; //nfSize is the number of adjacent faces to vi, try nfSize=1 for no averaging
            if (vo>=0)
                atomicAdd(&output[vo*C*r+cout], out_feat/nfSize);
        }
    }
}

// vtMap:      NvIn;   // only non-negative mapid got output features
// nfCount:    NvIn;
// face:       NfIn*3;
// coeff:      NfIn*K;
// filter:     K*C*r;
// gradOutput: NvOut*(C*r)
// gradInput:  NfIn*C;
__global__ void facet2vertex_input_backward(int NfIn, int C, int r, const int K, const int* vtMap,
                                    const int* nfCount, const int* face, const float* coeff,
                                    const float* filter, const float* gradOutput, float* gradInput)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x; // thread index
    int fcIdx = idx/(C*r);  // global face index in the batch
    int cout = idx%(C*r);  // output channel ID
    int cin = cout/r;     // input channel ID

    if (fcIdx<NfIn)     // index must be in the legal range
    {
        // a fuzzy combined weights
        float weight = 0;
        for(int k=0;k<K;k++)
        {
            float xi_k = coeff[fcIdx*K+k];
            weight += xi_k*filter[k*C*r+cout];
        }

        // gradInput is on faces, each face collect gradients from three vertices
        // better no atomic addition
        int v[3] = {face[3*fcIdx], face[3*fcIdx+1], face[3*fcIdx+2]};
        for(int k=0;k<3;k++)    // aggregate context of vertex from adjacent faces
        {
            int vi = v[k];
            int vo = vtMap[vi];
            int nfSize = nfCount[vo];
            if (vo>=0)
            {
                float derIn = gradOutput[vo*C*r+cout]*weight/nfSize;
                atomicAdd(&gradInput[fcIdx*C+cin], derIn);
            }
        }
    }
}

// vtMap:      NvIn;   // only non-negative mapid got output features
// nfCount:    NvIn;
// face:       NfIn*3;
// coeff:      NfIn*K;
// input:      NfIn*C;
// gradOutput: NvOut*(C*r)
// gradFilter: K*C*r;
__global__ void facet2vertex_filter_backward(int NfIn, int C, int r, const int K, const int* vtMap,
                                     const int* nfCount, const int* face, const float* coeff,
                                     const float* input, const float* gradOutput,
                                     float* gradFilter, int sharedMemSize, int startIdx)
{
    extern __shared__ float gradPerBlock[]; // the gradient on each block
    for (int i=threadIdx.x;i<sharedMemSize;i+=blockDim.x)
    {
        gradPerBlock[i] = 0; // for 1D block
    }
    __syncthreads();

    int idx = blockIdx.x*blockDim.x + threadIdx.x; // thread index
    int fcIdx = idx/(C*r);  // global face index in the batch
    int cout = idx%(C*r);  // output channel ID
    int cin = cout/r;     // input channel ID

    int endIdx = sharedMemSize+startIdx;
    if (fcIdx<NfIn)     // index must be in the legal range
    {
        int v[3] = {face[3*fcIdx], face[3*fcIdx+1], face[3*fcIdx+2]};

        for(int k=0;k<K;k++)
        {
            int currIdx = k*C*r+cout;
            if((currIdx>=startIdx) && (currIdx<endIdx)) // within the shared memory
            {
                float derFilt = coeff[fcIdx*K+k]*input[fcIdx*C+cin];
                for(int m=0;m<3;m++)
                {
                    int vi = v[m];
                    int vo = vtMap[vi];
                    int nfSize = nfCount[vo];
                    if (vo>=0)
                        atomicAdd(&gradPerBlock[currIdx-startIdx], gradOutput[vo*C*r+cout]*derFilt/nfSize);
                }
            }
        }
    }
    __syncthreads();

    for (int i=threadIdx.x;i<sharedMemSize;i+=blockDim.x)
    {
        atomicAdd(&gradFilter[i+startIdx],gradPerBlock[i]); // for 1D block
    }
}

void facet2vertexConv3dLauncher(int NfIn, int C, int r, int K, const int* vtMap, const int* nfCount, const int* face,
                                const float* coeff, const float* input, const float* filter, float* output)
{
    int numGrid = NfIn*C*r/1024 + 1;
    facet2vertex_conv3d_forward<<<numGrid,1024>>>(NfIn, C, r, K, vtMap, nfCount, face, coeff, input, filter, output);
}
void facet2vertexConv3dGradLauncher(int NfIn, int C, int r, int K, const int* vtMap, const int* nfCount, const int* face,
                                    const float* coeff, const float* input, const float* filter, const float* gradOutput,
                                    float* gradInput, float* gradFilter)
{
    int numGrid = NfIn*C*r/1024 + 1;
    facet2vertex_input_backward<<<numGrid,1024>>>(NfIn, C, r, K, vtMap, nfCount, face, coeff, filter, gradOutput, gradInput);

    // titan xp has shared memory of 49152 bytes, each float value takes 4 bytes in the memory
    int maxSharedMemSize = int(49152/sizeof(float));

    int maxIter = (K*C*r)/maxSharedMemSize;
    int remainder = (K*C*r)%maxSharedMemSize;
    for(int iter=0;iter<maxIter;iter++)
    {
        facet2vertex_filter_backward<<<numGrid,1024,sizeof(float)*maxSharedMemSize>>>(NfIn, C, r, K, vtMap, nfCount, face,
                                                                                      coeff, input, gradOutput, gradFilter,
                                                                                  maxSharedMemSize, maxSharedMemSize*iter);
    }
    if(remainder>0) // fill the remainder
    {
        facet2vertex_filter_backward<<<numGrid,1024,sizeof(float)*remainder>>>(NfIn, C, r, K, vtMap, nfCount, face,
                                                                               coeff, input, gradOutput, gradFilter,
                                                                               remainder, maxSharedMemSize*maxIter);
    }
}


