#include <torch/extension.h>
#include <vector>
//#include <iostream>

typedef int64_t LLint;

// CUDA forward and backward declarations
void depthwiseConv3dLauncher(int Np, int Mp, int Nout, int F, int C, int r,
                             const int* nnCount, const int* nnIndex, const int* binIndex,
                             const float* input, const float* filter, float* output);
void depthwiseConv3dGradLauncher(int Np, int Mp, int Nout, int F, int C, int r,
                                 const int* nnCount, const int* nnIndex, const int* binIndex,
                                 const float* input, const float* filter, const float* gradOutput,
                                 float* gradInput, float* gradFilter);
void fuzzyDepthwiseConv3dLauncher(int Np, int Mp, int Nout, int F, int C, int r, int T, const int* nnCount,
                                  const int* nnIndex, const int* binIndex, const float* binCoeff,
                                  const float* input, const float* filter, float* output);
void fuzzyDepthwiseConv3dGradLauncher(int Np, int Mp, int Nout, int F, int C, int r, int T,
                                      const int* nnCount, const int* nnIndex, const int* binIndex,
                                      const float* binCoeff, const float* input, const float* filter,
                                      const float* gradOutput, float* gradInput, float* gradFilter);

// C++ interface
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_DIM(x,dims) TORCH_CHECK(x.dim()==dims, #x " must have dimension of " #dims)
#define CHECK_INPUT(x,dims) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_DIM(x,dims);


torch::Tensor DepthwiseConv3d(
    torch::Tensor input,     // input features: concat_Np * in_channels
    torch::Tensor filter,    // convolution filter parameters: filter_size * in_channels * channel_multiplier
    torch::Tensor nn_count,  // number of neighbors: concat_Mp
    torch::Tensor nn_index,  // neighbor indices, each pair (outIdx, inIdx): Nout * 2
    torch::Tensor bin_index) // kernel bin indices: Nout
{
    CHECK_INPUT(input,2);
    CHECK_INPUT(filter,3);
    CHECK_INPUT(nn_count,1);
    CHECK_INPUT(nn_index,2);
    CHECK_INPUT(bin_index,1);

    // get the dims required by computations
    int Np   = input.size(0);    // number of input points
    int C    = input.size(1);    // number of input channels
    int F    = filter.size(0);   // filter bin size
    int r    = filter.size(2);   // depthwise channel multiplier
    int Mp   = nn_count.size(0); // number of output points
    int Nout = nn_index.size(0); // number of neighbor pairs

    TORCH_CHECK(filter.size(1)==C, "Input Channel size error of the filter");
    TORCH_CHECK(nn_index.dim()==2, "The rank of nn_index should be 2.");
    TORCH_CHECK(bin_index.dim()==1, "The rank of bin_index should be 1.");

    // get pointers to the input tensors
    const float* in_ptr     = input.data_ptr<float>();
    const float* filt_ptr   = filter.data_ptr<float>();
    const int* nnCount_ptr  = nn_count.data_ptr<int32_t>();
    const int* nnIndex_ptr  = nn_index.data_ptr<int32_t>();
    const int* binIndex_ptr = bin_index.data_ptr<int32_t>();

    // create an output tensor
    auto output = torch::zeros({Mp,C*r}, input.options());
    float* out_ptr = output.data_ptr<float>();

    depthwiseConv3dLauncher(Np, Mp, Nout, F, C, r, nnCount_ptr, nnIndex_ptr, binIndex_ptr,
                            in_ptr, filt_ptr, out_ptr);
    return output;
}


std::vector<torch::Tensor> DepthwiseConv3dGrad(
    torch::Tensor grad_output,  // gradient of output features: concat_Mp * out_channels
    torch::Tensor input,        // input features: concat_Np * in_channels
    torch::Tensor filter,       // convolution filter parameters: filter_size * in_channels * channel_multiplier
    torch::Tensor nn_count,     // number of neighbors: concat_Mp
    torch::Tensor nn_index,     // neighbor indices: Nout * 2
    torch::Tensor bin_index)    // kernel bin indices: Nout
{
    CHECK_INPUT(input,2);
    CHECK_INPUT(filter,3);
    CHECK_INPUT(grad_output,2);
    CHECK_INPUT(nn_count,1);
    CHECK_INPUT(nn_index,2);
    CHECK_INPUT(bin_index,1);

     // get the dims required by computations
    int Np   = input.size(0);    // number of input points
    int C    = input.size(1);    // number of input channels
    int F    = filter.size(0);   // filter bin size
    int r    = filter.size(2);   // depthwise channel multiplier
    int Mp   = nn_count.size(0); // number of output points
    int Nout = nn_index.size(0); // number of neighbor pairs

    TORCH_CHECK(filter.size(1)==C, "Input Channel size error of the filter");
    TORCH_CHECK(nn_index.dim()==2, "The rank of nn_index should be 2.");
    TORCH_CHECK(bin_index.dim()==1, "The rank of bin_index should be 1.");

    // get pointers to the input tensors
    const float* in_ptr      = input.data_ptr<float>();
    const float* filt_ptr    = filter.data_ptr<float>();
    const float* gradOut_ptr = grad_output.data_ptr<float>();
    const int* nnCount_ptr   = nn_count.data_ptr<int32_t>();
    const int* nnIndex_ptr   = nn_index.data_ptr<int32_t>();
    const int* binIndex_ptr  = bin_index.data_ptr<int32_t>();

    // create output tensors
    auto grad_input = torch::zeros({Np,C}, input.options());
    auto grad_filter = torch::zeros({F,C,r}, input.options());
    float* gradIn_ptr = grad_input.data_ptr<float>();
    float* gradFilt_ptr = grad_filter.data_ptr<float>();

    depthwiseConv3dGradLauncher(Np, Mp, Nout, F, C, r, nnCount_ptr, nnIndex_ptr, binIndex_ptr,
                                in_ptr, filt_ptr, gradOut_ptr, gradIn_ptr, gradFilt_ptr);
    return {grad_input, grad_filter};
}


torch::Tensor FuzzyDepthwiseConv3d(
    torch::Tensor input,        // input features: concat_Np * in_channels
    torch::Tensor filter,       // convolution filter parameters: filter_size * in_channels * channel_multiplier
    torch::Tensor nn_count,     // number of neighbors: concat_Mp
    torch::Tensor nn_index,     // neighbor indices: Nout * 2
    torch::Tensor bin_index,    // kernel bin indices: Nout * 3
    torch::Tensor bin_coeff)    // kernel bin coefficients: Nout * 3
{
    CHECK_INPUT(input,2);
    CHECK_INPUT(filter,3);
    CHECK_INPUT(nn_count,1);
    CHECK_INPUT(nn_index,2);
    CHECK_INPUT(bin_index,2);
    CHECK_INPUT(bin_coeff,2);

    // get the dims required by computations
    int Np   = input.size(0);    // number of input points
    int C   = input.size(1);     // number of input channels
    int F    = filter.size(0);   // filter bin size
    int r    = filter.size(2);   // depthwise channel multiplier
    int Mp   = nn_count.size(0); // number of output points
    int Nout = nn_index.size(0); // number of neighbor pairs
    int T    = bin_index.size(1);// maximum number of clusters/bins covered

    TORCH_CHECK(filter.size(1)==C, "Channel size error of the filter");
    TORCH_CHECK(nn_index.dim()==2, "The rank of nn_index should be 2.");
    TORCH_CHECK(bin_index.dim()==2, "The rank of bin_index should be 2.");
    TORCH_CHECK(bin_coeff.dim()==2, "The rank of bin_coeff should be 2.");

    // get c++ pointers to the input tensors
    const float* in_ptr = input.data_ptr<float>();
    const float* filt_ptr = filter.data_ptr<float>();
    const int* nnCount_ptr = nn_count.data_ptr<int32_t>();
    const int* nnIndex_ptr = nn_index.data_ptr<int32_t>();
    const int* binIndex_ptr = bin_index.data_ptr<int32_t>();
    const float* binCoeff_ptr = bin_coeff.data_ptr<float>();

    // create an output tensor
    auto output = torch::zeros({Mp,C*r}, input.options());
    float* out_ptr = output.data_ptr<float>();
    fuzzyDepthwiseConv3dLauncher(Np, Mp, Nout, F, C, r, T, nnCount_ptr, nnIndex_ptr, binIndex_ptr,
                                 binCoeff_ptr, in_ptr, filt_ptr, out_ptr);
    return output;
};



std::vector<torch::Tensor> FuzzyDepthwiseConv3dGrad(
    torch::Tensor grad_output,  // gradient of output features: concat_Mp * out_channels
    torch::Tensor input,        // input features: concat_Np * in_channels
    torch::Tensor filter,       // convolution filter parameters: filter_size * in_channels * channel_multiplier
    torch::Tensor nn_count,     // number of neighbors: concat_Mp
    torch::Tensor nn_index,     // neighbor indices: Nout * 2
    torch::Tensor bin_index,    // kernel bin indices: Nout * 3
    torch::Tensor bin_coeff)    // kernel bin coefficients: Nout * 3
{
    CHECK_INPUT(input,2);
    CHECK_INPUT(filter,3);
    CHECK_INPUT(grad_output,2);
    CHECK_INPUT(nn_count,1);
    CHECK_INPUT(nn_index,2);
    CHECK_INPUT(bin_index,2);
    CHECK_INPUT(bin_coeff,2);

    // get the dims required by computations
    int Np   = input.size(0);      // number of input points
    int C    = input.size(1);      // number of input channels
    int F    = filter.size(0);     // filter bin size
    int r    = filter.size(2);     // depthwise channel multiplier
    int Mp   = nn_count.size(0);   // number of output points
    int Nout = nn_index.size(0);   // number of neighbor pairs
    int T   = bin_index.size(1);   // maximum number of clusters/bins covered

    TORCH_CHECK(filter.size(1)==C, "Channel size error of the filter");
    TORCH_CHECK(nn_index.dim()==2, "The rank of nn_index should be 2.");
    TORCH_CHECK(bin_index.dim()==2, "The rank of bin_index should be 2.");
    TORCH_CHECK(bin_coeff.dim()==2, "The rank of bin_coeff should be 2.");

    // get c++ pointers to the input tensors
    const float* in_ptr = input.data_ptr<float>();
    const float* filt_ptr = filter.data_ptr<float>();
    const float* gradOut_ptr = grad_output.data_ptr<float>();
    const int* nnCount_ptr = nn_count.data_ptr<int32_t>();
    const int* nnIndex_ptr = nn_index.data_ptr<int32_t>();
    const int* binIndex_ptr = bin_index.data_ptr<int32_t>();
    const float* binCoeff_ptr = bin_coeff.data_ptr<float>();

    // create output tensors
    auto grad_input = torch::zeros({Np,C}, input.options());
    auto grad_filter = torch::zeros({F,C,r}, input.options());
    float* gradIn_ptr = grad_input.data_ptr<float>();
    float* gradFilt_ptr = grad_filter.data_ptr<float>();

    fuzzyDepthwiseConv3dGradLauncher(Np, Mp, Nout, F, C, r, T, nnCount_ptr, nnIndex_ptr, binIndex_ptr,
                                     binCoeff_ptr, in_ptr, filt_ptr, gradOut_ptr, gradIn_ptr, gradFilt_ptr);
    return {grad_input, grad_filter};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("SPH3D_forward",&DepthwiseConv3d, "DepthwiseConv3d forward (CUDA)");
    m.def("SPH3D_backward", &DepthwiseConv3dGrad, "DepthwiseConv3d backward (CUDA)");
    m.def("fuzzySPH3D_forward", &FuzzyDepthwiseConv3d, "FuzzyDepthwiseConv3d forward (CUDA)");
    m.def("fuzzySPH3D_backward", &FuzzyDepthwiseConv3dGrad, "FuzzyDepthwiseConv3d backward (CUDA)");
}



