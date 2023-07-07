#include <torch/extension.h>
#include <vector>
//#include <iostream>


// CUDA declarations
void weightedInterpolateLauncher(int Np, int Mp, int C, int K, const int* nnIndex,
                                 const float* input, const float* weight, float* output);
void weightedInterpolateGradLauncher(int Np, int Mp, int C,  int K, const int* nnIndex,
                                     const float* gradOutput, const float* weight, float* gradInput);


// C++ interface
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_DIM(x,dims) TORCH_CHECK(x.dim()==dims, #x " must have dimension of " #dims)
#define CHECK_INPUT(x,dims) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_DIM(x,dims);


torch::Tensor WeightedInterpolate(
    torch::Tensor input,     // input features: concat_Mp * in_channels
    torch::Tensor weight,    // weights: concat_Np * 3
    torch::Tensor nn_index)  // neighbor indices: concat_Np * 3
{
    CHECK_INPUT(input,2);
    CHECK_INPUT(weight,2);
    CHECK_INPUT(nn_index,2);

    // get the dims required by computations
    int Mp = input.size(0);     // number of input points
    int C  = input.size(1);     // number of input channels
    int Np = nn_index.size(0);  // number of output points
    int K  = nn_index.size(1);  // max number of neighbors sampled

    TORCH_CHECK(nn_index.dim()==2, "rank of nn_index should be 2, i.e. (Nout,K)");

    // get c++ pointers to the input tensors
    const float* in_ptr     = input.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const int* nnIndex_ptr  = nn_index.data_ptr<int32_t>();

    // create an output tensor
    auto output = torch::zeros({Np,C}, input.options());
    float* out_ptr = output.data_ptr<float>();

    weightedInterpolateLauncher(Np, Mp, C, K, nnIndex_ptr, in_ptr, weight_ptr, out_ptr);
    return output;
}



torch::Tensor WeightedInterpolateGrad(
    torch::Tensor grad_output, // gradient of unpooled features: concat_Np * in_channels
    torch::Tensor input,       // input features: concat_Mp * in_channels
    torch::Tensor weight,      // weights: concat_Np * 3
    torch::Tensor nn_index)    // neighbor indices: concat_Np * 3
{
    CHECK_INPUT(input,2);
    CHECK_INPUT(grad_output,2);
    CHECK_INPUT(weight,2);
    CHECK_INPUT(nn_index,2);

    // get the dims required by computations
    int Mp = input.size(0);        // number of input points
    int C  = input.size(1);        // number of input channels
    int Np = grad_output.size(0);  // number of output points
    int K  = nn_index.size(1);     // max number of neighbors sampled

    TORCH_CHECK(nn_index.dim()==2, "rank of nn_index should be 2, i.e. (Nout,K)");

    // get pointers to the input tensors
    const float* gradOut_ptr = grad_output.data_ptr<float>();
    const float* weight_ptr  = weight.data_ptr<float>();
    const int* nnIndex_ptr   = nn_index.data_ptr<int32_t>();

    // create an output tensor
    auto grad_input = torch::zeros({Mp,C}, input.options());
    float* gradIn_ptr = grad_input.data_ptr<float>();

    weightedInterpolateGradLauncher(Np, Mp, C, K, nnIndex_ptr, gradOut_ptr, weight_ptr, gradIn_ptr);
    return grad_input;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward",  &WeightedInterpolate,     "WeightedInterpolate forward (CUDA)");
    m.def("backward", &WeightedInterpolateGrad, "WeightedInterpolate backward (CUDA)");
}