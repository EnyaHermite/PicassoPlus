#include <torch/extension.h>
#include <vector>
//#include <iostream>


// CUDA forward and backward declarations
void maxPool3dLauncher(int Np, int Mp, int Nout, int C, const int* nnCount, const int* nnIndex,
                       const float* input, float* output, int* maxIndex);
void maxPool3dGradLauncher(int Np, int Mp, int C, const int* maxIndex, const float* gradOutput, float* gradInput);
void avgPool3dLauncher(int Np, int Mp, int Nout, int C, const int* nnCount, const int* nnIndex,
                       const float* input, float* output);
void avgPool3dGradLauncher(int Np, int Mp, int Nout, int C, const int* nnCount,
                           const int* nnIndex, const float* gradOutput, float* gradInput);


// C++ interface
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_DIM(x,dims) TORCH_CHECK(x.dim()==dims, #x " must have dimension of " #dims)
#define CHECK_INPUT(x,dims) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_DIM(x,dims);


std::vector<torch::Tensor> MaxPool3d(
    torch::Tensor input,      // input features: concat_Np * in_channels
    torch::Tensor nn_count,   // number of neighbors: concat_Mp
    torch::Tensor nn_index)   // neighbor and kernel bin indices: Nout * 2
{
    CHECK_INPUT(input,2);
    CHECK_INPUT(nn_count,1);
    CHECK_INPUT(nn_index,2);

    // get the dims required by computations
    int Np   = input.size(0);    // number of input points
    int C    = input.size(1);    // number of input channels
    int Mp   = nn_count.size(0); // number of output points
    int Nout = nn_index.size(0); // number of neighbor pairs

    TORCH_CHECK(nn_count.dim()==1, "Dimension of nn_count should be 1.");
    TORCH_CHECK(nn_index.dim()==2 && nn_index.size(1)==2, "Shape of database points requires to be (Nout,2).");

    // get c++ pointers to the input tensors
    const float* in_ptr    = input.data_ptr<float>();
    const int* nnCount_ptr = nn_count.data_ptr<int32_t>();
    const int* nnIndex_ptr = nn_index.data_ptr<int32_t>();

    // create output tensors
    auto output = torch::zeros({Mp,C}, input.options());
    auto max_index = torch::zeros({Mp,C}, input.options().dtype(torch::kInt32));
    float* out_ptr = output.data_ptr<float>();
    int* maxIndex_ptr = max_index.data_ptr<int32_t>();

    maxPool3dLauncher(Np, Mp, Nout, C, nnCount_ptr, nnIndex_ptr, in_ptr, out_ptr, maxIndex_ptr);
    return {output, max_index};
}


torch::Tensor MaxPool3dGrad(
    torch::Tensor grad_output,   // gradient of pooled features: concat_Mp * in_channels
    torch::Tensor input,         // input features: concat_Np * in_channels
    torch::Tensor max_index)     // the neighbor gives maximum activation: concat_Mp * in_channels
{
    CHECK_INPUT(input,2);
    CHECK_INPUT(grad_output,2);
    CHECK_INPUT(max_index,2);

    // get the dims required by computations
    int Np = input.size(0);       // number of input points
    int C  = input.size(1);       // number of input channels
    int Mp = grad_output.size(0); // number of output points

    TORCH_CHECK(max_index.dim()==2, "rank of max_index should be 2, i.e. (Mp, in_channels)");

    // get the c++ pointers to the input tensors
    const float* gradOut_ptr = grad_output.data_ptr<float>();
    const int* maxIndex_ptr  = max_index.data_ptr<int32_t>();

    // create an output tensor
    auto grad_input = torch::zeros({Np,C}, input.options());
    float* gradIn_ptr = grad_input.data_ptr<float>();

    maxPool3dGradLauncher(Np, Mp, C, maxIndex_ptr, gradOut_ptr, gradIn_ptr);
    return grad_input;
}


torch::Tensor AvgPool3d(
    torch::Tensor input,     // input features: concat_Np * in_channels
    torch::Tensor nn_count,  // number of neighbors: concat_Mp
    torch::Tensor nn_index)  // neighbor and kernel bin indices: Nout * 2
{
    CHECK_INPUT(input,2);
    CHECK_INPUT(nn_count,1);
    CHECK_INPUT(nn_index,2);

    // get the dims required by computations
    int Np   = input.size(0);    // number of input points
    int C    = input.size(1);    // number of input channels
    int Mp   = nn_count.size(0); // number of output points
    int Nout = nn_index.size(0); // number of neighbor pairs

    TORCH_CHECK(nn_count.dim()==1,"Dimension of nn_count should be 1.");
    TORCH_CHECK(nn_index.dim()==2,"Shape of nn_index should be (Nout,2).");

    // flatten the input tensors
    const float* in_ptr    = input.data_ptr<float>();
    const int* nnCount_ptr = nn_count.data_ptr<int32_t>();
    const int* nnIndex_ptr = nn_index.data_ptr<int32_t>();

    // Create an output tensor
    auto output = torch::zeros({Mp,C}, input.options());
    float* out_ptr = output.data_ptr<float>();

    avgPool3dLauncher(Np, Mp, Nout, C, nnCount_ptr, nnIndex_ptr, in_ptr, out_ptr);
    return output;
}


torch::Tensor AvgPool3dGrad(
    torch::Tensor grad_output, // gradient of pooled features: concat_Mp * in_channels
    torch::Tensor input,       // input features: concat_Np * in_channels
    torch::Tensor nn_count,    // number of neighbors: concat_Mp
    torch::Tensor nn_index)    // neighbor and kernel bin indices: Nout * 2
{
    CHECK_INPUT(input,2);
    CHECK_INPUT(grad_output,2);
    CHECK_INPUT(nn_count,1);
    CHECK_INPUT(nn_index,2);

    // get the dims required by computations
    int Np   = input.size(0);       // number of input points
    int C    = input.size(1);       // number of input channels
    int Mp   = grad_output.size(1); // number of output points
    int Nout = nn_index.size(0);    // number of neighbor pairs

    TORCH_CHECK(nn_count.dim()==1, "Dimension of nn_count should be 1.");
    TORCH_CHECK(nn_index.dim()==2, "Shape of nn_index should be (Nout,2).");

    // flatten the input tensors
    const float* gradOut_ptr = grad_output.data_ptr<float>();
    const int* nnCount_ptr   = nn_count.data_ptr<int32_t>();
    const int* nnIndex_ptr   = nn_index.data_ptr<int32_t>();

    // Create an output tensor
    auto grad_input = torch::zeros({Np,C}, input.options());
    float* gradIn_ptr = grad_input.data_ptr<float>();

    avgPool3dGradLauncher(Np, Mp, Nout, C, nnCount_ptr, nnIndex_ptr, gradOut_ptr, gradIn_ptr);
    return grad_input;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("max_forward",  &MaxPool3d,     "MaxPool3d forward (CUDA)");
    m.def("max_backward", &MaxPool3dGrad, "MaxPool3d backward (CUDA)");
    m.def("avg_forward",  &AvgPool3d,     "AvgPool3d forward (CUDA)");
    m.def("avg_backward", &AvgPool3dGrad, "AvgPool3d backward (CUDA)");
}