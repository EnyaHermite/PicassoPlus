#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
//#include <iostream>


// CUDA forward and backward declarations
void maxPool3dLauncher(int nvIn, int C, const int* vtReplace, const int* vtMap,
                       const float* input, float* output, int* maxIndex);
void maxPool3dGradLauncher(int nvOut, int C, const int* maxIndex,
                           const float* gradOutput, float* gradInput);
void avgPool3dLauncher(int Nv, int C, const int* vtReplace, const int* vtMap,
                       const float* input, float* output);
void avgPool3dGradLauncher(int nvIn, int C, const int* vtReplace, const int* vtMap,
                           const float* gradOutput, float* gradInput);
void weightedPool3dLauncher(int nvIn, int C, const int* vtReplace, const int* vtMap,
                            const float* input, const float* weight, float* output);
void weightedPool3dGradLauncher(int nvIn, int C, const int* vtReplace, const int* vtMap,
                                const float* gradOutput, const float* weight, float* gradInput);

// C++ interface
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_DIM(x,dims) TORCH_CHECK(x.dim()==dims, #x " must have dimension of " #dims)
#define CHECK_INPUT(x,dims) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_DIM(x,dims);


std::vector<torch::Tensor> MeshMaxPool3d(
    torch::Tensor input,       // batch_npoints * in_channels
    torch::Tensor vt_replace,  // batch_npoints
    torch::Tensor vt_map,      // batch_npoints
    torch::Tensor vt_out)      // batch_mpoints * D
{
    CHECK_INPUT(input,2);
    CHECK_INPUT(vt_replace,1);
    CHECK_INPUT(vt_map,1);
    CHECK_INPUT(vt_out,2);

    // get the dims required by computations
    int nvIn  = input.size(0);   // number of input points
    int C     = input.size(1);   // number of input channels
    int nvOut = vt_out.size(0);   // number of output points

    TORCH_CHECK(input.dim()==2, "rank of input should be 2, i.e. (batch_npoints,channels)");

    // get c++ pointers to the input tensors
    const float* input_ptr   = input.data_ptr<float>();
    const int* vtReplace_ptr = vt_replace.data_ptr<int32_t>();
    const int* vtMap_ptr     = vt_map.data_ptr<int32_t>();

    // create output tensors
    auto output    = torch::zeros({nvOut,C}, input.options());
    auto max_index = torch::zeros({nvOut,C}, vt_map.options());
    float* output_ptr = output.data_ptr<float>();
    int* maxIndex_ptr = max_index.data_ptr<int32_t>();

    maxPool3dLauncher(nvIn, C, vtReplace_ptr, vtMap_ptr, input_ptr, output_ptr, maxIndex_ptr);
    return{output, max_index};
}
torch::Tensor MeshMaxPool3dGrad(
    torch::Tensor grad_output, // batch_mpoints * in_channels
    torch::Tensor input,       // batch_npoints * in_channels
    torch::Tensor max_index)   // batch_mpoints * in_channels
{
    CHECK_INPUT(input,2);
    CHECK_INPUT(grad_output,2);
    CHECK_INPUT(max_index,2);

    // get the dims required by computations
    int nvIn  = input.size(0);         // number of input points
    int C     = input.size(1);         // number of input channels
    int nvOut = grad_output.size(0);   // number of output points

    TORCH_CHECK(max_index.dim()==2, "rank of max_index should be 2, i.e. (batch_mpoints, in_channels)");

    // get c++ pointers to the input tensors
    const float* gradOut_ptr = grad_output.data_ptr<float>();
    const int* maxIndex_ptr  = max_index.data_ptr<int32_t>();

    // Create an output tensor
    auto grad_input = torch::zeros({nvIn,C}, input.options());
    float* gradIn_ptr = grad_input.data_ptr<float>();

    maxPool3dGradLauncher(nvOut, C, maxIndex_ptr, gradOut_ptr, gradIn_ptr);
    return grad_input;
}

torch::Tensor MeshAvgPool3d(
    torch::Tensor input,        // batch_npoints * in_channels
    torch::Tensor vt_replace,   // batch_npoints
    torch::Tensor vt_map,       // batch_npoints
    torch::Tensor vt_out)        // batch_mpoints * D
{
    CHECK_INPUT(input,2);
    CHECK_INPUT(vt_replace,1);
    CHECK_INPUT(vt_map,1);
    CHECK_INPUT(vt_out,2);

    // get the dims required by computations
    int nvIn  = input.size(0);   // number of input points
    int C     = input.size(1);   // number of input channels
    int nvOut = vt_out.size(0);   // number of output points

    TORCH_CHECK(input.dim()==2, "rank of input should be 2, i.e. (batch_npoints,channels)");

    // get c++ pointers to the input tensors
    const float* input_ptr   = input.data_ptr<float>();
    const int* vtReplace_ptr = vt_replace.data_ptr<int32_t>();
    const int* vtMap_ptr     = vt_map.data_ptr<int32_t>();

    // Create an output tensor
    auto output = torch::zeros({nvOut,C}, input.options());
    float* output_ptr = output.data_ptr<float>();

    avgPool3dLauncher(nvIn, C, vtReplace_ptr, vtMap_ptr, input_ptr, output_ptr);
    return output;
}
torch::Tensor  MeshAvgPool3dGrad(
    torch::Tensor grad_output, // batch_mpoints * in_channels
    torch::Tensor input,       // batch_npoints * in_channels
    torch::Tensor vt_replace,  // batch_npoints
    torch::Tensor vt_map)      // batch_npoints
{
    CHECK_INPUT(input,2);
    CHECK_INPUT(grad_output,2);
    CHECK_INPUT(vt_replace,1);
    CHECK_INPUT(vt_map,1);

    // get the dims required by computations
    int nvIn = input.size(0);   // number of input points
    int C    = input.size(1);   // number of input channels

    TORCH_CHECK(input.dim()==2, "rank of input should be 2, i.e. (batch_npoints,channels)");

    // get c++ pointers to the input tensors
    const float* input_ptr   = input.data_ptr<float>();
    const float* gradOut_ptr = grad_output.data_ptr<float>();
    const int* vtReplace_ptr = vt_replace.data_ptr<int32_t>();
    const int* vtMap_ptr     = vt_map.data_ptr<int32_t>();

    // Create an output tensor
    auto grad_input = torch::zeros({nvIn,C}, input.options());
    float* gradIn_ptr = grad_input.data_ptr<float>();

    avgPool3dGradLauncher(nvIn, C, vtReplace_ptr, vtMap_ptr, gradOut_ptr, gradIn_ptr);
    return grad_input;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("max_forward",  &MeshMaxPool3d,     "MeshMaxPool3d forward (CUDA)");
  m.def("max_backward", &MeshMaxPool3dGrad, "MeshMaxPool3d backward (CUDA)");
  m.def("avg_forward",  &MeshAvgPool3d,     "MeshAvgPool3d forward (CUDA)");
  m.def("avg_backward", &MeshAvgPool3dGrad, "MeshAvgPool3d backward (CUDA)");
}