#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
//#include <iostream>

// CUDA forward and backward declarations
void interpolateLauncher(int nvIn, int C, int nvOut, const int* vtReplace, const int* vtMap,
                         const float* input, float* output);
void interpolateGradLauncher(int nvIn, int C, int nvOut, const int* vtReplace, const int* vtMap,
                             const float* gradOutput, float* gradInput);

// C++ interface
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_DIM(x,dims) TORCH_CHECK(x.dim()==dims, #x " must have dimension of " #dims)
#define CHECK_INPUT(x,dims) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_DIM(x,dims);


torch::Tensor MeshInterpolate(
    torch::Tensor input,       // batch_mpoints * in_channels
    torch::Tensor vt_replace,  // batch_npoints
    torch::Tensor vt_map)      // batch_npoints
{
    CHECK_INPUT(input,2);
    CHECK_INPUT(vt_replace,1);
    CHECK_INPUT(vt_map,1);

    // get the dims required by computations, (nvIn<nvOut)
    int nvIn  = input.size(0);   // number of input points
    int C     = input.size(1);   // number of input channels
    int nvOut = vt_map.size(0);  // number of output points

    TORCH_CHECK(input.dim()==2, "rank of input should be 2, i.e. (batch_mpoints,channels)");

    // flatten the input tensors
    const float* input_ptr     = input.data_ptr<float>();
    const int*   vtReplace_ptr = vt_replace.data_ptr<int32_t>();
    const int*   vtMap_ptr     = vt_map.data_ptr<int32_t>();

    // create an output tensor
    auto output = torch::zeros({nvOut,C}, input.options());
    float* output_ptr = output.data_ptr<float>();

    interpolateLauncher(nvIn, C, nvOut, vtReplace_ptr, vtMap_ptr, input_ptr, output_ptr);
    return output;
}
torch::Tensor MeshInterpolateGrad(
    torch::Tensor grad_output,  // batch_npoints * in_channels
    torch::Tensor input,        // batch_mpoints * in_channels
    torch::Tensor vt_replace,   // batch_npoints
    torch::Tensor vt_map)       // batch_npoints
{
    CHECK_INPUT(grad_output,2);
    CHECK_INPUT(input,2);
    CHECK_INPUT(vt_replace,1);
    CHECK_INPUT(vt_map,1);

    // get the dims required by computations, (nvIn<nvOut)
    int nvIn  = input.size(0);   // number of input points
    int C     = input.size(1);   // number of input channels
    int nvOut = vt_map.size(0);  // number of output points

    TORCH_CHECK(input.dim()==2, "rank of input should be 2, i.e. (batch_mpoints,channels)");

    // get array pointers to the input tensors
    const float* input_ptr   = input.data_ptr<float>();
    const float* gradOut_ptr = grad_output.data_ptr<float>();
    const int* vtReplace_ptr = vt_replace.data_ptr<int32_t>();
    const int* vtMap_ptr     = vt_map.data_ptr<int32_t>();

    // create an output tensor
    auto grad_input = torch::zeros({nvIn,C}, input.options());
    float* gradIn_ptr = grad_input.data_ptr<float>();

    interpolateGradLauncher(nvIn, C, nvOut, vtReplace_ptr, vtMap_ptr, gradOut_ptr, gradIn_ptr);
    return grad_input;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("forward",  &MeshInterpolate,     "MeshInterpolate forward (CUDA)");
  m.def("backward", &MeshInterpolateGrad, "MeshInterpolate backward (CUDA)");
}
