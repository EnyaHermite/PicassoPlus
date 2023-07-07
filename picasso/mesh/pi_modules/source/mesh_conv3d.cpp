#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
//#include <iostream>


// CUDA forward and backward declarations
void facet2vertexConv3dLauncher(int NfIn, int C, int r, int K, const int* vtMap, const int* nfCount, const int* face,
                                const float* coeff, const float* input, const float* filter, float* output);
void vertex2facetConv3dLauncher(int NfIn, int C, int r, const int* face,
                                const float* input, const float* filter, float* output);
void facet2facetConv3dLauncher(int NfIn, int F, int Cin, int Cout, const int* numInterior, const float* intrplWgts,
                               const float* input, const float* filter, float* output);

void facet2vertexConv3dGradLauncher(int NfIn, int C, int r, int K, const int* vtMap, const int* nfCount, const int* face,
                                    const float* coeff, const float* input, const float* filter, const float* gradOutput,
                                    float* gradInput, float* gradFilter);
void vertex2facetConv3dGradLauncher(int NfIn, int C, int r, const int* face, const float* input, const float* filter,
                                    const float* gradOutput, float* gradInput, float* gradFilter);
void facet2facetConv3dGradLauncher(int NfIn, int F, int Cin, int Cout, const int* numInterior, const float* intrplWgts,
                                   const float* input, const float* filter, const float* gradOutput,
                                   float* gradInput, float* gradFilter);


// C++ interface
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_DIM(x,dims) TORCH_CHECK(x.dim()==dims, #x " must have dimension of " #dims)
#define CHECK_INPUT(x,dims) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_DIM(x,dims);


torch::Tensor facet2vertexConv3d(
    torch::Tensor input,    // input face features: concat_NfIn * in_channels
    torch::Tensor filter,   // convolution: filter_size * in_channels * channel_multiplier
    torch::Tensor coeff,    // face coefficients: concat_NfIn * filter_size
    torch::Tensor face,     // face vertex list: concat_NfIn * 3
    torch::Tensor nf_count, // number of adjacent faces for each output vertex: concat_NvOut
    torch::Tensor vt_map)   // vertex mapping from input to output vertices: concat_NvIn
{
    CHECK_INPUT(input,2);
    CHECK_INPUT(filter,3);
    CHECK_INPUT(coeff,2);
    CHECK_INPUT(face,2);
    CHECK_INPUT(nf_count,1);
    CHECK_INPUT(vt_map,1);

    // get the dims required by computations
    int NfIn  = input.size(0);     // number of faces by concatenating batch samples
    int C     = input.size(1);     // number of input channels
    int K     = filter.size(0);    // number of template models in the filters
    int r     = filter.size(2);    // depthwise channel multiplier
    int NvOut = nf_count.size(0);  // number of output vertices/points
    int NvIn  = vt_map.size(0);    // number of input  vertices/points

    TORCH_CHECK(filter.size(1)==C, "Input Channel size error of the filter");

    // get c++ pointers to the input tensors
    const float* input_ptr   = input.data_ptr<float>();
    const float* filter_ptr  = filter.data_ptr<float>();
    const float* coeff_ptr   = coeff.data_ptr<float>();
    const int*   face_ptr    = face.data_ptr<int32_t>();
    const int*   nfCount_ptr = nf_count.data_ptr<int32_t>();
    const int*   vtMap_ptr   = vt_map.data_ptr<int32_t>();

    // Create an output tensor
    auto output = torch::zeros({NvOut,C*r},input.options());
    float* output_ptr = output.data_ptr<float>();

    facet2vertexConv3dLauncher(NfIn, C, r, K, vtMap_ptr, nfCount_ptr, face_ptr, coeff_ptr,
                               input_ptr, filter_ptr, output_ptr);

    return output;
}
std::vector<torch::Tensor> facet2vertexConv3dGrad(
    torch::Tensor grad_output,  // gradient of output vertex features: concat_NvOut * out_channels
    torch::Tensor input,        // input face features: concat_NfIn * in_channels
    torch::Tensor filter,       // convolution: filter_size * in_channels * channel_multiplier
    torch::Tensor coeff,        // face coefficients: concat_NfIn * filter_size
    torch::Tensor face,         // face vertex list: concat_NfIn * 3
    torch::Tensor nf_count,     // number of adjacent faces for each output vertex: concat_NvOut
    torch::Tensor vt_map)       // vertex mapping from input to output vertices: concat_NvIn
{
    CHECK_INPUT(input,2);
    CHECK_INPUT(filter,3);
    CHECK_INPUT(coeff,2);
    CHECK_INPUT(grad_output,2);
    CHECK_INPUT(face,2);
    CHECK_INPUT(nf_count,1);
    CHECK_INPUT(vt_map,1);

    // get the dims required by computations
    int NfIn  = input.size(0);     // number of faces by concatenating batch samples
    int C     = input.size(1);     // number of input channels
    int K     = filter.size(0);    // number of template models in the filters
    int r     = filter.size(2);    // depthwise channel multiplier
    int NvOut = nf_count.size(0);  // number of output vertices/points
    int NvIn  = vt_map.size(0);    // number of input  vertices/points

    TORCH_CHECK(filter.size(1)==C, "Input Channel size error of the filter");

    // get c++ pointers to the input tensors
    const float* input_ptr   = input.data_ptr<float>();
    const float* filter_ptr  = filter.data_ptr<float>();
    const float* coeff_ptr   = coeff.data_ptr<float>();
    const float* gradOut_ptr = grad_output.data_ptr<float>();
    const int*   face_ptr    = face.data_ptr<int32_t>();
    const int*   nfCount_ptr = nf_count.data_ptr<int32_t>();
    const int*   vtMap_ptr   = vt_map.data_ptr<int32_t>();

    // Create an output tensor
    auto grad_input  = torch::zeros({NfIn,C}, input.options());
    auto grad_filter = torch::zeros({K,C,r},  input.options());

    float* gradInput_ptr  = grad_input.data_ptr<float>();
    float* gradFilter_ptr = grad_filter.data_ptr<float>();

    facet2vertexConv3dGradLauncher(NfIn, C, r, K, vtMap_ptr, nfCount_ptr, face_ptr, coeff_ptr, input_ptr,
                                   filter_ptr, gradOut_ptr, gradInput_ptr, gradFilter_ptr);

    return {grad_input, grad_filter};
}
torch::Tensor vertex2facetConv3d(
    torch::Tensor input,    // input vertex features: concat_NvIn * in_channels
    torch::Tensor filter,   // convolution: 3 * in_channels * channel_multiplier
    torch::Tensor face)     // face vertex list: concat_NfIn * 3
{
    CHECK_INPUT(input,2);
    CHECK_INPUT(filter,3);
    CHECK_INPUT(face,2);

    // get the dims required by computations
    int NvIn = input.size(0);     // number of vertices by concatenating batch samples
    int C    = input.size(1);     // number of input channels
    int r    = filter.size(2);    // depthwise channel multiplier
    int NfIn = face.size(0);      // number of facets

    TORCH_CHECK(filter.size(0)==3 && filter.size(1)==C,
                "3 filters required, or input Channel size error of the filter");

    // get c++ pointers to the input tensors
    const float* input_ptr  = input.data_ptr<float>();
    const float* filter_ptr = filter.data_ptr<float>();
    const int*   face_ptr   = face.data_ptr<int32_t>();

    // create an output tensor
    auto output = torch::zeros({NfIn,C*r}, input.options());
    float* output_ptr = output.data_ptr<float>();

    vertex2facetConv3dLauncher(NfIn, C, r, face_ptr, input_ptr, filter_ptr, output_ptr);
    return output;
}
std::vector<torch::Tensor> vertex2facetConv3dGrad(
    torch::Tensor grad_output,  // gradient of output face features: concat_NfIn * out_channels
    torch::Tensor input,        // input vertex features: concat_NvIn * in_channels
    torch::Tensor filter,       // convolution: 3 * in_channels * channel_multiplier
    torch::Tensor face)         // face vertex list: concat_NfIn * 3
{
    CHECK_INPUT(input,2);
    CHECK_INPUT(filter,3);
    CHECK_INPUT(grad_output,2);
    CHECK_INPUT(face,2);

    // get the dims required by computations
    int NvIn = input.size(0);     // number of vertices by concatenating batch samples
    int C    = input.size(1);     // number of input channels
    int r    = filter.size(2);    // depthwise channel multiplier
    int NfIn = face.size(0);      // number of facets

    TORCH_CHECK(filter.size(0)==3 && filter.size(1)==C,
                "3 filters required, or input Channel size error of the filter");

    // get c++ pointers to the input tensors
    const float* input_ptr   = input.data_ptr<float>();
    const float* filter_ptr  = filter.data_ptr<float>();
    const float* gradOut_ptr = grad_output.data_ptr<float>();
    const int*   face_ptr    = face.data_ptr<int32_t>();

    // create output tensors
    auto grad_input  = torch::zeros({NvIn,C}, input.options());
    auto grad_filter = torch::zeros({3,C,r},  input.options());
    float* gradInput_ptr  = grad_input.data_ptr<float>();
    float* gradFilter_ptr = grad_filter.data_ptr<float>();

    vertex2facetConv3dGradLauncher(NfIn, C, r, face_ptr, input_ptr, filter_ptr,
                                   gradOut_ptr, gradInput_ptr, gradFilter_ptr);
    return {grad_input, grad_filter};
}


// apply positional encoding to the barycentric coordinates
torch::Tensor facet2facetConv3d(
    torch::Tensor input,            // input face features: [concat_NfIn*maxK, in_channels]
    torch::Tensor filter,           // convolution: [out_channels, in_channels, K]
    torch::Tensor bary_coeff,       // face Barycentric interpolation weights: [concat_NfIn*maxK, K]
    torch::Tensor num_texture)      // number of interior interpolated: concat_NfIn
{
    CHECK_INPUT(input,2);
    CHECK_INPUT(filter,3);
    CHECK_INPUT(bary_coeff,2);
    CHECK_INPUT(num_texture,1);

    // get the dims required by computations
    int NiK  = input.size(0);       // total number of interiors interpolated in NfIn faces
    int Cin  = input.size(1);       // number of input channels
    int Cout = filter.size(0);      // number of output channels
    int F    = filter.size(2);      // number of filters
    int NfIn = num_texture.size(0); // number of faces by concatenating batch samples

    TORCH_CHECK(filter.size(1)==Cin,"input Channel size error of the filter.");
    TORCH_CHECK(F<=50,"A maximum of 50 filters are supported in face2face convolution.");
    TORCH_CHECK(bary_coeff.size(1)==F, "Dimension of bary_coeff mismatches number of filters.");
    TORCH_CHECK(bary_coeff.dim()==2 && bary_coeff.size(0)==NiK, "The shape of bary_coeff should be (NiK, Dim)).");

    // get c++ pointers to the input tensors
    const float* input_ptr      = input.data_ptr<float>();
    const float* filter_ptr     = filter.data_ptr<float>();
    const float* baryCoeff_ptr  = bary_coeff.data_ptr<float>();
    const int*   numTexture_ptr = num_texture.data_ptr<int32_t>();

    // create an output tensor
    auto output  = torch::zeros({NfIn,Cout}, input.options());
    float* output_ptr = output.data_ptr<float>();

    facet2facetConv3dLauncher(NfIn, F, Cin, Cout, numTexture_ptr, baryCoeff_ptr,
                              input_ptr, filter_ptr, output_ptr);
    return output;
}
std::vector<torch::Tensor> facet2facetConv3dGrad(
    torch::Tensor grad_output,  // gradient of output face features: [concat_NfIn, out_channels]
    torch::Tensor input,        // input face features: [concat_NfIn*maxK, in_channels]
    torch::Tensor filter,       // convolution: [3, in_channels, out_channels]
    torch::Tensor bary_coeff,   // face Barycentric interpolation weights: [concat_NfIn*maxK, 3]
    torch::Tensor num_texture)  // number of interior interpolated: concat_NfIn
{
    CHECK_INPUT(input,2);
    CHECK_INPUT(filter,3);
    CHECK_INPUT(grad_output,2);
    CHECK_INPUT(bary_coeff,2);
    CHECK_INPUT(num_texture,1);

    // get the dims required by computations
    int NiK  = input.size(0);       // total number of interiors interpolated in NfIn faces
    int Cin  = input.size(1);       // number of input channels
    int Cout = filter.size(0);      // number of output channels
    int F    = filter.size(2);      // number of filters
    int NfIn = num_texture.size(0); // number of faces by concatenating batch samples

    TORCH_CHECK(filter.size(1)==Cin,"input Channel size error of the filter.");
    TORCH_CHECK(bary_coeff.size(1)==F, "Dimension of bary_coeff mismatches number of filters.");
    TORCH_CHECK(bary_coeff.dim()==2 && bary_coeff.size(0)==NiK, "The shape of bary_coeff should be (NiK, Dim)).");

    // get c++ pointers to the input tensors
    const float* input_ptr      = input.data_ptr<float>();
    const float* filter_ptr     = filter.data_ptr<float>();
    const float* gradOut_ptr    = grad_output.data_ptr<float>();
    const float* baryCoeff_ptr  = bary_coeff.data_ptr<float>();
    const int*   numTexture_ptr = num_texture.data_ptr<int32_t>();

    // create tensors
    auto grad_input  = torch::zeros({NiK,Cin}, input.options());
    auto grad_filter = torch::zeros({Cout,Cin,F}, input.options());
    float* gradInput_ptr  = grad_input.data_ptr<float>();
    float* gradFilter_ptr = grad_filter.data_ptr<float>();

    facet2facetConv3dGradLauncher(NfIn, F, Cin, Cout, numTexture_ptr, baryCoeff_ptr, input_ptr,
                                  filter_ptr, gradOut_ptr, gradInput_ptr, gradFilter_ptr);
    return {grad_input, grad_filter};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("facet2vertex_forward",  &facet2vertexConv3d, "facet2vertexConv3d forward (CUDA)");
  m.def("facet2vertex_backward", &facet2vertexConv3dGrad, "facet2vertexConv3d backward (CUDA)");
  m.def("vertex2facet_forward",  &vertex2facetConv3d, "vertex2facetConv3d forward (CUDA)");
  m.def("vertex2facet_backward", &vertex2facetConv3dGrad,"vertex2facetConv3d backward (CUDA)");
  m.def("facet2facet_forward",   &facet2facetConv3d, "facet2facetConv3d forward (CUDA)");
  m.def("facet2facet_backward",  &facet2facetConv3dGrad, "facet2facetConv3d backward (CUDA)");
}
