#include <torch/extension.h>
#include <vector>
//#include <iostream>


// CUDA declarations
int computeOutputSize(int B, const int* nvOut);
void farthestPointSampleLauncher(int B, int Np, const int* nvIn, const int* nvOut,
                                 const float* xyzIn, float* temp, int* indexOut);
void farthestPointSample3DLauncher(int b,int n,int m,const float * inp,float * temp,int * out);


// C++ interface
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_DIM(x,dims) TORCH_CHECK(x.dim()==dims, #x " must have dimension of " #dims)
#define CHECK_INPUT(x,dims) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_DIM(x,dims);


torch::Tensor FarthestPointSample(
    torch::Tensor xyz_in,       // concatenated point xyz: concat_Np * 3
    torch::Tensor nv_in,        // number of points in each input batch sample: batch_size
    torch::Tensor nv_out)       // number of points in each output batch sample: batch_size
{
    CHECK_INPUT(xyz_in,2);
    CHECK_INPUT(nv_in,1);
    CHECK_INPUT(nv_out,1);

    // get the dims required by computations
    int Np = xyz_in.size(0);
    int B  = nv_in.size(0);

    TORCH_CHECK(xyz_in.dim()==2 && xyz_in.size(1)==3, "FarthestPointSample expects (Np,3) inp shape");

    const float* xyzIn_ptr = xyz_in.data_ptr<float>();
    const int*   nvIn_ptr  = nv_in.data_ptr<int32_t>();
    const int*   nvOut_ptr = nv_out.data_ptr<int32_t>();

    int Mp = computeOutputSize(B, nvOut_ptr); // extract Mp from nvOut

    auto temp = torch::zeros({Np}, xyz_in.options());
    auto index_out = torch::zeros({Mp}, xyz_in.options().dtype(torch::kInt32));
    float* temp_ptr   = temp.data_ptr<float>();
    int* indexOut_ptr = index_out.data_ptr<int32_t>();

    farthestPointSampleLauncher(B, Np, nvIn_ptr, nvOut_ptr, xyzIn_ptr, temp_ptr, indexOut_ptr);
    return index_out;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("fps",      &FarthestPointSample,   "FarthestPointSample (CUDA)");
}



