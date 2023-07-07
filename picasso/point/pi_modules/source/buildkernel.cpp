#include <torch/extension.h>
#include <vector>
//#include <iostream>

typedef int64_t LLint;

// CUDA forward and backward declarations
void sphericalKernelLauncher(int Nout, int n, int p, int q, float radius, const float* database,
                             const float* query, const int* nnIndex, const float* nnDist, int* filtIndex);
void fuzzySphericalKernelLauncher(int Nout, int n, int p, int q, float radius, const float* database, const float* query,
                                  const int* nnIndex, const float* nnDist, int* filtIndex, float* filtCoeff);

// C++ interface
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_DIM(x,dims) TORCH_CHECK(x.dim()==dims, #x " must have dimension of " #dims)
#define CHECK_INPUT(x,dims) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_DIM(x,dims);


torch::Tensor SphericalKernel(
    torch::Tensor database,    // database points: concat_Np * 3 (x,y,z)
    torch::Tensor query,       // query points: concat_Mp * 3
    torch::Tensor nn_index,    // neighbor and kernel bin indices: Nout * 2
    torch::Tensor nn_dist,     // distance to the neighbors: Nout
    float radius,              // range search radius
    int n_azim,                // division along azimuth direction
    int p_elev,                // division along elevation direction
    int q_radi)                // division along radius direction
{
    CHECK_INPUT(database,2);
    CHECK_INPUT(query,2);
    CHECK_INPUT(nn_index,2);
    CHECK_INPUT(nn_dist,1);
    
    TORCH_CHECK(radius>0, "Range search requires radius>0");
    TORCH_CHECK(n_azim>2 && n_azim%2==0, "Need n_azim>2 and n_azim%2==0");
    TORCH_CHECK(p_elev>0 && p_elev%2==0, "Need p_elev>0 and p_elev%2==0");
    TORCH_CHECK(q_radi>0, "Need q_radi>0");

    // get the dims required by computations
    int Np = database.size(0); // number of database points
    int Mp = query.size(0);    // number of query points
    int Nout = nn_index.size(0); // number of neighbor pairs

    TORCH_CHECK(database.dim()==2 && database.size(1)==3,
                "Shape of database points requires to be (Np, 3)");
    TORCH_CHECK(query.dim()==2 && query.size(1)==3,
                "Shape of query points requires to be (Mp, 3)");
    TORCH_CHECK(nn_index.dim()==2  && nn_index.size(1)==2,
                "Shape of database points requires to be (Nout, 2)");

    // get pointers to the input tensors
    const float* database_ptr = database.data_ptr<float>();
    const float* query_ptr = query.data_ptr<float>();
    const int* nnIndex_ptr = nn_index.data_ptr<int32_t>();
    const float* nnDist_ptr = nn_dist.data_ptr<float>();

    // create an output tensor
    auto filt_index = torch::zeros({Nout}, nn_index.options());
    int* filtIndex_ptr = filt_index.data_ptr<int32_t>();

    sphericalKernelLauncher(Nout, n_azim, p_elev, q_radi, radius, database_ptr,
                            query_ptr, nnIndex_ptr, nnDist_ptr, filtIndex_ptr);
    return filt_index;
}


std::vector<torch::Tensor> FuzzySphericalKernel(
    torch::Tensor database, // database points: concat_Np * 3 (x,y,z)
    torch::Tensor query,    // query points: concat_Mp * 3
    torch::Tensor nn_index, // neighbor and kernel bin indices: (Nout, 2)
    torch::Tensor nn_dist,  // distance to the neighbors: Nout
    float radius,           // range search radius
    int n_azim,             // division along azimuth direction
    int p_elev,             // division along elevation direction
    int q_radi)             // division along radius direction
{
    CHECK_INPUT(database,2);
    CHECK_INPUT(query,2);
    CHECK_INPUT(nn_index,2);
    CHECK_INPUT(nn_dist,1);

    TORCH_CHECK(radius>0, "Range search requires radius>0");
    TORCH_CHECK(n_azim>2 && n_azim%2==0, "Need n_azim>2 and n_azim%2==0");
    TORCH_CHECK(p_elev>0 && p_elev%2==0, "Need p_elev>0 and p_elev%2==0");
    TORCH_CHECK(q_radi>0, "Need q_radi>0");

    // get the dims required by computations
    int Np   = database.size(0); // number of database points
    int Mp   = query.size(0);    // number of query points
    int Nout = nn_index.size(0); // neighbor pairs

    TORCH_CHECK(database.dim()==2 && database.size(1)==3,
                "Shape of database points requires to be (Np, 3)");
    TORCH_CHECK(query.dim()==2 && query.size(1)==3,
                "Shape of query points requires to be (Mp, 3)");
    TORCH_CHECK(nn_index.dim()==2  && nn_index.size(1)==2,
                "Shape of database points requires to be (Nout, 2)");

    // get pointers to the input tensors
    const float* database_ptr = database.data_ptr<float>();
    const float* query_ptr = query.data_ptr<float>();
    const int* nnIndex_ptr = nn_index.data_ptr<int32_t>();
    const float* nnDist_ptr = nn_dist.data_ptr<float>();

    // create output tensors
    auto filt_index = torch::zeros({Nout,3}, nn_index.options());
    auto filt_coeff = torch::zeros({Nout,3}, database.options());
    int* filtIndex_ptr = filt_index.data_ptr<int32_t>();
    float* filtCoeff_ptr = filt_coeff.data_ptr<float>();

    fuzzySphericalKernelLauncher(Nout, n_azim, p_elev, q_radi, radius, database_ptr, query_ptr,
                                 nnIndex_ptr, nnDist_ptr, filtIndex_ptr, filtCoeff_ptr);
    return {filt_index, filt_coeff};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("SPH3D",&SphericalKernel, "SphericalKernel (CUDA)");
    m.def("fuzzySPH3D", &FuzzySphericalKernel, "FuzzySphericalKernel (CUDA)");
}












