#include <torch/extension.h>
#include <vector>
//#include <iostream>

typedef long int LLint;

// CUDA forward and backward declarations
LLint countSphereNeighborLauncher(int B, int Np, int Mp, float radius, int nnSample, const float* database,
                                  const float* query, const int* nvDatabase, const int* nvQuery, LLint* cntInfo);
void buildSphereNeighborLauncher(int B, int Np, int Mp, float radius, const float* database,
                                 const float* query, const int* nvDatabase, const int* nvQuery,
                                 const LLint* cntInfo, int* nnIndex, float* nnDist);
LLint countCubeNeighborLauncher(int B, int Np, int Mp, float length, int nnSample, const float* database,
                               const float* query, const int* nvDatabase, const int* nvQuery, LLint* cntInfo);
void buildCubeNeighborLauncher(int B, int Np, int Mp, float length, int gridSize, const float* database,
                               const float* query, const int* nvDatabase, const int* nvQuery,
                               const LLint* cntInfo, int* nnIndex);
void buildNearestNeighborLauncher(int B, int Np, int Mp, const float* database, const float* query,
                                  const int* nvDatabase, const int* nvQuery, int* nnIndex, float* nnDist);


// C++ interface
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_DIM(x,dims) TORCH_CHECK(x.dim()==dims, #x " must have dimension of " #dims)
#define CHECK_INPUT(x,dims) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_DIM(x,dims);


std::vector<torch::Tensor> BuildSphereNeighbor(
    torch::Tensor database,     // database points: concat_Np * 3
    torch::Tensor query,        // query points: concat_Mp * 3
    torch::Tensor nv_database,  // batch: each element is the vertex number of a database sample
    torch::Tensor nv_query,     // batch: each element is the vertex number of a query sample
    float radius,               // range search radius
    int nn_sample)              // max number of neighbors sampled in the range
{
    CHECK_INPUT(database,2);
    CHECK_INPUT(query,2);
    CHECK_INPUT(nv_database,1);
    CHECK_INPUT(nv_query,1);

    TORCH_CHECK(radius>0, "Range search requires radius>0");
    TORCH_CHECK(nn_sample>0, "BuildSphereNeighbor requires nn_sample>0");

    // get the dims required by computations
    int Np = database.size(0);    // number of database points
    int Mp = query.size(0);       // number of query points
    int B  = nv_database.size(0); // batch size

    TORCH_CHECK(database.dim()==2 && database.size(1)==3,
                "Shape of database points requires to be (Np, 3)");
    TORCH_CHECK(query.dim()==2 && query.size(1)==3,
                "Shape of query points requires to be (Mp, 3)");
    TORCH_CHECK(nv_database.dim()==1 && nv_query.dim()==1 && nv_database.size(0)==nv_query.size(0),
                "Shape of nv_database and nv_query should be identical");

    // get pointers to the input tensors
    const float* database_ptr   = database.data_ptr<float>();
    const float* query_ptr      = query.data_ptr<float>();
    const int*   nvDatabase_ptr = nv_database.data_ptr<int32_t>();
    const int*   nvQuery_ptr    = nv_query.data_ptr<int32_t>();

    // create an output tensor
    auto cnt_info = torch::zeros({Mp}, database.options().dtype(torch::kInt64));
    LLint* cntInfo_ptr = cnt_info.data_ptr<long int>();
    LLint Nout = countSphereNeighborLauncher(B, Np, Mp, radius, nn_sample,
                                             database_ptr, query_ptr, nvDatabase_ptr, nvQuery_ptr, cntInfo_ptr);

    auto nn_index = torch::zeros({Nout,2}, database.options().dtype(torch::kInt32));
    auto nn_dist  = torch::zeros({Nout},   database.options());
    int*   nnIndex_ptr = nn_index.data_ptr<int32_t>();
    float* nnDist_ptr  = nn_dist.data_ptr<float>();

    buildSphereNeighborLauncher(B, Np, Mp, radius, database_ptr, query_ptr,
                                nvDatabase_ptr, nvQuery_ptr, cntInfo_ptr, nnIndex_ptr, nnDist_ptr);
    return {cnt_info, nn_index, nn_dist};
}


std::vector<torch::Tensor> BuildCubeNeighbor(
    torch::Tensor database,     // database points: concat_Np * 3
    torch::Tensor query,        // query points: concat_Mp * 3
    torch::Tensor nv_database,  // batch: each element is the vertex number of a database sample
    torch::Tensor nv_query,     // batch: each element is the vertex number of a query sample
    float length,               // cube size: length * length * length
    int nn_sample,              // max number of neighbors sampled in the range
    int grid_size)              // division along azimuth direction
{
    CHECK_INPUT(database,2);
    CHECK_INPUT(query,2);
    CHECK_INPUT(nv_database,1);
    CHECK_INPUT(nv_query,1);

    TORCH_CHECK(length>0,    "Cube size requires length>0");
    TORCH_CHECK(nn_sample>0, "BuildSphereNeighbor requires nn_sample>0");
    TORCH_CHECK(grid_size>0, "Need grid_size>0");

    // get the dims required by computations
    int Np = database.size(0);    // number of database points
    int Mp = query.size(0);       // number of query points
    int B  = nv_database.size(0); // batch size

    TORCH_CHECK(database.dim()==2 && database.size(1)==3,
                "Shape of database points requires to be (Np, 3)");
    TORCH_CHECK(query.dim()==2 && query.size(1)==3,
                "Shape of query points requires to be (Mp, 3)");
    TORCH_CHECK(nv_database.dim()==1 && nv_query.dim()==1 && nv_database.size(0)==nv_query.size(0),
                "Shape of nv_database and nv_query should be identical");

    // get pointers to the input tensors
    const float* database_ptr   = database.data_ptr<float>();
    const float* query_ptr      = query.data_ptr<float>();
    const int*   nvDatabase_ptr = nv_database.data_ptr<int32_t>();
    const int*   nvQuery_ptr    = nv_query.data_ptr<int32_t>();

    // create an output tensor
    auto cnt_info = torch::zeros({Mp}, database.options().dtype(torch::kInt64));
    LLint* cntInfo_ptr = cnt_info.data_ptr<long int>();
    LLint Nout = countCubeNeighborLauncher(B, Np, Mp, length, nn_sample, database_ptr, query_ptr,
                                           nvDatabase_ptr, nvQuery_ptr, cntInfo_ptr);

    auto nn_index = torch::zeros({Nout,3}, database.options().dtype(torch::kInt32));
    int* nnIndex_ptr = nn_index.data_ptr<int32_t>();

    buildCubeNeighborLauncher(B, Np, Mp, length, grid_size, database_ptr, query_ptr,
                              nvDatabase_ptr, nvQuery_ptr, cntInfo_ptr, nnIndex_ptr);
    return {cnt_info, nn_index};
}


std::vector<torch::Tensor> BuildNearestNeighbor(
    torch::Tensor database,     // database points: concat_Np * 3
    torch::Tensor query,        // query points: concat_Mp * 3
    torch::Tensor nv_database,  // batch: each element is the vertex number of a database sample
    torch::Tensor nv_query)     // batch: each element is the vertex number of a query sample
{
    CHECK_INPUT(database,2);
    CHECK_INPUT(query,2);
    CHECK_INPUT(nv_database,1);
    CHECK_INPUT(nv_query,1);

    // get the dims required by computations
    int Np = database.size(0);    // number of database points
    int Mp = query.size(0);       // number of query points
    int B  = nv_database.size(0); // batch size

    TORCH_CHECK(database.dim()==2 && database.size(1)==3,
                "Shape of database points requires to be (Np, 3)");
    TORCH_CHECK(query.dim()==2 && query.size(1)==3,
                "Shape of query points requires to be (Mp, 3)");
    TORCH_CHECK(nv_database.dim()==1 && nv_query.dim()==1 && nv_database.size(0)==nv_query.size(0),
                "Shape of nv_database and nv_query should be identical");

    const int nn_out = 3; // find 3 nearest neighbors

    // get pointers the input tensors
    const float* database_ptr   = database.data_ptr<float>();
    const float* query_ptr      = query.data_ptr<float>();
    const int*   nvDatabase_ptr = nv_database.data_ptr<int32_t>();
    const int*   nvQuery_ptr    = nv_query.data_ptr<int32_t>();

    // create output tensors
    auto nn_index = torch::zeros({Mp,nn_out}, database.options().dtype(torch::kInt32));
    auto nn_dist  = torch::zeros({Mp,nn_out}, database.options());
    int* nnIndex_ptr  = nn_index.data_ptr<int32_t>();
    float* nnDist_ptr = nn_dist.data_ptr<float>();

    buildNearestNeighborLauncher(B, Np, Mp, database_ptr, query_ptr,
                                 nvDatabase_ptr, nvQuery_ptr, nnIndex_ptr, nnDist_ptr);
    return {nn_index, nn_dist};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("range",&BuildSphereNeighbor, "BuildSphereNeighbor (CUDA)");
    m.def("cube", &BuildCubeNeighbor, "BuildCubeNeighbor (CUDA)");
    m.def("knn3",  &BuildNearestNeighbor, "BuildNearestNeighbor (CUDA)");
}



