#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
//#include <iostream>


// CUDA declarations
void meshDecimationLauncher(const bool useArea, const float wgtBnd,       //hyperparams
                            const int B, const int D, const int Nv, const int Nf, const int* nvIn, const int* mfIn,  //inputs
                            const int* nv2Remove, const float* vertexIn, const int* faceIn, const float* planeIn,    //inputs
                            int* nvOut, int* mfOut, float* vertexOut, int* faceOut, int* vtReplace, int* vtMap,      //ouputs
                            bool* isDegenerate);
void combineClustersLauncher(const int nvA, const int nvB, const int* repA, const int* mapA,
                             const int* repB, const int* mapB, int* repOut, int* mapOut);
void countVertexAdjfaceLauncher(int NfIn, const int* face, const int* vtMap, int* nfcount);


// C++ interface
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_DIM(x,dims) TORCH_CHECK(x.dim()==dims, #x " must have dimension of " #dims)
#define CHECK_INPUT(x,dims) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_DIM(x,dims);


std::vector<torch::Tensor> MeshDecimation(
    torch::Tensor vertexIn,     // concat_Nv * Dim(Dim>=3 at least xyz)
    torch::Tensor faceIn,       // concat_Nf * 3
    torch::Tensor planeIn,      // concat_Nf * 5, [normal,d,area] of faces
    torch::Tensor nvIn,         // batch: each element is the vertex number of an input sample
    torch::Tensor mfIn,         // batch: each element is the face   number of an input sample
    torch::Tensor nv2Remove,    // batch: expected number of vertices to remove in an input sample
    bool useArea, float wgtBnd)
{
    CHECK_INPUT(vertexIn,2);
    CHECK_INPUT(faceIn,2);
    CHECK_INPUT(planeIn,2);
    CHECK_INPUT(nvIn,1);
    CHECK_INPUT(mfIn,1);
    CHECK_INPUT(nv2Remove,1);

    // get the dims required by computations
    int Nv = vertexIn.size(0);  // number of input vertices/points
    int D  = vertexIn.size(1);  // dimension of input vertices
    int Nf = faceIn.size(0);    // number of input faces
    int B  = nvIn.size(0);      // batch size

    // conditional checks and validation
    TORCH_CHECK(vertexIn.dim()==2 && vertexIn.size(1)>=3,
                "The shape of input vertex should be (Nv, 3+)).");
    TORCH_CHECK(faceIn.dim()==2 && faceIn.size(1)==3,
                "The shape of input face should be (Nf, 3)).");
    TORCH_CHECK(planeIn.dim()==2 && planeIn.size(0)==Nf && planeIn.size(1)==5,
                "The shape of input face normals should be (Nf, 5)).");

//    std::cout<<"The input vertex dimension is "<<D<<","<<Nv<<","<<Nf<<","<<B<<","<<useArea<<","<<wgtBnd<<std::endl;

    // get c++ pointers to the input tensors
    const float* vertexIn_ptr = vertexIn.data_ptr<float>();
    const int* faceIn_ptr     = faceIn.data_ptr<int32_t>();
    const float* planeIn_ptr  = planeIn.data_ptr<float>();
    const int* nvIn_ptr       = nvIn.data_ptr<int32_t>();
    const int* mfIn_ptr       = mfIn.data_ptr<int32_t>();
    const int* nv2Remove_ptr  = nv2Remove.data_ptr<int32_t>();

    // create output tensors
    auto vertexOut    = torch::zeros({Nv,D}, vertexIn.options());
    auto faceOut      = torch::zeros({Nf,3}, faceIn.options());
    auto isDegenerate = torch::zeros({Nf}, faceIn.options().dtype(torch::kBool));
    auto vtReplace    = torch::zeros({Nv}, faceIn.options());
    auto vtMap        = torch::zeros({Nv}, faceIn.options());
    auto nvOut        = torch::zeros({B}, faceIn.options());
    auto mfOut        = torch::zeros({B}, faceIn.options());
    vertexOut    = vertexOut.contiguous();
    faceOut      = faceOut.contiguous();
    isDegenerate = isDegenerate.contiguous();
    vtReplace    = vtReplace.contiguous();
    vtMap        = vtMap.contiguous();
    nvOut        = nvOut.contiguous();
    mfOut        = mfOut.contiguous();
    float* vertexOut_ptr   = vertexOut.data_ptr<float>();
    int* faceOut_ptr       = faceOut.data_ptr<int32_t>();
    bool* isDegenerate_ptr = isDegenerate.data_ptr<bool>();
    int* vtReplace_ptr     = vtReplace.data_ptr<int32_t>();
    int* vtMap_ptr         = vtMap.data_ptr<int32_t>();
    int* nvOut_ptr         = nvOut.data_ptr<int32_t>();
    int* mfOut_ptr         = mfOut.data_ptr<int32_t>();

    meshDecimationLauncher(useArea, wgtBnd,
                           B, D, Nv, Nf, nvIn_ptr, mfIn_ptr, nv2Remove_ptr, vertexIn_ptr, faceIn_ptr, planeIn_ptr,
                           nvOut_ptr, mfOut_ptr, vertexOut_ptr, faceOut_ptr, vtReplace_ptr, vtMap_ptr, isDegenerate_ptr);
   return{vertexOut, faceOut, isDegenerate, vtReplace, vtMap, nvOut, mfOut};
}

std::vector<torch::Tensor> CombineClusters(
    torch::Tensor repA,      // concat_Nv: (vertex replacement: clustering information)
    torch::Tensor mapA,      // concat_Nv: (vertex mapping: map input to output vertices)
    torch::Tensor repB,      // concat_Nv: (vertex replacement: clustering information)
    torch::Tensor mapB)      // concat_Nv: (vertex mapping: map input to output vertices)
{
    CHECK_INPUT(repA,1);
    CHECK_INPUT(mapA,1);
    CHECK_INPUT(repB,1);
    CHECK_INPUT(mapB,1);

    // get the dims required by computations
    int nvA = repA.size(0);    // number of input faces
    int nvB = repB.size(0);    // number of input vertices/points

    // conditional checks and validation
    TORCH_CHECK(repA.size(0)==mapA.size(0), "The shape of repA and mapA should be identical.");
    TORCH_CHECK(repB.size(0)==mapB.size(0), "The shape of repB and mapB should be identical.");

    // get c++ pointers to the input tensors
    const int* repA_ptr = repA.data_ptr<int32_t>();
    const int* mapA_ptr = mapA.data_ptr<int32_t>();
    const int* repB_ptr = repB.data_ptr<int32_t>();
    const int* mapB_ptr = mapB.data_ptr<int32_t>();

    // Create an output tensor
    auto repOut = torch::zeros({nvA}, repA.options());
    auto mapOut = torch::zeros({nvA}, repA.options());
    repOut = repOut.contiguous();
    mapOut = mapOut.contiguous();
    int* repOut_ptr = repOut.data_ptr<int32_t>();
    int* mapOut_ptr = mapOut.data_ptr<int32_t>();

    combineClustersLauncher(nvA, nvB, repA_ptr, mapA_ptr, repB_ptr, mapB_ptr, repOut_ptr, mapOut_ptr);
    return {repOut, mapOut};

}
torch::Tensor  CountVertexAdjface(
    torch::Tensor faceIn,         // face vertex list: concat_NfIn * 3
    torch::Tensor vtMap,         // vertex mapping from input to output vertices: concat_NvIn
    torch::Tensor vtOut)         // vertices in decimated mesh: concat_NvOut * 3
{
    CHECK_INPUT(faceIn,2);
    CHECK_INPUT(vtMap,1);
    CHECK_INPUT(vtOut,2);

    // get the dims required by computations
    int NfIn  = faceIn.size(0);  // number of input faces
    int NvOut = vtOut.size(0);   // number of output vertices/points

    // conditional checks and validation
    TORCH_CHECK(faceIn.dim()==2 && faceIn.size(1)==3, "The shape of input face should be (NfIn, 3)).");
    TORCH_CHECK(vtOut.dim()==2 && vtOut.size(1)==3, "The shape of vertexOut should be (NvOut,3).");

    // get c++ pointers to the input tensors
    const int* faceIn_ptr = faceIn.data_ptr<int32_t>();
    const int* vtMap_ptr  = vtMap.data_ptr<int32_t>();

    // Create an output tensor
    auto nfCount = torch::zeros({NvOut}, faceIn.options());
    nfCount = nfCount.contiguous();
    int* nfCount_ptr = nfCount.data_ptr<int32_t>();

    countVertexAdjfaceLauncher(NfIn, faceIn_ptr, vtMap_ptr, nfCount_ptr);
    return nfCount;
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("simplify", &MeshDecimation, "MeshDecimation (CUDA)");
  m.def("combine_clusters", &CombineClusters, "CombineClusters (CUDA)");
  m.def("count_vertex_adjface", &CountVertexAdjface, "CountVertexAdjface (CUDA)");
}
