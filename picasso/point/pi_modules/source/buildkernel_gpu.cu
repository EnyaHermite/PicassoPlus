#ifndef M_PI
#define M_PI           3.141592653589793F  /* pi */
#endif

#ifndef M_EPS
#define M_EPS          1.01e-3F             /* epsilon */
#endif

struct point3d
{
    float x=0, y=0, z=0;
};

// database:  concat_Np*3, (x,y,z)
// query:     concat_Mp*3, (x,y,z)
// nnCount:   concat_Mp
// nnIndex:   Nout*2
// nnDist:    Nout
// filtIndex: Nout
__global__ void build_spherical_kernel(int Nout, int n, int p, int q, const float radius, const float* database,
                                       const float* query, const int* nnIndex, const float* nnDist, int* filtIndex)
{
    int pairIdx = blockIdx.x*blockDim.x + threadIdx.x; // global face index in the batch

    if (pairIdx<Nout) // index must be in the legal range
    {
        int outIdx = nnIndex[pairIdx*2];
        int inIdx = nnIndex[pairIdx*2+1];

        // get the neighbor indices
        point3d ptQuery, pt, delta;

        ptQuery.x = query[outIdx*3];
        ptQuery.y = query[outIdx*3+1];
        ptQuery.z = query[outIdx*3+2];

        pt.x = database[inIdx*3];
        pt.y = database[inIdx*3+1];
        pt.z = database[inIdx*3+2];

        delta.x = pt.x - ptQuery.x;
        delta.y = pt.y - ptQuery.y;
        delta.z = pt.z - ptQuery.z;

        float dist = nnDist[pairIdx]; // the sqrt distance
        float dist2D = delta.x*delta.x + delta.y*delta.y;
        dist2D = sqrtf(dist2D);

        filtIndex[pairIdx] = 0;
        if (dist>(0.3*radius)) // update the bin index
        {
            float theta = atan2f(delta.y, delta.x);
            float phi = atan2f(delta.z, dist2D);

            theta = theta<M_PI?theta:(-M_PI);
            theta = theta>(-M_PI)?theta:(-M_PI);
            theta += M_PI;

            phi = phi<(M_PI/2)?phi:(M_PI/2);
            phi = phi>(-M_PI/2)?phi:(-M_PI/2);
            phi += M_PI/2;

            float alpha = theta*n/2/M_PI;
            float beta = phi*p/M_PI;
            float gamma = dist*q/radius;

            int nID = min(n-1, int(alpha));
            int pID = min(p-1, int(beta));
            int qID = min(q-1, int(gamma));

            filtIndex[pairIdx] = qID*p*n + pID*n + nID + 1;
        }
    }
}

// database:  concat_Np*3, (x,y,z)
// query:     concat_Mp*3, (x,y,z)
// nnIndex:   Nout*2
// nnDist:    Nout
// filtIndex: Nout*3
// filtCoeff: Nout*3
__global__ void build_fuzzy_spherical_kernel(int Nout, int n, int p, int q, const float radius, const float* database,
                                             const float* query, const int* nnIndex, const float* nnDist,
                                             int* filtIndex, float* filtCoeff)
{
    int pairIdx = blockIdx.x*blockDim.x + threadIdx.x; // global face index in the batch

    if (pairIdx<Nout) // index must be in the legal range
    {
        int outIdx = nnIndex[pairIdx*2];
        int inIdx = nnIndex[pairIdx*2+1];

        // get the neighbor indices
        point3d ptQuery, pt, delta;

        ptQuery.x = query[outIdx*3];
        ptQuery.y = query[outIdx*3+1];
        ptQuery.z = query[outIdx*3+2];

        pt.x = database[inIdx*3];
        pt.y = database[inIdx*3+1];
        pt.z = database[inIdx*3+2];

        delta.x = pt.x - ptQuery.x;
        delta.y = pt.y - ptQuery.y;
        delta.z = pt.z - ptQuery.z;

        float dist = nnDist[pairIdx]; // the sqrt distance
        float dist2D = delta.x*delta.x + delta.y*delta.y;
        dist2D = sqrtf(dist2D);

        float selfCoeff = max(0.0,1-dist/radius);
        filtIndex[pairIdx*3] = 0;
        filtCoeff[pairIdx*3] = selfCoeff; // self-loop coeffcient

        // compute the coefficients of fuzzy bins
        float theta = atan2f(delta.y, delta.x);
        float phi = atan2f(delta.z, dist2D);

        theta = theta<M_PI?theta:(-M_PI);
        theta = theta>(-M_PI)?theta:(-M_PI);
        theta += M_PI;

        phi = phi<(M_PI/2)?phi:(M_PI/2);
        phi = phi>(-M_PI/2)?phi:(-M_PI/2);
        phi += M_PI/2;

        int nID, pID[2], qID;
        float alpha, beta[2], gamma;

        alpha   = theta*n/2/M_PI;
        beta[0] = phi*p/M_PI;
        gamma   = dist*q/radius;

        nID    = min(n-1, int(alpha));
        pID[0] = min(p-1, int(beta[0]));
        qID    = min(q-1, int(gamma));

        beta[0] -= pID[0];

        pID[1] = beta[0]<0.5? max(0,pID[0]-1):min(p-1,pID[0]+1);

        int pIN = (pID[0]==pID[1])?1:2;
        beta[1] = (pIN==1)?0:abs(beta[0] - 0.5);
        beta[0] = 1 - beta[1];

        for(int px=0;px<pIN;px++)
        {
            int f = qID*p*n + pID[px]*n + nID + 1;
            filtIndex[pairIdx*3+px+1] = f;
            filtCoeff[pairIdx*3+px+1] = beta[px]*(1-selfCoeff);
        }
    }
}


void sphericalKernelLauncher(int Nout, int n, int p, int q, float radius, const float* database,
                             const float* query, const int* nnIndex, const float* nnDist, int* filtIndex)
{
    int numGrid = int(Nout/1024) + 1;  // Nout is the total number of neighbor pairs
    build_spherical_kernel<<<numGrid,1024>>>(Nout, n, p, q, radius, database, query, nnIndex, nnDist, filtIndex);

}

void fuzzySphericalKernelLauncher(int Nout, int n, int p, int q, float radius,
                                  const float* database, const float* query, const int* nnIndex,
                                  const float* nnDist, int* filtIndex, float* filtCoeff)
{
    int numGrid = int(Nout/1024) + 1;  // Nout is the total number of neighbor pairs
    build_fuzzy_spherical_kernel<<<numGrid,1024>>>(Nout, n, p, q, radius, database, query,
                                                   nnIndex, nnDist, filtIndex, filtCoeff);
}
