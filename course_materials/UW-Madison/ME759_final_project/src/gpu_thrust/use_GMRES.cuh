#ifndef USE_GMRES_DEF
#define USE_GMRES_DEF

#include "matrix.cuh"
#include "vector.cuh"

using namespace gpu_thrust;

int useGMRES();
int useGMRES2();
int useGMRES3();
int useGMRES_n256();

#endif