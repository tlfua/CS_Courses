#include "use_GMRES.h"

int main()
{
    omp_set_num_threads(20);

    useGMRES();
    useGMRES2();
    useGMRES3();
    // useGMRES_n32();
    
    return 0;
}