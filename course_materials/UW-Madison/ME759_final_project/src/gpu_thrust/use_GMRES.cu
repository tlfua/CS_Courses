#include "use_GMRES.cuh"
#include "matrix.cuh"
#include "vector.cuh"

#include <cstdlib>
#include <ctime>
#include <cmath>

int useGMRES()
{
    matrix A(5,5);
    A.set(1, 1, 0.8780);  A.set(1, 2, 0.8316);  A.set(1, 3, 0.2663);  A.set(1, 4, 0.9787);  A.set(1, 5, 0.0239);
    A.set(2, 1, 0.1159);  A.set(2, 2, 0.2926);  A.set(2, 3, 0.2626);  A.set(2, 4, 0.7914);  A.set(2, 5, 0.2085);
    A.set(3, 1, 0.9857);  A.set(3, 2, 0.5109);  A.set(3, 3, 0.5826);  A.set(3, 4, 0.2115);  A.set(3, 5, 0.2943);
    A.set(4, 1, 0.8573);  A.set(4, 2, 0.7512);  A.set(4, 3, 0.4431);  A.set(4, 4, 0.9486);  A.set(4, 5, 0.3660);
    A.set(5, 1, 0.4416);  A.set(5, 2, 0.3803);  A.set(5, 3, 0.4465);  A.set(5, 4, 0.0586);  A.set(5, 5, 0.8501);

    vector b(5);
    b.set(1, 1);  b.set(2, 1);  b.set(3, 1);  b.set(4, 1);  b.set(5, 1);

    vector x0(5);
    std::cout << "matrix A: \n\n" << A;
    std::cout << "vector b: \n\n" << b;
    std::cout << "initial guess vector, x0: \n\n" << x0;

    vector x(5);
    int iter_count;
    GMRES(A, b, x0, x, iter_count);

    std::cout << "GMRES solution, x is: \n\n" << x;
    std::cout << "Check: Ax = \n\n" << A * x;
    std::cout << "Backslash solution b/A \n\n" << b / A;

    return 1;
}

int useGMRES2()
{
    matrix A(4,4);
    A.set(1, 1, 0.8780);  A.set(1, 2, 0.8316);  A.set(1, 3, 0.2663);  A.set(1, 4, 0.9787);
    A.set(2, 1, 0.1159);  A.set(2, 2, 0.2926);  A.set(2, 3, 0.2626);  A.set(2, 4, 0.7914);
    A.set(3, 1, 0.9857);  A.set(3, 2, 0.5109);  A.set(3, 3, 0.5826);  A.set(3, 4, 0.2115);
    A.set(4, 1, 0.8573);  A.set(4, 2, 0.7512);  A.set(4, 3, 0.4431);  A.set(4, 4, 0.9486);

    vector b(4);
    b.set(1, 1);  b.set(2, 1);  b.set(3, 1);  b.set(4, 1);

    vector x0(4);

    std::cout << "matrix A: \n\n" << A;
    std::cout << "vector b: \n\n" << b;
    std::cout << "initial guess vector, x0: \n\n" << x0;

    vector x(4);
    int iter_count;
    GMRES(A, b, x0, x, iter_count);

    std::cout << "GMRES solution, x is: \n\n" << x;
    std::cout << "Check: Ax = \n\n" << A*x;
    std::cout << "Backslash solution b/A \n\n" << b/A;
    
    return 1;
}

int useGMRES3()
{
    matrix A(3, 3);
    A.set(1, 1, 0);  A.set(1, 2, 1);  A.set(1, 3, 0);
    A.set(2, 1, 1);  A.set(2, 2, 0);  A.set(2, 3, 0);
    A.set(3, 1, 0);  A.set(3, 2, 0);  A.set(3, 3, 1);

    vector b(3);
    b.set(1, 2);  b.set(2, 2);  b.set(3, 2);

    vector x0(3);

    std::cout << "matrix A: \n\n" << A;
    std::cout << "vector b: \n\n" << b;
    std::cout << "initial guess vector, x0: \n\n" << x0;

    vector x(3);
    int iter_count;
    GMRES(A, b, x0, x, iter_count);

    std::cout << "GMRES solution, x is: \n\n" << x;
    std::cout << "Check: Ax = \n\n" << A * x;
    
    return 1;
}

int useGMRES_n256()
{
    srand (time(NULL));

    int n ;
    // for this experiment, pow_ = 8 would take too long!
    for (int pow_ = 5; pow_ <= 5; ++pow_){
        
        n = pow(2, pow_);

        matrix A(n, n);
        for (int i = 1; i <= n; ++i){
	    for (int j = 1; j <= n; ++j){
	        A.set(i, j, (double)rand() / (double) RAND_MAX);
	    }
	}

        vector b(n);
        for (int i = 1; i <= n; ++i){
            b.set(i, 1);
        }

        vector x0(n);

        // std::cout << "matrix A: \n\n" << A;
        // std::cout << "vector b: \n\n" << b;
        // std::cout << "initial guess vector, x0: \n\n" << x0;

        vector x(n);
        int iter_count;
        GMRES(A, b, x0, x, iter_count);

        // std::cout << "GMRES solution, x is: \n\n" << x;
        std::cout << "Check: Ax = \n\n" << A * x;
        
    }
    
    return 1;
}
