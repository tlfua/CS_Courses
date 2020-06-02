#include "use_matrix.h"

int use_matrix()
{
    int m = 4, n = 4;
    matrix A(m, n);

    std::cout << "Matrix A has " << rows(A) << " rows" <<"\n";
    std::cout << "Matrix A has " << columns(A) << " columns" <<"\n\n";
    std::cout << "The newly created matrix, A looks like: \n\n";
    std::cout << A;
    
    
    std::cout << "Now, we will assign some values to the entries: \n\n";
    for (int i = 1; i <= m; i++){
        for (int j = 1; j <= n; j++){
            A(i, j) = i + j;
        }
    }
    std::cout << A;
    std::cout << "The (1,1) entry of matrix A is: " << A(1,1) << "\n\n";
    
    std::cout << "matrix, B has 5 times the entries of A: \n\n";
    matrix B(m,n);
    for (int i = 1; i <= m; i++){
        for (int j = 1; j <= n; j++){
            B(i, j) = 5 * (i + j);
        }
    }
    std::cout << B;

    std::cout << "Add together the two matrices (A+B): \n\n";
    std::cout << A + B;
    std::cout << "Subtract the matrices (A−B): \n\n";
    std::cout << A - B;
    
    std::cout << "Create matrix C = A + A + A using the '=', \n\n";
    matrix C(m,n);
    C = A + A + A;
    std::cout << C;

    std::cout << "Create matrix D such that D = +C \n\n";
    matrix D(m,n);
    D = +C;
    std::cout << D;

    std::cout << "Create matrix E such that E = -C \n\n";
    matrix E(m,n);
    E = -C;
    std::cout << E;

    std::cout << "Try copying the above matrix to be F: \n\n";
    matrix F(m,n);
    F = matrix(E);
    std::cout << F;

    std::cout << "G = A is automatically sized: \n\n";
    matrix G = A;
    std::cout << G;

    std::cout << "Create matrix H = 6*G: \n\n";
    matrix H = 6 * G;
    std::cout << H;

    std::cout << "Create matrix J = H*0.5: \n\n";
    matrix J = H * 0.5;
    std::cout << J;

    std::cout << "Overwrite matrix B = J / 10: \n\n";
    B = J / 10;
    std::cout << B;

    std::cout << "Create a 'vector', x: \n\n";
    matrix x(n, 1);
    for (int i = 1; i <= n; ++i){
        x(i, 1) = i;
    }

    std::cout << x;
    std::cout << "Multiply the Matrix A by x: (Ax) \n\n";
    std::cout << A * x;

    //////////////////////////// Vector Stuff
    std::cout << "Create an empty vector using default constructor \n\n";
    vector a;
    std::cout << "Vector a has " << rows(a) << "rows \n";
    std::cout << "Vector a has " << columns(a) << "columns \n\n";

    std::cout << "Create a vector of size " << n << "\n\n";
    vector b(n);
    std::cout << "Vector b has " << rows(b) << "rows \n";
    std::cout << "Vector b has " << columns(b) << "columns \n\n";
    std::cout << b;

    std::cout << "Put some values into the entries: \n\n";
    for (int i = 1; i <= n; i++){
        b(i) = i;
    }
    std::cout << b;

    std::cout << "Multiply matrix A by vector b \n\n";
    std::cout << A * b;

    std::cout << "Create a 7 x 7 identity matrix \n\n";
    B = eye(7);
    std::cout << B;

    matrix T(4,4);
    A = T;
    std::cout << "Perform GE w. PP on the following 4x4 matrix: \n\n";
    A(1,1)=2;  A(1,2) = 1;  A(1,3) = 1;  A(1,4) = 0;
    A(2,1)=4;  A(2,2) = 3;  A(2,3) = 3;  A(2,4) = 1;
    A(3,1)=8;  A(3,2) = 7;  A(3,3) = 9;  A(3,4) = 5;
    A(4,1)=6;  A(4,2) = 7;  A(4,3) = 9;  A(4,4) = 8;
    std::cout << "Matrix A is: \n\n" << A;
    std::cout << "Vector b is: \n\n" << b << "Solve x=b/A: \n\n";
    x = b / A;
    std::cout << "The solution to the problem, x is: \n\n" << x;
    std::cout << "And as a check, multiply A*x: \n\n";
    std::cout << A * x;

    std::cout << "Try another example: \n\n";
    matrix AA(3,3);
    AA(1,1)=3;  AA(1,2) = 17;  AA(1,3) = 10;
    AA(2,1)=2;  AA(2,2) = 4;   AA(2,3) = -2;
    AA(3,1)=6;  AA(3,2) = 18;  AA(3,3) = -12;
    vector c(3);
    c(1)=1;  c(2)=2;  c(3) =3;
    std::cout << "Matrix AA is: \n\n" << AA;
    std::cout << "Vector c is \n\n" << c;
    matrix y = c / AA;
    std::cout << "The solution to the problem, y is: \n\n" << y;
    std::cout << "And as a check, multiply A*y: \n\n";
    std::cout << AA * y;

    std::cout << "org A: \n\n";
    std::cout << A;
    A = resize(A, 10, 3);
    std::cout << "after resizing, A: \n\n";
    std::cout << A;

    std::cout << "AA: \n\n";
    std::cout << AA;
    std::cout << "transposed AA: \n\n";
    std::cout<< ~AA;

    vector d(10);
    for (int i=1; i <= 10; i++){
        d(i) = i;
    }
    std::cout << d;
    std::cout<< ~d * d;

    exit(0);
}