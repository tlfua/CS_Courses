#include "vector.h"

sequential::vector::vector(int no_of_elements)
        : matrix(no_of_elements, 1)
{}

sequential::vector sequential::operator+(const vector& A, const vector& B)
{
    int row_A = rows(A);
    int row_B = rows(B);

    if(row_A != row_B){
        std::cout << "Error: Matrices of different dimensions";
        std::cout << "Returned first argument";
        return A;
    } else {
        vector C(row_A);
        std::transform(std::begin(A.data), std::end(A.data), std::begin(B.data), std::begin(C.data), std::plus<double>());
        return C;
    }
}

sequential::vector sequential::operator-(const vector& A, const vector& B)
{
    int row_A = rows(A);
    int row_B = rows(B);

    if(row_A != row_B){
        std::cout << "Error: Matrices of different dimensions";
        std::cout << "Returned first argument";
        return A;
    } else {
        vector C(row_A);
        std::transform(std::begin(A.data), std::end(A.data), std::begin(B.data), std::begin(C.data), std::minus<double>());
        return C;
    }
}

sequential::vector sequential::operator*(const double& p, const vector& A)
{
    vector B(A.rows);
    std::transform(std::begin(A.data), std::end(A.data), std::begin(B.data), [p](auto& elem){ return elem * p; });
    return B;
}

sequential::vector sequential::operator*(const vector& A, const double& p)
{
    vector B(A.rows);
    std::transform(std::begin(A.data), std::end(A.data), std::begin(B.data), [p](auto& elem){ return elem * p; });
    return B;
}

sequential::vector sequential::operator/(const vector& A, const double& p)
{
    vector B(A.rows);
    std::transform(std::begin(A.data), std::end(A.data), std::begin(B.data), [p](auto& elem){ return elem / p; });
    return B;
}

sequential::vector sequential::operator+(const vector& A)
{
    return sequential::vector(A);
}

sequential::vector sequential::operator-(const vector& A)
{
    vector B(A.rows);
    std::transform(std::begin(A.data), std::end(A.data), std::begin(B.data), std::negate<double>());
    return B;
}

// Overloads (), so x(i) returns the ith entry
double& sequential::vector::operator()(int i)
{
    if (i < 1) {
        std::cout << "Error: Your index may be too small \n\n";
    } else if (i > rows) {
        std::cout << "Error: Your index may be too large \n\n";
    }
    return data[i - 1];
}

// sequential::vector sequential::mat2vec(sequential::matrix& A)
// {
//     sequential::vector v(rows(A));
//     for (int i = 1; i <= rows(A); i++){
//         v(i) = A(i, 1);
//     }
//     return v;
// }

double sequential::norm(sequential::vector& v, int p) {
    // define variables and initialise sum
    double res, sum = 0.0;

    for (int i=1; i <= rows(v); i++) {
        // floating point absolute value
        sum += pow(fabs(v(i)), p);
    }

    // std::reduce

    res = pow(sum, 1.0 / ((double)(p)));
    return res;
}

void sequential::GMRES(matrix& A, matrix& b, matrix& x0, matrix& x, int& k, double tol)
{
    vector r0 = mat2vec(b - A * x0);

    double normr0 = norm(r0);

    double residual = 1.0;

    vector v = r0 / normr0;

    // int k = 1;

    matrix J, Jtotal;
    Jtotal = eye(2);

    matrix H(1,1), Htemp, HH, bb(1,1), c, cc;
    matrix tempMat, V, Vold, hNewCol;
    vector w, vj(rows(v));

    bb(1,1) = normr0;

    V = v;
    k = 1;
    while (residual > tol){

        Vold = V;
      
        H = resize(H, k + 1, k);

        w = mat2vec(A * v);

        for (int j=1; j <= k; j++) {
            for (int i=1; i <= rows(V); i++) {
                vj(i) = V(i,j);
            }
            tempMat = ~vj * w;

            H(j,k) = tempMat(1,1);

            w = w - H(j,k) * vj;
        }

        H(k+1,k) = norm(w);

        v = w / H(k+1,k);

        V = resize(V,rows(V),k+1);

        for (int i=1; i <= rows(V); i++) {
            // copy entries of v to new column of V
            V(i,k+1) = v(i);
        }

        if (k==1) {
            // First pass through, Htemp=H
            Htemp = H;
        } else {
            // for subsequent passes, Htemp=Jtotal*H
            Jtotal = resize(Jtotal, k+1, k+1);
            Jtotal(k+1,k+1) = 1;
            Htemp = Jtotal * H;
        }

        // Form next Givens rotation matrix
        J = eye(k - 1);
        J = resize(J,k+1,k+1);

        J(k,k) = Htemp(k,k) / pow(pow(Htemp(k,k),2) + pow(Htemp(k+1,k),2),0.5);
        J(k,k+1) = Htemp(k+1,k) / pow(pow(Htemp(k,k),2) + pow(Htemp(k+1,k),2),0.5);
        J(k+1,k) = -Htemp(k+1,k) / pow(pow(Htemp(k,k),2) + pow(Htemp(k+1,k),2),0.5);
        J(k+1,k+1) = Htemp(k,k) / pow(pow(Htemp(k,k),2) + pow(Htemp(k+1,k),2),0.5);

        Jtotal = J * Jtotal;

        HH = Jtotal * H;

        for (int i=1; i <= k+1; i++) {
            for (int j=1; j <= k; j++) {
                // set all 'small' values to zero
                if (fabs(HH(i,j)) < 1e-15) {
                    HH(i, j) = 0;
                }
            }
        }

        bb = resize(bb,k+1,1);

        c = Jtotal*bb;

        residual = fabs(c(k+1,1));

        k++;
    }
    // std::cout<< "GMRES iteration converged in " << k - 1 << " steps\n\n";

    HH = resize(HH,rows(HH) - 1, columns(HH)); //std::cout<< "HH: \n\n" << HH;
    
    cc = resize(c,rows(HH),1); //std::cout<< "cc: \n\n" << cc;
    
    matrix yy = cc/HH;
    
    vector y = mat2vec(yy);
//std::cout<< "y: \n\n" << y;
// chop the newest column off of matrix V
    V = resize(V,rows(V), columns(V) - 1);
    
    x = mat2vec(x0 + V * y);
    // return x;

    --k;
}