#ifndef VECTOR_DEF
#define VECTOR_DEF

#include "matrix.cuh"

namespace gpu_thrust
{

class vector;

vector operator+(const vector& A, const vector& B);
vector operator-(const vector& A, const vector& B);
vector operator*(const double& p, const vector& A);
vector operator*(const vector& A, const double& p);
vector operator/(const vector& A, const double& p);
vector operator+(const vector& A);
vector operator-(const vector& A);

template<typename T>
vector mat2vec(T&& A);

double norm(vector& v, int p=2);
void GMRES(matrix& A, matrix& b, matrix& x0, matrix& x, int& k, double tol=1e-6);

// 'vector' inherits from 'matrix' in a public way
class vector: public matrix
{
public:
//////////////////////// Constructors ////////////////////////////////////
    vector() = default; // ctor
    ~vector() = default; // dtor
    vector(const vector& v) = default; // copy ctor
    vector& operator=(const vector& v) = default; // copy assignment
    vector(int no_of_elements);
    vector(const thrust::device_vector<double>& _data_d, int _rows) : matrix(_data_d, _rows, 1) {}

////////////////// Binary Operators //////////////////////////////////////
    friend vector operator+(const vector& A, const vector& B);
    friend vector operator-(const vector& A, const vector& B);

    friend vector operator*(const double& p, const vector& A);
    friend vector operator*(const vector& A, const double& p);
    friend vector operator/(const vector& A, const double& p);

////////////////////// Unary operators //////////////////////////////////
    friend vector operator+(const vector& A);
    friend vector operator-(const vector& A);

/////////////////////// Other operators /////////////////////////////////
    // double& operator()(int i);
    double get(int i);
    void set(int i, double val);

////////////////////// Functions that are friends //////////////////////
    template<typename T>
    friend vector mat2vec(T&& A);

    friend double norm(vector& v, int);
    friend void GMRES(matrix& A, matrix& b, matrix& x0, matrix& x, int& k, double tol);
};

template<typename T>
vector mat2vec(T&& A)
{
    gpu_thrust::vector v(rows(A));
    for (int i = 1; i <= rows(A); i++){
        v.set(i, A.get(i, 1));
    }
    return v;
}

}
#endif