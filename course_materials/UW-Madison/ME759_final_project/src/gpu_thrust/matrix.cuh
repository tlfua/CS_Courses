#ifndef MATRIX_DEF
#define MATRIX_DEF

#include <iostream>
#include <cmath>
#include <iterator>
#include <functional> // std::plus 
#include <algorithm> // std::transform
// #include <execution>
#include <string>
#include <vector>
#include <cassert>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/functional.h>
// #include <thrust/sequence.h>
// #include <thrust/fill.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/inner_product.h>
#include <thrust/execution_policy.h>

namespace gpu_thrust
{

class matrix;

// friend function pre-definition
matrix operator+(const matrix& A, const matrix& B);
matrix operator-(const matrix& A, const matrix& B);
matrix operator*(const double& p, const matrix& A);
matrix operator*(const matrix& A, const double& p);
matrix operator*(const matrix& A, const matrix& B);
matrix operator/(const matrix& A, const double& p);
matrix operator/(const matrix& b, const matrix& A);
matrix operator+(const matrix& A);
matrix operator-(const matrix& A);
matrix operator~(const matrix& A);

// vector mat2vec(matrix A);

int rows(const matrix& A);
int columns(const matrix& A);

std::ostream& operator<<(std::ostream& output, const matrix& A);

matrix eye(int size);
matrix permute_r(int n, int i, int j);
int find_pivot(matrix& A, int column);
matrix resize(matrix& A, int m, int n);

struct multiply
{
    double m;
    multiply(double _m) : m(_m) {}

    __host__ __device__
    double operator()(double x)
    {
        return m * x;
    }
};

struct divide
{
    double d;
    divide(double _d) : d(_d) {}

    __host__ __device__
    double operator()(double x)
    {
        return x / d;
    }
};

template<typename T>
struct power : public thrust::unary_function<T,T>
{
    int p;
    power(int _p) : p(_p) {}

    __host__ __device__
    T operator()(const T &x) const
    {
        return pow(x, p);
    }
};

class matrix
{
protected:
    int rows, columns;
    thrust::host_vector<double> data;
    // std::vector<int> increments;

public:
/////////////////// Constructors //////////////////////////////////////////
    matrix() = default; // ctor
    ~matrix() = default; // dtor
    matrix(const matrix& A) = default; // copy ctor
    matrix& operator=(const matrix &v) = default; // copy assignment
    matrix(int no_of_rows, int no_of_columns);
    
    matrix(const thrust::device_vector<double>& _data_d, int _rows, int _cols) : data(_data_d), rows(_rows), columns(_cols) {}

    /*
    void set_data(const thrust::device_vector<double>& _data_d)
    {
        thrust::copy(_data_d.begin(), _data_d.end(), this->data.begin());
    } 
    */
    
////////////////// Binary Operators ///////////////////////////////////////
    friend matrix operator+(const matrix& A, const matrix& B);
    friend matrix operator-(const matrix& A, const matrix& B);
    
    friend matrix operator*(const double& p, const matrix& A);
    friend matrix operator*(const matrix& A, const double& p);
    friend matrix operator*(const matrix& A, const matrix& B);
    
    friend matrix operator/(const matrix& A, const double& p);
    friend matrix operator/(const matrix& b, const matrix& A);

////////////////////// Unary operators ////////////////////////////////////
    friend matrix operator+(const matrix& A);
    friend matrix operator-(const matrix& A);

    // Overload ~ to mean transpose
    friend matrix operator~(const matrix& A);

/////////////////////// Other operators ///////////////////////////////////
    // Overloads (), so A(i,j) returns the i,j entry a la MATLAB
    // double &operator()(int i, int j);
    double get(int i, int j);
    void set(int i, int j, double val);
    
    thrust::host_vector<double> get_data() const
    {
        return this->data;
    }
    
////////////////////// Other friend functions /////////////////////////
    friend int rows(const matrix& A); // Returns the row dimension
    friend int columns(const matrix& A); // Returns the column dimension

    friend std::ostream& operator<<(std::ostream& output, const matrix& A);

    friend matrix eye(int size);
    friend matrix permute_r(int n, int i, int j);
    friend int find_pivot(matrix& A, int column);
    friend matrix resize(matrix& A, int m, int n);
};

}
#endif
