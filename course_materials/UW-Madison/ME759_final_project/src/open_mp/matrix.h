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
#include <omp.h>

namespace sequential
{

class matrix;
// class vector;

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

class matrix
{
protected:
    int rows, columns;
    std::vector<double> data;
    std::vector<int> increments;
public:
/////////////////// Constructors/ Destructor //////////////////////////////
    matrix() = default; // ctor
    ~matrix() = default; // dtor
    matrix(const matrix& A) = default; // copy ctor
    matrix& operator=(const matrix &v) = default; // copy assignment
    matrix(int no_of_rows, int no_of_columns);
////////////////// Binary Operators ///////////////////////////////////////
    friend matrix operator+(const matrix& A, const matrix& B);
    friend matrix operator-(const matrix& A, const matrix& B);
    friend matrix operator*(const double& p, const matrix& A);
    friend matrix operator*(const matrix& A, const double& p);
    friend matrix operator*(const matrix& A, const matrix& B);
    friend matrix operator/(const matrix& A, const double& p);
    friend matrix operator/(const matrix& b, const matrix& A);
////////////////////// Unary Operators ////////////////////////////////////
    friend matrix operator+(const matrix& A);
    friend matrix operator-(const matrix& A);
    friend matrix operator~(const matrix& A);  // ~ means transpose
/////////////////////// Access Matrix Element /////////////////////////////
    double &operator()(int i, int j);
/////////////////////// Other friend functions /////////////////////////
    friend int rows(const matrix& A); 
    friend int columns(const matrix& A); 
    friend std::ostream& operator<<(std::ostream& output, const matrix& A);
    friend matrix eye(int size);
    friend matrix permute_r(int n, int i, int j);
    friend int find_pivot(matrix& A, int column);
    friend matrix resize(matrix& A, int m, int n);
};

}
#endif
