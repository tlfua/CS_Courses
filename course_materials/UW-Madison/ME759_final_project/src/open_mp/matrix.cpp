#include "matrix.h"

// sequential::matrix::matrix() : rows(0), columns(0)
// {}

sequential::matrix::matrix(int no_of_rows, int no_of_columns) : rows(no_of_rows), columns(no_of_columns)
{
    
    // A is an array of m pointers, each pointing to the
    // first entry in the vector (of length, 'no of columns')
    // xx = new double * [no_of_rows];

    // row major
    data = std::vector<double>(rows * columns, 0.0);

    for (int col_iter = 0; col_iter < columns; ++col_iter){
        increments.push_back(col_iter);
    }
}

///////////////////// Binary Operators ///////////////////////////////////

// Overload the + operator to evaluate: A + B, where A and B are matrices
sequential::matrix sequential::operator+(const matrix& A, const matrix& B)
{
    int m = rows(A), n = columns(A), p = rows(B), q = columns(B);

    if(m != p || n != q){
        std::cout << "Error: Matrices of different dimensions";
        std::cout << "Returned first argument";
        return A;
    } else {
        matrix C(m,n);
        std::transform(std::begin(A.data), std::end(A.data), std::begin(B.data), std::begin(C.data), std::plus<double>());
        return C;
    }
}

// Overload the - operator to evaluate: A - B, where A and B are matrices
sequential::matrix sequential::operator-(const matrix& A, const matrix& B)
{
    int m = rows(A), n = columns(A), p = rows(B), q = columns(B);

    if(m != p || n != q){
        std::cout << "Error: Matrices of different dimensions";
        std::cout << "Returned first argument";
        return A;
    } else {
        matrix C(m,n);
        std::transform(std::begin(A.data), std::end(A.data), std::begin(B.data), std::begin(C.data), std::minus<double>());
        return C;
    }
}

// Definition of multiplication between a scalar, p and a matrix, A
sequential::matrix sequential::operator*(const double& p, const matrix& A)
{
    // Create a matrix with the same dimensions as A
    matrix B(A.rows, A.columns);
    std::transform(std::begin(A.data), std::end(A.data), std::begin(B.data), [p](auto& elem){ return elem * p; });
    return B;
}

// Definition of multiplication between a matrix, A and a scalar, p
sequential::matrix sequential::operator*(const matrix& A, const double& p)
{
    // Create a matrix with the same dimensions as A
    matrix B(A.rows, A.columns);
    std::transform(std::begin(A.data), std::end(A.data), std::begin(B.data), [p](auto& elem){ return elem * p; });
    return B;
}

sequential::matrix sequential::operator*(const matrix& A, const matrix& B) {
    // Use assertion to check matrix dimensions are consistent
    assert(A.columns == B.rows);
    
    matrix C(A.rows, B.columns);
    #pragma omp parallel for
    for (int A_row_iter = 0; A_row_iter < A.rows; ++A_row_iter) {
        for (int B_col_iter = 0; B_col_iter < B.columns; ++B_col_iter) {

            // ToDo: to make it more efficient later
            double sum = 0;
            std::for_each(std::begin(A.increments), std::end(A.increments), [&A, &B, A_row_iter, B_col_iter, &sum](int k){
                sum += A.data[A_row_iter * A.columns + k] * B.data[B_col_iter + k * B.columns];
            });
            C.data[A_row_iter * B.columns + B_col_iter] = sum;
        }
    }
    return C;
}

// Definition of division of a matrix, A by a scalar, p i.e. A/p
sequential::matrix sequential::operator/(const matrix& A, const double& p)
{
    // Create a matrix with the same dimensions as A
    matrix B(A.rows, A.columns);
    std::transform(std::begin(A.data), std::end(A.data), std::begin(B.data), [p](auto& elem){ return elem / p; });
    return B;
}

///////////////////// Unary Operators ///////////////////////////////////
sequential::matrix sequential::operator+(const matrix& A)
{
    return sequential::matrix(A);
}

sequential::matrix sequential::operator-(const matrix& A)
{
    // Create a temporary matrix with the same dimensions as A
    matrix B(A.rows, A.columns);
    std::transform(std::begin(A.data), std::end(A.data), std::begin(B.data), std::negate<double>());
    return B;
}

// Overload the ~ operator to mean transpose
sequential::matrix sequential::operator~(const matrix& A)
{
    matrix B(A.columns, A.rows);

    // Set the entires of B to be the same as those in A
    for (int i = 0; i < B.rows; i++) {
        for (int j = 0; j < B.columns; j++) {
            B.data[i * B.columns + j] = A.data[j * A.columns + i];
        }
    }
    return B;
}

// Can call or assign values.
double& sequential::matrix::operator()(int i, int j)
{
    if(i < 1 || j < 1){
        std::cout << "Error: One of your indices may have been too small \n\n";
    } else if (i > rows || j > columns){
        std::cout << "Error: One of your indices may have been too large \n\n";
    }
    return data[(i - 1) * this->columns + (j - 1)];
}

// Returns the private field, 'rows'
int sequential::rows(const matrix& A)
{
    return A.rows;
}

// Returns the private field, 'columns'
int sequential::columns(const matrix& A)
{
    return A.columns;
}

std::ostream& sequential::operator<<(std::ostream& output, const matrix &A)
{
    for (int i=0; i < A.rows * A.columns; ++i){
        output << " " << A.data[i];
        if ((i + 1) % A.columns == 0) {
            output << "\n";
        }
    }
    output << "\n";

    return output;
}

sequential::matrix sequential::eye(int size)
{
    matrix temp_eye(size, size);
  
    for (int i = 0; i < size; ++i) {
        temp_eye.data[i * size + i] = 1; 
    }
    return temp_eye;
}

// Function that returns an nxn permutation matrix which swaps rows i and j
sequential::matrix sequential::permute_r(int n, int i, int j)
{
    matrix I = eye(n);
    
    // Zero the diagonal entries in the given rows
    I(i, i) = 0;
    I(j, j) = 0;
    
    // Set the appropriate values to be 1
    I(i, j) = 1;
    I(j, i) = 1;
    
    return I;
}

// Function that returns the row number of the largest
// sub−diagonal value of a given column
int sequential::find_pivot(matrix& A, int column)
{
    // Initialise maxval to be diagonal entry in column, 'column'
    double maxval = fabs(A(column, column));
    
    // Initialise rowval to be column
    int rowval=column;

    for(int i=column+1; i <= A.rows; i++){
        if (fabs(A(i, column)) > maxval) {
            // Update maxval and rowval if bigger than previous maxval
            maxval = fabs(A(i, column));
            rowval = i;
        }
    }
    return rowval;
}

// Function that returns an mxn matrix with entries that
// are the same as matrix A, where possible
sequential::matrix sequential::resize(matrix& A, int m, int n) {
    int p, q;
    matrix Mout(m, n);

    // select lowest of each matrix dimension
    if (m <= A.rows) {
        p = m;
    } else {
        p = A.rows;
    }

    if(n <= A.columns) {
        q = n;
    } else {
        q = A.columns;
    }
  
    // ToDo: try std::copy
    // copy across relevant values
    for (int i=1; i <= p; i++) {
        for (int j=1; j <= q; j++) {
            Mout(i,j) = A(i,j);
        }
    }
    return Mout;
}

// baskslash
// Definition of division of a vector, b by a matrix, A i.e. y=b/A
sequential::matrix sequential::operator/(const matrix& b, const matrix& A)
{
    int n = A.rows;
    
    // Create empty matrices, P &
    matrix P, L;
    
    // Create and intialise U &
    matrix Utemp = eye(n);
    matrix Atemp=A;
    //std::cout << U << "\n\n";

    for (int j = 1; j < n; j++){
        // Create appropriate permutation matrix, P
        P = permute_r(n, find_pivot(Atemp, j), j);
        Utemp = P * Utemp;
        Atemp = Utemp * A;
        L = eye(n);
        for (int i = j + 1; i <= n; i++){
            // Check for division by zero
            assert(fabs(Atemp(j,j)) > 1.0e-015);
            // Compute multiplier and store in sub−diagonal entry of L
            L(i,j)= -Atemp(i,j) / Atemp(j,j);
        }
        Utemp = L * Utemp;
        Atemp = Utemp * A;
    }

    // Now loop through and set to zero any values which are almost zero
    for (int j=1; j < n ; j++) {
        for (int i=j+1; i <= n ; i++) {
            if (fabs(Atemp(i,j)) < 5.0e-016){
                Atemp(i,j) = 0;
            }
        }
    }

    // So, to solve Ax=b, we do: (Utemp*A)x=Utemp*b i.e.
    // Set U=Utemp*A=Atemp, compute
    // solve Ux=y (upper triangular
    matrix U = Utemp * A;
    matrix y = Utemp * b;

    // Create result vector
    matrix x(n,1);

    // Compute last entry of vector x (first step in back subs)
    x(n,1)=y(n,1) / U(n,n);

    double temp = 0;
    for (int i = n - 1; i >= 1; i--){
        temp = y(i, 1);
        for (int j = n; j > i; j--){
            temp = temp - U(i, j) * x(j, 1);
        }
        x(i, 1) = temp / U(i, i);
    }
    return x;
}