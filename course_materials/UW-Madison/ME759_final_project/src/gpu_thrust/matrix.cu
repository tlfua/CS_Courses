#include "matrix.cuh"

gpu_thrust::matrix::matrix(int no_of_rows, int no_of_columns) : rows(no_of_rows), columns(no_of_columns)
{
    // row major
    data = thrust::device_vector<double>(rows * columns, 0.0);
}

///////////////////// Binary Operators ///////////////////////////////////
// Overload the + operator to evaluate: A + B, where A and B are matrices
gpu_thrust::matrix gpu_thrust::operator+(const matrix& A, const matrix& B)
{
    int m = rows(A), n = columns(A), p = rows(B), q = columns(B);

    if(m != p || n != q){
        std::cout << "Error: Matrices of different dimensions";
        std::cout << "Returned first argument";
        return A;
    } else {
        /*
        matrix C(m,n);
        thrust::transform(A.data.begin(), A.data.end(), B.data.begin(), C.data.begin(), thrust::plus<double>());
        return C;
        */
        thrust::device_vector<double> A_data_d(A.get_data());
        thrust::device_vector<double> B_data_d(B.get_data());
        thrust::device_vector<double> C_data_d(m * n);
        thrust::transform(thrust::device,\
                            A_data_d.begin(), A_data_d.end(),\
                            B_data_d.begin(),\
                            C_data_d.begin(),\
                            thrust::plus<double>());
        return matrix(C_data_d, m, n);
    }
}

// Overload the - operator to evaluate: A - B, where A and B are matrices
gpu_thrust::matrix gpu_thrust::operator-(const matrix& A, const matrix& B)
{
    int m = rows(A), n = columns(A), p = rows(B), q = columns(B);

    if(m != p || n != q){
        std::cout << "Error: Matrices of different dimensions";
        std::cout << "Returned first argument";
        return A;
    } else {
        /*
        matrix C(m,n);
        thrust::transform(A.data.begin(), A.data.end(), B.data.begin(), C.data.begin(), thrust::minus<double>());
        return C;
        */
        thrust::device_vector<double> A_data_d(A.get_data());
        thrust::device_vector<double> B_data_d(B.get_data());
        thrust::device_vector<double> C_data_d(m * n);
        thrust::transform(thrust::device,\
                            A_data_d.begin(), A_data_d.end(),\
                            B_data_d.begin(),\
                            C_data_d.begin(),\
                            thrust::minus<double>());
        return matrix(C_data_d, m, n);
    }
}

// Definition of multiplication between a scalar, p and a matrix, A
gpu_thrust::matrix gpu_thrust::operator*(const double& p, const matrix& A)
{
    /*
    matrix B(A.rows, A.columns);
    multiply mul = multiply(p);
    thrust::transform(A.data.begin(), A.data.end(), B.data.begin(), mul);
    return B;
    */
    thrust::device_vector<double> A_data_d(A.get_data());
    thrust::device_vector<double> B_data_d(A.rows * A.columns);
    multiply mul = multiply(p);
    thrust::transform(thrust::device,\
                        A_data_d.begin(), A_data_d.end(),\
                        B_data_d.begin(),\
                        mul);
    return matrix(B_data_d, A.rows, A.columns);
}

// Definition of multiplication between a matrix, A and a scalar, p
gpu_thrust::matrix gpu_thrust::operator*(const matrix& A, const double& p)
{
    /*
    matrix B(A.rows, A.columns);
    multiply mul = multiply(p);
    thrust::transform(A.data.begin(), A.data.end(), B.data.begin(), mul);
    return B;
    */
    thrust::device_vector<double> A_data_d(A.get_data());
    thrust::device_vector<double> B_data_d(A.rows * A.columns);
    multiply mul = multiply(p);
    thrust::transform(thrust::device,\
                        A_data_d.begin(), A_data_d.end(),\
                        B_data_d.begin(),\
                        mul);
    return matrix(B_data_d, A.rows, A.columns);
}

gpu_thrust::matrix gpu_thrust::operator*(const matrix& A, const matrix& B)
{
    // Use assertion to check matrix dimensions are consistent
    assert(A.columns == B.rows);
    /*
    matrix C(A.rows, B.columns);
    matrix B_T = ~B;

    thrust::device_vector<double> A_data_d(A.get_data());
    thrust::device_vector<double> B_T_data_d(B_T.get_data());

    for (int A_row_iter = 0; A_row_iter < A.rows; ++A_row_iter) {
        for (int B_col_iter = 0; B_col_iter < B.columns; ++B_col_iter) {
            C.data[A_row_iter * B.columns + B_col_iter] =\
                thrust::inner_product(thrust::device,\
                    A_data_d.begin() + A_row_iter * A.columns, A_data_d.begin() + (A_row_iter + 1) * A.columns,\
                    B_T_data_d.begin() + B_col_iter * A.columns,\
                    0.0);
        }
    }
    return C;
    */
    matrix C(A.rows, B.columns);
    for (int A_row_iter = 0; A_row_iter < A.rows; ++A_row_iter) {
        for (int B_col_iter = 0; B_col_iter < B.columns; ++B_col_iter) {

            // ToDo: to make it more efficient later
            double sum = 0;
            for (int k = 0; k < A.columns; ++k) {
                sum += A.data[A_row_iter * A.columns + k] * B.data[B_col_iter + k * B.columns];
            }
            C.data[A_row_iter * B.columns + B_col_iter] = sum;
        }
    }
    return C;
}

// Definition of division of a matrix, A by a scalar, p i.e. A/p
gpu_thrust::matrix gpu_thrust::operator/(const matrix& A, const double& p)
{
    /*
    matrix B(A.rows, A.columns);
    divide div = divide(p);
    thrust::transform(A.data.begin(), A.data.end(), B.data.begin(), div);
    return B;
    */
    thrust::device_vector<double> A_data_d(A.get_data());
    thrust::device_vector<double> B_data_d(A.rows * A.columns);
    divide div = divide(p);
    thrust::transform(thrust::device,\
                        A_data_d.begin(), A_data_d.end(),\
                        B_data_d.begin(),\
                        div);
    return matrix(B_data_d, A.rows, A.columns);
}

///////////////////// Unary Operators ///////////////////////////////////
gpu_thrust::matrix gpu_thrust::operator+(const matrix& A)
{
    return gpu_thrust::matrix(A);
}

gpu_thrust::matrix gpu_thrust::operator-(const matrix& A)
{
    /*
    matrix B(A.rows, A.columns);
    thrust::transform(A.data.begin(), A.data.end(), B.data.begin(), thrust::negate<double>());
    return B;
    */
    thrust::device_vector<double> A_data_d(A.get_data());
    thrust::device_vector<double> B_data_d(A.rows * A.columns);
    thrust::transform(thrust::device,\
                        A_data_d.begin(), A_data_d.end(),\
                        B_data_d.begin(),\
                        thrust::negate<double>());
    return matrix(B_data_d, A.rows, A.columns);
}

// Overload the ~ operator to mean transpose
gpu_thrust::matrix gpu_thrust::operator~(const matrix& A)
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

// get value
double gpu_thrust::matrix::get(int i, int j)
{
    if(i < 1 || j < 1){
        std::cout << "Error: One of your indices may have been too small \n\n";
    } else if (i > rows || j > columns){
        std::cout << "Error: One of your indices may have been too large \n\n";
    }
    return data[(i - 1) * this->columns + (j - 1)];
}

// set value
void gpu_thrust::matrix::set(int i, int j, double val)
{
    if(i < 1 || j < 1){
        std::cout << "Error: One of your indices may have been too small \n\n";
    } else if (i > rows || j > columns){
        std::cout << "Error: One of your indices may have been too large \n\n";
    }
    data[(i - 1) * this->columns + (j - 1)] = val;
}

// Returns the private field, 'rows'
int gpu_thrust::rows(const matrix& A)
{
    return A.rows;
}

// Returns the private field, 'columns'
int gpu_thrust::columns(const matrix& A)
{
    return A.columns;
}

std::ostream& gpu_thrust::operator<<(std::ostream& output, const matrix &A)
{
    std::vector<double> tmp_v(A.data.size());
    thrust::copy(A.data.begin(), A.data.end(), tmp_v.begin());
    for (int i=0; i < A.rows * A.columns; ++i){
        output << " " << tmp_v[i];
        if ((i + 1) % A.columns == 0) {
            output << "\n";
        }
    }
    output << "\n";

    return output;
}

gpu_thrust::matrix gpu_thrust::eye(int size)
{
    // matrix temp_eye(size, size);
    matrix I(size, size);
    for (int i = 0; i < size; ++i) {
        I.data[i * size + i] = 1;
    }
    return I;
}

// Function that returns an nxn permutation matrix which swaps rows i and j
gpu_thrust::matrix gpu_thrust::permute_r(int n, int i, int j)
{
    matrix I = eye(n);
    
    // Zero the diagonal entries in the given rows
    I.set(i, i, 0);
    I.set(j, j, 0);
    
    // Set the appropriate values to be 1
    I.set(i, j, 1);
    I.set(j, i, 1);
    
    return I;
}

// Function that returns the row number of the largest
// sub−diagonal value of a given column
int gpu_thrust::find_pivot(matrix& A, int column)
{
    // Initialise maxval to be diagonal entry in column, 'column'
    double maxval = fabs(A.get(column, column));
    
    // Initialise rowval to be column
    int rowval = column;

    for(int i = column + 1; i <= A.rows; i++){
        if (fabs(A.get(i, column)) > maxval) {
            // Update maxval and rowval if bigger than previous maxval
            maxval = fabs(A.get(i, column));
            rowval = i;
        }
    }
    return rowval;
}

// Function that returns an mxn matrix with entries that
// are the same as matrix A, where possible
gpu_thrust::matrix gpu_thrust::resize(matrix& A, int m, int n) {
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
            Mout.set(i, j, A.get(i,j));
        }
    }
    return Mout;
}

// baskslash
// Definition of division of a vector, b by a matrix, A i.e. y=b/A
gpu_thrust::matrix gpu_thrust::operator/(const matrix& b, const matrix& A)
{
    int n = A.rows;
    
    // Create empty matrices, P &
    matrix P, L;
    
    // Create and intialise U &
    matrix Utemp = eye(n);
    matrix Atemp = A;
    //std::cout << U << "\n\n";

    for (int j = 1; j < n; j++){
        // Create appropriate permutation matrix, P
        P = permute_r(n, find_pivot(Atemp, j), j);
        Utemp = P * Utemp;
        Atemp = Utemp * A;
        L = eye(n);
        for (int i = j + 1; i <= n; i++){
            // Check for division by zero
            assert(fabs(Atemp.get(j, j)) > 1.0e-015);
            // Compute multiplier and store in sub−diagonal entry of L
            L.set(i, j, -Atemp.get(i, j) / Atemp.get(j, j));
        }
        Utemp = L * Utemp;
        Atemp = Utemp * A;
    }

    // Now loop through and set to zero any values which are almost zero
    for (int j = 1; j < n ; j++) {
        for (int i = j + 1; i <= n ; i++) {
            if (fabs(Atemp.get(i, j)) < 5.0e-016){
                Atemp.set(i, j, 0);
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
    x.set(n, 1, y.get(n, 1) / U.get(n, n));

    double temp = 0;
    for (int i = n - 1; i >= 1; i--){
        temp = y.get(i, 1);
        for (int j = n; j > i; j--){
            temp = temp - U.get(i, j) * x.get(j, 1);
        }
        x.set(i, 1, temp / U.get(i, i));
    }
    return x;
}
