#include "vector.cuh"

gpu_thrust::vector::vector(int no_of_elements)
        : matrix(no_of_elements, 1)
{}

gpu_thrust::vector gpu_thrust::operator+(const vector& A, const vector& B)
{
    int row_A = rows(A);
    int row_B = rows(B);

    if(row_A != row_B){
        std::cout << "Error: Matrices of different dimensions";
        std::cout << "Returned first argument";
        return A;
    } else {
        /*
        vector C(row_A);
        thrust::transform(A.data.begin(), A.data.end(), B.data.begin(), C.data.begin(), thrust::plus<double>());
        return C;
        */
        thrust::device_vector<double> A_data_d(A.get_data());
        thrust::device_vector<double> B_data_d(B.get_data());
        thrust::device_vector<double> C_data_d(row_A);
        thrust::transform(thrust::device,\
                            A_data_d.begin(), A_data_d.end(),\
                            B_data_d.begin(),\
                            C_data_d.begin(),\
                            thrust::plus<double>());
        return vector(C_data_d, row_A);
    }
}

gpu_thrust::vector gpu_thrust::operator-(const vector& A, const vector& B)
{
    int row_A = rows(A);
    int row_B = rows(B);

    if(row_A != row_B){
        std::cout << "Error: Matrices of different dimensions";
        std::cout << "Returned first argument";
        return A;
    } else {
        /*
        vector C(row_A);
        thrust::transform(A.data.begin(), A.data.end(), B.data.begin(), C.data.begin(), thrust::minus<double>());
        return C;
        */
        thrust::device_vector<double> A_data_d(A.get_data());
        thrust::device_vector<double> B_data_d(B.get_data());
        thrust::device_vector<double> C_data_d(row_A);
        thrust::transform(thrust::device,\
                            A_data_d.begin(), A_data_d.end(),\
                            B_data_d.begin(),\
                            C_data_d.begin(),\
                            thrust::minus<double>());
        return vector(C_data_d, row_A);
    }
}

gpu_thrust::vector gpu_thrust::operator*(const double& p, const vector& A)
{   
    /*
    vector B(A.rows);
    multiply mul = multiply(p);
    thrust::transform(A.data.begin(), A.data.end(), B.data.begin(), mul);
    return B;
    */
    thrust::device_vector<double> A_data_d(A.get_data());
    thrust::device_vector<double> B_data_d(A.rows);
    multiply mul = multiply(p);
    thrust::transform(thrust::device,\
                        A_data_d.begin(), A_data_d.end(),\
                        B_data_d.begin(),\
                        mul);
    return vector(B_data_d, A.rows);
}

gpu_thrust::vector gpu_thrust::operator*(const vector& A, const double& p)
{
    /*
    vector B(A.rows);
    multiply mul = multiply(p);
    thrust::transform(A.data.begin(), A.data.end(), B.data.begin(), mul);
    return B;
    */
    thrust::device_vector<double> A_data_d(A.get_data());
    thrust::device_vector<double> B_data_d(A.rows);
    multiply mul = multiply(p);
    thrust::transform(thrust::device,\
                        A_data_d.begin(), A_data_d.end(),\
                        B_data_d.begin(),\
                        mul);
    return vector(B_data_d, A.rows);
}

gpu_thrust::vector gpu_thrust::operator/(const vector& A, const double& p)
{
    /*
    vector B(A.rows);
    divide div = divide(p);
    thrust::transform(A.data.begin(), A.data.end(), B.data.begin(), div);
    return B;
    */
    thrust::device_vector<double> A_data_d(A.get_data());
    thrust::device_vector<double> B_data_d(A.rows);
    divide div = divide(p);
    thrust::transform(thrust::device,\
                        A_data_d.begin(), A_data_d.end(),\
                        B_data_d.begin(),\
                        div);
    return vector(B_data_d, A.rows);
}

gpu_thrust::vector gpu_thrust::operator+(const vector& A)
{
    return gpu_thrust::vector(A);
}

gpu_thrust::vector gpu_thrust::operator-(const vector& A)
{
    /*
    vector B(A.rows);
    thrust::transform(A.data.begin(), A.data.end(), B.data.begin(), thrust::negate<double>());
    return B;   
    */
    thrust::device_vector<double> A_data_d(A.get_data());
    thrust::device_vector<double> B_data_d(A.rows);
    thrust::transform(thrust::device,\
                        A_data_d.begin(), A_data_d.end(),\
                        B_data_d.begin(),\
                        thrust::negate<double>());
    return vector(B_data_d, A.rows);
}

double gpu_thrust::vector::get(int i)
{
    if (i < 1) {
        std::cout << "Error: Your index may be too small \n\n";
    } else if (i > rows) {
        std::cout << "Error: Your index may be too large \n\n";
    }
    return data[i - 1];
}

void gpu_thrust::vector::set(int i, double val)
{
    if (i < 1) {
        std::cout << "Error: Your index may be too small \n\n";
    } else if (i > rows) {
        std::cout << "Error: Your index may be too large \n\n";
    }
    data[i - 1] = val;
}

// sequential::vector sequential::mat2vec(sequential::matrix& A)
// {
//     sequential::vector v(rows(A));
//     for (int i = 1; i <= rows(A); i++){
//         v(i) = A(i, 1);
//     }
//     return v;
// }

double gpu_thrust::norm(gpu_thrust::vector& v, int p)
{
    /*
    power<double> my_pow = power<double>(p);
    double sum = thrust::transform_reduce(\
                    v.data.begin(), v.data.end(),\
                    my_pow,
                    0,
                    thrust::plus<double>());
    return pow(sum, 1.0 / ((double)(p)));
    */
    
    double res, sum = 0.0;
    for (int i=1; i <= rows(v); i++) {
        // floating point absolute value
        sum += pow(fabs(v.get(i)), p);
    }
    res = pow(sum, 1.0 / ((double)(p)));
    return res;
}

void gpu_thrust::GMRES(matrix& A, matrix& b, matrix& x0, matrix& x, int& k, double tol)
{
    vector r0 = mat2vec(b - A * x0);

    double normr0 = norm(r0);

    double residual = 1.0;

    vector v = r0 / normr0;

    // int k = 1;

    matrix J, Jtotal;
    Jtotal = eye(2);

    matrix H(1, 1), Htemp, HH, bb(1, 1), c, cc;
    matrix tempMat, V, Vold, hNewCol;
    vector w, vj(rows(v));

    bb.set(1, 1, normr0);

    V = v;
    k = 1;
    while (residual > tol){

        Vold = V;
      
        H = resize(H, k + 1, k);

        w = mat2vec(A * v);

        for (int j = 1; j <= k; j++) {
            for (int i = 1; i <= rows(V); i++) {
                vj.set(i, V.get(i,j));
            }
            tempMat = ~vj * w;

            H.set(j, k, tempMat.get(1,1));

            w = w - H.get(j, k) * vj;
        }

        H.set(k + 1, k, norm(w));

        v = w / H.get(k + 1, k);

        V = resize(V, rows(V), k + 1);

        for (int i=1; i <= rows(V); i++) {
            // copy entries of v to new column of V
            V.set(i, k + 1, v.get(i));
        }

        if (k == 1) {
            // First pass through, Htemp=H
            Htemp = H;
        } else {
            // for subsequent passes, Htemp=Jtotal*H
            Jtotal = resize(Jtotal, k + 1, k + 1);
            Jtotal.set(k + 1, k + 1, 1);
            Htemp = Jtotal * H;
        }

        // Form next Givens rotation matrix
        J = eye(k - 1);
        J = resize(J,k+1,k+1);

        J.set(k, k, Htemp.get(k, k) / pow(pow(Htemp.get(k, k), 2) + pow(Htemp.get(k + 1, k), 2), 0.5));
        J.set(k, k + 1, Htemp.get(k + 1, k) / pow(pow(Htemp.get(k, k), 2) + pow(Htemp.get(k + 1, k), 2), 0.5));
        J.set(k + 1, k, -Htemp.get(k + 1, k) / pow(pow(Htemp.get(k, k), 2) + pow(Htemp.get(k + 1, k), 2), 0.5));
        J.set(k + 1, k + 1, Htemp.get(k, k) / pow(pow(Htemp.get(k, k), 2) + pow(Htemp.get(k + 1, k), 2), 0.5));

        Jtotal = J * Jtotal;

        HH = Jtotal * H;

        for (int i = 1; i <= k + 1; i++) {
            for (int j = 1; j <= k; j++) {
                // set all 'small' values to zero
                if (fabs(HH.get(i, j)) < 1e-15) {
                    HH.set(i, j, 0);
                }
            }
        }

        bb = resize(bb, k + 1, 1);

        c = Jtotal * bb;

        residual = fabs(c.get(k + 1, 1));

        k++;
    }
    // std::cout<< "GMRES iteration converged in " << k - 1 << " steps\n\n";

    HH = resize(HH, rows(HH) - 1, columns(HH)); //std::cout<< "HH: \n\n" << HH;
    
    cc = resize(c, rows(HH), 1); //std::cout<< "cc: \n\n" << cc;
    
    matrix yy = cc / HH;
    
    vector y = mat2vec(yy);
//std::cout<< "y: \n\n" << y;
// chop the newest column off of matrix V
    V = resize(V, rows(V), columns(V) - 1);
    
    x = mat2vec(x0 + V * y);
    // return x;

    --k;
}