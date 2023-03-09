#include <iostream>
#include <cmath>
#include <cstring>
#include <cblas.h>
#include <chrono>
#include <vector>

using namespace std;


void random_init(double *A, int n) {
    for(int i = 0; i < n; i++) {
        A[i] = double(int(rand()) % 10 + 1) / 100;
    }
}


void print_matrix(double *A, int n) {
    cout << "[";
    for(int i = 0 ; i < n; i++) {
        cout << "[";
        for (int j = 0 ; j < n; j++) {
            cout << A[j * n + i] << ", ";
        }
        cout << "],";
        cout << endl;
    }
    cout << "]";
    cout << endl;
}


void SG(double *A, double *b, double *x, double *p, double *r, int n, double eps) {
    if (cblas_dnrm2(n, r, 1) < eps) {
        return;
    }
    cblas_dcopy(n, r, 1, p, 1);

    double *tmp = new double [n];
    for (int i = 1; i <= n; i++) {
        if (i % 5 == 0) {
            cblas_dgemv(CblasColMajor, CblasNoTrans, n, n, -1.0, A, n, x, 1, 0.0, r, 1);
            cblas_daxpy(n, 1.0, b, 1, r, 1);
        }

        double alpha = cblas_ddot(n, r, 1, r, 1);
        double beta = 1 / alpha;
        cblas_dgemv(CblasColMajor, CblasNoTrans, n, n, 1.0, A, n, p, 1, 1.0, tmp, 1);
        alpha /= cblas_ddot(n, p, 1, tmp, 1);
        cblas_dgemv(CblasColMajor, CblasNoTrans, n, n, -alpha, A, n, p, 1, 1.0, r, 1);
        cblas_daxpy(n, alpha, p, 1, x, 1);
        
        if (cblas_dnrm2(n, r, 1) < eps) {
            return;
        }
        beta *= cblas_ddot(n, r, 1, r, 1);
        cblas_dscal(n, beta, p, 1);
        cblas_daxpy(n, 1.0, r, 1, p, 1);
    }
    cout << endl << "Full iterations" << endl;
}


int main() {
    int n;
    cin >> n;
    srand(time(NULL));

    double eps = 1e-6;
    double *A, *L, *b, *x_sol, *x, *p, *r;
    L = new double [n * n];
    A = new double [n * n];
    b = new double [n];
    x_sol = new double[n];
    x = new double [n];
    p = new double [n];
    r = new double [n];


    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (j >= i) {
                L[j + i * n] = double(int(rand()) % 10 + 1) / 100;
            }
        }
    }
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, n, n, n, 1.0, L, n, L, n, 1.0, A, n); 
    random_init(x_sol, n);
    random_init(x, n);

    cblas_dgemv(CblasColMajor, CblasNoTrans, n, n, 1.0, A, n, x_sol, 1, 1.0, b, 1);
    cblas_dgemv(CblasColMajor, CblasNoTrans, n, n, -1.0, A, n, x, 1, 1.0, r, 1);
    cblas_daxpy(n, 1.0, b, 1, r, 1);

    SG(A, b, x, p, r, n, eps);
    cout << cblas_dnrm2(n, r, 1) << endl; 



    /* 
    print_matrix(A, n);

    for (int i = 0; i < n; i++) {
        cout << x_sol[i] << " " << x[i] << endl; 
    }
    cout << endl;

    for (int i = 0; i < n; i++) {
        cout << b[i] << " ";
    }
    */
}
