#include <iostream>
#include <cmath>
#include <cstring>
#include <cblas.h>
#include <random>
#include <chrono>
#include <vector>
# include <omp.h>


using namespace std;

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution <>floatDist(-1, 0);


void random_init(double *A, int n) {
    #pragma omp parallel for 
    for(int i = 0; i < n; i++) {
        A[i] = (-floatDist(gen));
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


double ddot(double *a, double *b, int n) {
    double sum = 0;

    #pragma omp parallel for reduction(+:sum)
    for (int i = 0 ; i < n ; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}


void dgemv(double *A, double *p, double *tmp, int n, double alpha = 1) {

    #pragma omp parallel for 
    for (int i = 0; i < n; i++) {
        tmp[i] = 0;
    }

    for (int i = 0; i < n; i++) {
        #pragma omp parallel for 
        for (int j = 0; j < n; j++) {
           tmp[j] += alpha * A[i * n + j] * p[i]; 
        }
    }
}


void daxpy(double *x, double *y, int n, double alpha = 1) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        y[i] += x[i] * alpha;
    }
}


void SG(double *A, double *b, double *x, double *p, double *r, int n, double eps) {
    double r0 = pow(ddot(r, r, n), 0.5);
    if (r0 < eps) {
        return;
    }

    #pragma omp parallel for 
    for (int i = 0; i < n; i++) {
        p[i] = r[i];
    }

    double *tmp = new double [n];
    for (int i = 1; i <= n; i++) {
        if (i % 10 == 0) {
            dgemv(A, x, r, n, -1);
            daxpy(b, r, n);
        }

        double alpha = ddot(r, r, n);
        double beta = 1 / alpha;
        dgemv(A, p, tmp, n);
        alpha /= ddot(tmp, p, n);
        daxpy(tmp, r, n, -alpha);
        daxpy(p, x, n, alpha);
        
        if (pow(ddot(r, r, n), 0.5) / r0  < eps) {
            cout << i << " iterations" << endl;
            return;
        }
        beta *= ddot(r, r, n);
        
        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            p[i] *= beta;
        }
        daxpy(r, p, n);
    }
    cout << endl << "Full iterations" << endl;
}


int main(int argc, char *argv[]) {
    int n = std::stoi(argv[1]);
    srand(time(NULL));

    double eps = 1e-8;
    double *A, *b, *x_sol, *x, *p, *r;
    A = new double [n * n];
    b = new double [n];
    x_sol = new double[n];
    x = new double [n];
    p = new double [n];
    r = new double [n];


    for (int i = 0 ; i < n ; i++) {
        double sum = 0;
        #pragma omp parallel for reduction(+:sum)
        for (int j = 0; j < n; j++) {
            if (j > i) {
                A[i * n + j] = (-floatDist(gen));
                A[j * n + i] = A[i * n + j];
            }
            if (i != j) {
                sum += A[i * n + j];
            }
        }
        A[i * n + i] = sum + 1;
    }

    random_init(x_sol, n);
    random_init(x, n);

    dgemv(A, x_sol, b, n);
    dgemv(A, x, r, n, -1);
    daxpy(b, r, n);


    double r0 = pow(ddot(r, r, n), 0.5);
    SG(A, b, x, p, r, n, eps);

    cout << pow(ddot(r, r, n), 0.5) / r0 << endl;

}
