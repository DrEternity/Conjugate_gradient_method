#include <iostream>
#include <cmath>
#include <cstring>
#include <random>
#include <chrono>
#include <vector>
#include <omp.h>
#include <chrono>


using namespace std;

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution <>floatDist(-1, 0);
int num_threads = 6;


void random_init(double *A, int n) {
    for(int i = 0; i < n; i++) {
        A[i] = (-floatDist(gen));
    }
}


template <class T>
double ddot(T *a, T *b, int n) {
    double sum = 0;
    for (int i = 0 ; i < n ; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}


template <class T>
void daxpy(T *x, T *y, int n, double alpha = 1) {
    for (int i = 0; i < n; i++) {
        y[i] += x[i] * alpha;
    }
}


template <class T, class Matrix>
void SG(Matrix &A, T *b, T *x, double eps) {
    int n = A.order();
    T *r, *p;
    r = new T [n];
    p = new T [n];

    A.matvec(x, r, -1);
    daxpy(b, r, n);
   
    double r0 = pow(ddot(r, r, n), 0.5);

    for (int i = 0; i < n; i++) {
        p[i] = r[i];
    }
    T *tmp = new T [n];
    for (int i = 1; i <= n; i++) {
        if (i % 15 == 0) {
            A.matvec(x, r, -1);
            daxpy(b, r, n);
        }
        double alpha = ddot(r, r, n);
        double beta = 1 / alpha;
        A.matvec(p, tmp);
        alpha /= ddot(tmp, p, n);
        daxpy(tmp, r, n, -alpha);
        daxpy(p, x, n, alpha);
        
        if (pow(ddot(r, r, n), 0.5) / r0  < eps) {
            cout << "EPS: " << pow(ddot(r, r, n), 0.5) / r0 << endl; 
            cout << i << " iterations" << endl;
            return;
        }
        beta *= ddot(r, r, n);
        daxpy(p, p, n, beta - 1);
        daxpy(r, p, n);
    }
    cout << "EPS: " << pow(ddot(r, r, n), 0.5) / r0 << endl; 
    cout << endl << "Full iterations" << endl;
}


class Matrix {
    private:
        int n;
        double* A;
    public:
        Matrix (int size) {
            n = size;
            A = new double [n * n];
            for (int i = 0; i < n; i++) {
                double sum = 0;
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
        }

        int order() {
            return n;
        }

        void matvec(double* p, double* tmp, double alpha=1) {
            int i, j;    
            #pragma omp parallel for num_threads(num_threads)
            for (i = 0; i < n; i++) {
                tmp[i] = alpha * A[i * n] * p[0];
                for (j = 1; j < n; j++) {
                    tmp[i] += alpha * A[i * n + j] * p[j]; 
                }
            }
        }
};


int main(int argc, char *argv[]) {
    int n = std::stoi(argv[1]);
    if (argc > 2) {
        num_threads = std::stoi(argv[2]); 
    }
    srand(time(NULL));
    Matrix A(n);

    double *b, *x_sol, *x_0;
    double eps = 1e-8;
    b = new double [n];
    x_sol = new double[n];
    x_0 = new double [n];

    random_init(x_sol, n);
    random_init(x_0, n);
    A.matvec(x_sol, b);


    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    SG(A, b, x_0, eps);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Time for SG: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
}

