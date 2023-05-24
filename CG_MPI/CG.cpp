#include <iostream>
#include <cstring>
#include <random>
#include "mpi.h"
#include <string>
#include <chrono>
#include <fstream>

using namespace std;

class Matrix {
    public:
        int col, row;
        MPI_Comm comm;
        int rank, size;
        double* local_A;

        Matrix (double *local_x_sol, int col, MPI_Comm comm) : col(col), comm(comm) {
            MPI_Comm_rank(comm, &rank);
            MPI_Comm_size(comm, &size);
            string file_name = "part" + to_string(rank) + ".txt"; 

            row = col / size + ((rank < col % size) ? 1 : 0);

            local_A = new double[row * col];

            // чтение матриц из файлов
            ifstream fin(file_name.c_str());
            for (int i = 0; i < row * col; i++) {
                fin >> local_A[i];
            }
            for (int i = 0; i < row; i++) {
                fin >> local_x_sol[i];
            }
            fin.close();
        }

        void matvec(double *local_x_sol_recieve, double *res, double alpha=1.0) {
            int extra = ((rank < col % size) ? 0 : 1); 
            int offset = (col / size) * rank + ((rank < col % size) ? rank : col % size);
            int dest = (rank + size - 1) % size;
            int source = (rank + 1) % size;
            double *local_x_sol_send = new double[row + extra];
            memcpy(local_x_sol_send, local_x_sol_recieve, sizeof(double) * (row + extra));

            memset(res, 0, sizeof(double) * row);
            for (int k = 0; k < size; k++) {
                int col_block = col / size + (((k + rank) % size < col % size) ? 1 : 0);
                MPI_Request reqs[2];
                MPI_Status stats[2];
                MPI_Irecv(local_x_sol_recieve, row + extra, MPI_DOUBLE, source, 21, comm, &reqs[0]);
                MPI_Isend(local_x_sol_send, row + extra, MPI_DOUBLE, dest, 21, comm, &reqs[1]); 
                for (int i = 0; i < row; i++) {
                    for (int j = 0; j < col_block; j++) {
                        res[i] += alpha * local_x_sol_send[j] * local_A[i * col + (j + offset)];
                    }
                }
                offset = (offset + col_block) % col; 
                MPI_Waitall(2, reqs, stats);
                memcpy(local_x_sol_send, local_x_sol_recieve, sizeof(double) * (row + extra));
                MPI_Barrier(comm);
            }
            delete[] local_x_sol_send;
        }

        ~Matrix() {
            delete[] local_A;
        }
};

template <class T>
double ddot(T *a, T *b, int n, MPI_Comm comm) {
    double sum = 0;
    for (int i = 0 ; i < n ; i++) {
        sum += a[i] * b[i];
    }
    double global_r0;
    MPI_Allreduce(&sum, &global_r0, 1, MPI_DOUBLE, MPI_SUM, comm);
    return global_r0;
}

template <class T>
void daxpy(T *x, T *y, int n, double alpha = 1) {
    for (int i = 0; i < n; i++) {
        y[i] += x[i] * alpha;
    }
}


template <class T, class Matrix>
void SG(Matrix &A, T *b, T *x, double eps) {
    int n = A.row;
    T *r, *p;
    r = new T [n];
    p = new T [n];

    A.matvec(x, r, -1);
    daxpy(b, r, n);
  
    double r0 = pow(ddot(r, r, n, A.comm), 0.5);

    memcpy(p, r, sizeof(T) * n); 
    T *Ap = new T [n];
    for (int i = 1; i <= A.col; i++) {
        if (i % 15 == 0) {
            A.matvec(x, r, -1);
            daxpy(b, r, n);
        }
        double alpha = ddot(r, r, n, A.comm);
        double beta = 1 / alpha;
        A.matvec(p, Ap);
        alpha /= ddot(Ap, p, n, A.comm);

        daxpy(Ap, r, n, -alpha);
        daxpy(p, x, n, alpha);
        
        double change = pow(ddot(r, r, n, A.comm), 0.5) / r0;
        if (change < eps) {
            if (A.rank == 0) {
                cout << "EPS: " << change << endl; 
                cout << i << " iterations" << endl;
            }
            return;
        }
        beta *= ddot(r, r, n, A.comm);
        daxpy(p, p, n, beta - 1);
        daxpy(r, p, n);
    }
    cout << "EPS: " << pow(ddot(r, r, n, A.comm), 0.5) / r0 << endl; 
    cout << endl << "Full iterations" << endl; 
}


int main(int argc, char *argv[]) {    
    MPI_Init(&argc, &argv);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution <>floatDist(-1, 0);

    int rank, size;
    double eps = 1e-8;
    int n = stoi(argv[1]);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int m = n / size + ((rank < n % size) ? 1 : 0);
    int extra = ((rank < n % size) ? 0 : 1); 

    double *local_x_sol_recieve = new double[m + extra]{0}; // решение
    double *local_b = new double[m]{0};
    double *x_0 = new double [m];
    
    for (int i = 0; i < m; i++) {
        x_0[i] = (-floatDist(gen));
    }
    Matrix A(local_x_sol_recieve, n, MPI_COMM_WORLD);
    A.matvec(local_x_sol_recieve, local_b);
   
    SG(A, local_b, x_0, eps);

    delete[] local_x_sol_recieve;
    delete[] local_b;

    MPI_Finalize();
}

