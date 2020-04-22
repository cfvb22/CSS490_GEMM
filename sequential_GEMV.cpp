/* *****************************************************************************
 CSS 490    : High Performance Computing
 Instructor : Dr. Parsons
 Written by : Camila Valdebenito
 Date       : April 18, 2020
 Program 1- part 3
 Comparing performance between Intel MKL dgemv and sequential gemv
 *******************************************************************************
 
 --------------------------------------------------------------------------------
 INSTRUCTIONS
 ------------
 To compile :
 cd /opt/intel/compilers_and_libraries_2020.1.216/mac/bin
 source compilervars.sh intel64
 g++ sequential_GEMV.cpp -lmkl_rt -o mv_mult_sequential
 
 To run     :
 Client can run the file on its own or specify the matric dimension
 1) ./mv_mult_sequential
 2) ./mv_mult_sequential [insert N]
 --------------------------------------------------------------------------------
 
 Program Description:
    - This program compares the performance of a naive matrix mulitplication
    function against INTEL MKL DGEMV. Prints out GFLOPS of each method and
    returns the residual.
 
 IMPLEMENTS    y = y + A * X
 A = input matrix (N x N)
 X = input vector (N x 1)
 
 
 
 ********************************************************************************/

#include <iostream>
#include<math.h>
#include <mkl.h>
#include "seq_gemv.h"
#include <time.h>
#include <cmath>


using namespace std;

double calculate_residual(double*, double*, double*, int);

int main(int argc, const char * argv[])
{
    double *a;                  // input matrix
    double *x;                  // input vector
    double *y_mkl;              // output vector for MKL DGEMV
    double *y_seq;              // output vector for sequential DGEMV

    double alpha, beta;
    int N, lda, incx, incy;
    int LOOP_COUNT;             // repetitions
    clock_t start, stop;        // timer
    
    // Takes matrix dimensions from the user
    if(argc < 2)
    {
        cout << "\nEnter the matrix size N = ";
        cin >> N;
        cout << "N is set to: " << N << endl;
    }
    else
    {
        N = atoi(argv[1]);
    }
    

    
    // Initializes variables
    alpha = 1.0;
    beta = 0;
    incx = 1;
    incy = 1;
    lda = N;
    LOOP_COUNT = 50;
    
    // Allocates 64-byte-aligned memory
    a = (double*) mkl_malloc( N * N * sizeof(double), 64 );
    x = (double*) mkl_malloc( N * sizeof(double), 64 );
    y_seq = (double*) mkl_malloc( N * sizeof(double), 64 );
    y_mkl = (double*) mkl_malloc( N * sizeof(double), 64 );


    
    initialize_matrix<double>(N, N, a);
    initialize_matrix<double>(N, 1, x);
    initialize_y<double>(N, y_seq);
    initialize_y<double>(N, y_mkl);
    

    //print_matrix<double>(N,(char *)"a", a);
    //print_vector<double>(N, (char *)"x", x);
    
    double seq_time_start = dsecnd();
    for (int i = 0; i < LOOP_COUNT; i++)
    {
        seq_gemv(a, x, y_seq, N, N);
    }
    double seq_time_end = dsecnd();
    // Computes execution time of sequential gemv
    double seq_time_avg = (seq_time_end - seq_time_start) / LOOP_COUNT;


    double mkl_time_start = dsecnd();
    for (int i = 0; i < LOOP_COUNT; i++)
    {
        cblas_dgemv(CblasColMajor, CblasNoTrans,
                    N, N, alpha, a, lda, x, incx, beta, y_mkl, incy);
    }
    double mkl_time_end = dsecnd();
    double mkl_time_avg = (mkl_time_end - mkl_time_start) / LOOP_COUNT;

    
    // Computes gFlop
    double gflop = (2.0 * N * N) * 1E-9;
    
//    print_vector<double>(N, (char *)"y_seq", y_seq);
//    print_vector<double>(N, (char *)"y_mkl", y_mkl);
    
    
    printf("\n\nComparing Performance between MKL and sequential DGEMV\n");
    printf("-------------------------------------------------------\n");

    printf ("PERFORMANCE METRICS:\n");
    printf("\tNumber of reps           :  %d\n", LOOP_COUNT);
    printf("\tMatrix dimension         :  %d\n", N);
    
    
    // Calculating GFlops
    printf("\tIntelMKL DGEMV           :  %5f GFlops\n", gflop / mkl_time_avg);
    printf("\tSequential DGEMV         :  %.5f  GFlops\n", gflop / seq_time_avg);
    
    
    double *y_residual = (double*) mkl_malloc( N * sizeof(double), 64 );
    double sum_squared_residual = calculate_residual(y_seq, y_mkl, y_residual, N) * 1E-10;
    
    printf("\tSum of squared residuals :  %f e-10\n", sum_squared_residual);
    //print_vector<double>(N, (char *)"residual", y_residual);
    
    // Deallocates memory
    mkl_free(a);
    mkl_free(x);
    mkl_free(y_mkl);
    mkl_free(y_seq);
    mkl_free(y_residual);

    
    printf ("\n*** Program completed. *** \n\n");
    
    
    return 0;
}

double calculate_residual(double* y_mine, double* y_MKL, double* y_residual, int N)
{
    double ss_residual;
    initialize_y<double>(N, y_residual);
    
    
    for (int i = 0; i < N; i++)
    {
        y_residual[i] = y_MKL[i] - y_mine[i];
        
        ss_residual += pow(y_residual[i], 2);
    }
    //print_vector<double>(N, (char *)"residual", y_residual);
    return ss_residual;
    
}


