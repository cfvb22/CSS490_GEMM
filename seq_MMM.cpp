/* *****************************************************************************
 CSS 490    : High Performance Computing
 Instructor : Dr. Parsons
 Written by : Camila Valdebenito
 Date       : April 20, 2020
 Lab2
 Sequential Matrix Multiplication
 
 --------------------------------------------------------------------------------
 INSTRUCTIONS
 ------------
 To compile : g++ seq_MMM.cpp

 To run     :
 Client can run the file on its own or specify the matric dimension
 1) ./a.out
 2) ./a.out [insert N]
 -------------------------------------------------------------------------------
 
 Program Description:
 
 - This program computes the produt of two square matrices sequentially
 
 IMPLEMENTS    C = C + A * B
 A = input matrix (N x N)
 B = input matrix (N x N)
 C = output matrix (N X N)
 ******************************************************************************/

#include "seq_functions.h"
#include <iostream>
using namespace std;
#include <stdio.h>
#include <stdlib.h>
//#include <mkl.h>
#include <thread>
#include <pthread.h>

// maximum size of matrix

//#define MAX 4
//int MAX = thread::hardware_concurrency();
// maximum number of threads
int MAX = 4;
int MAX_THREADS = thread::hardware_concurrency();
// represents the current thread
int step_i = 0;


int main(int argc, const char * argv[]) {
    clock_t start_seq, stop_seq, start_parallel, stop_parallel, start_parallel_tiled, stop_parallel_tiled;
    int N;
    double* A;
    double* B;
    double* C;
    double alpha, beta;
    int incx, incy;
    int LOOP_COUNT;
    
    printf("\n\n>>WELCOME! This program computes the product of two squared matrices.<<\n\n ");
    
    // If user has not specified, asks for matrix dimension
    if(argc < 2)
    {
        cout << "Enter the matrix size N = ";
        cin >> N;
        
    }
    else
    {
        N = atoi(argv[1]);
    }
    cout << "N is set to: " << N << endl<< endl;
    
    // Initializes variables
    alpha = 1.0;
    beta = 0;
    incx = 1;
    incy = 1;
    LOOP_COUNT = 1;
    
    // Allocates memory
    A = (double*) malloc( N * N * sizeof(double) );
    B = (double*) malloc( N * N * sizeof(double) );
    C = (double*) malloc( N * N * sizeof(double) );


    initialize_matrix<double>(N, N, A);
    initialize_matrix<double>(N, N, B);
    initialize_matrix<double>(N, N, C, 0);
    
    
    // Turn 1d arrays into multidimensional arrays to be used in
    // our dgemm parallel implementation
    // declare the new matrix objects of size N
    Matrix matA, matB, matC, matD;
    matA.size = N;
    matB.size = N;
    matC.size = N;
    matD.size = N;
    
    // initializes all three matrices with zeroes
    matA.initialize_matrix(0);
    matB.initialize_matrix(0);
    matC.initialize_matrix(0);
    matD.initialize_matrix(0);
    
    // converts the arrays generated previously into N x N matrices
    to_multidimension(A, matA, N);
    to_multidimension(B, matB, N);
    
    
    
    // Computes the average execution time of the sequential gemm()
    //double seq_time_start = dsecnd();
    start_seq = clock();
    for (int i = 0; i < LOOP_COUNT; i++)
    {
        seq_gemm(A, B, C, N, N, N, alpha, beta);

    }
    stop_seq = clock();
    
    
    // Computes the average execution time of the parallel gemm()
    start_parallel = clock();
    
    for (int i = 0; i < LOOP_COUNT; i++)
    {
        parallel_gemm(matA, matB, matC, N, N, N, alpha, beta);

    }
    stop_parallel = clock();

    // Computes the average execution time of parallel gemm tiled
    start_parallel_tiled = clock();
    for (int i = 0; i < LOOP_COUNT; i++) {

        parallel_gemm_tiled(matA, matB, matD, N, N, N, alpha, beta);
    }
    stop_parallel_tiled = clock();

    // Computes avg execution time of parallel gemm
    double parallel_time_avg = (stop_parallel - start_parallel) / ( LOOP_COUNT * (double)CLOCKS_PER_SEC );
    
    // Computes avg execution time of sequential gemm
    double seq_time_avg = (stop_seq - start_seq) / ( LOOP_COUNT * (double)CLOCKS_PER_SEC );

    // Computes avg execution time of parallel tiled gemm
    double parallel_tiled_time_avg = (stop_parallel_tiled - start_parallel_tiled) / (LOOP_COUNT * (double) CLOCKS_PER_SEC);

    // Computes gFlop
    double gflop = (2.0 * N * N * N) * 1E-9;
    
    
    double sum_sq_residual = calc_residual(C, matC, N, N);
    
//    print_matrix<double>(N, (char *)"A", A);
//    print_matrix<double>(N, (char *)"B", B);
//    print_matrix<double>(N, (char *)"C", C);
//    print_matrix(N, (char *) "D", matC);
    //const int con_threads = thread::hardware_concurrency();
    cout << "Number of concurrent threads supported are: "
         << MAX_THREADS << endl << endl;
    
    
    printf("\n\nComparing Performance between Parallel and Sequential GEMM\n");
    //printf("Computing the Performance of Sequential GEMM\n");
    printf("-----------------------------------------------------------\n");
    printf("Number of reps             :  %d\n", LOOP_COUNT);
    printf("Matrix dimension           :  %d\n", N);
    printf ("\nPERFORMANCE RESULTS:\n");
    
    printf ("1) Sequential Implementation\n");
    printf("\tAvg execution time       :  %f secs\n" ,seq_time_avg);
    // Calculates GFlops
    printf("\tGFLOP                    :  %.5f\n ", gflop);
    printf("\tGFLOP / sec              :  %.5f  GFlops\n", gflop / seq_time_avg);
    
    printf ("2) Parallel Implementation\n");
    printf("\tAvg execution time       :  %f secs\n" ,parallel_time_avg);
    // Calculating GFlops
    printf("\tGFLOP                    :  %.5f\n ", gflop);
    printf("\tGFLOP / sec              :  %.5f  GFlops\n", gflop / parallel_time_avg);

    printf ("2) Parallel Implementation Tiled\n");
    printf("\tAvg execution time       :  %f secs\n" ,parallel_tiled_time_avg);
    // Calculating GFlops
    printf("\tGFLOP                    :  %.5f\n ", gflop);
    printf("\tGFLOP / sec              :  %.5f  GFlops\n", gflop / parallel_tiled_time_avg);

    // With three different matrices we should change this so we dont just compare
    // two matrices but all three
    //printf("\nThe sum of squared residual  :  %f\n" , sum_sq_residual);
    
    // Deallocates memory
    free(A);
    free(B);
    free(C);
    
    printf ("\n*** Program completed. *** \n\n");
    
    return 0;
}
