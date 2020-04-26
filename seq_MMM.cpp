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
 
 To compile :
 cd /opt/intel/compilers_and_libraries_2020.1.216/mac/bin
 source compilervars.sh intel64
 g++ seq_MMM.cpp -lmkl_rt -o Lab2_MM_seq
 
 To run     :
 Client can run the file on its own or specify the matric dimension
 1) ./Lab2_MM_seq
 2) ./Lab2_MM_seq [insert N]
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
#include <mkl.h>
#include <thread>
#include <pthread.h>


// maximum number of threads
int MAX = 4;
int MAX_THREADS = 4;//thread::hardware_concurrency();
// represents the current thread
int step_i = 0;




int main(int argc, const char * argv[]) {
    clock_t start_seq, stop_seq, start_parallel, stop_parallel, start_parallel_tiled, stop_parallel_tiled, start_MKL, stop_MKL;
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
    LOOP_COUNT = 2;
    
    // Allocates memory for matrices used for CBlas
    A = (double*) malloc( N * N * sizeof(double) );
    B = (double*) malloc( N * N * sizeof(double) );
    C = (double*) malloc( N * N * sizeof(double) );


    initialize_matrix<double>(N, N, A);
    initialize_matrix<double>(N, N, B);
    initialize_matrix<double>(N, N, C, 0);
    
    
    // Turns 1d arrays into multidimensional arrays to be used in
    // our dgemm parallel and sequential implementation
    // declare the new matrix objects of size N
    Matrix matA, matB, matC, matD, matE;
    matA.size = N;
    matB.size = N;
    matC.size = N;
    matD.size = N;
    matE.size = N;
    
    // initializes all three matrices with zeroes
    matA.initialize_matrix(0);
    matB.initialize_matrix(0);
    matC.initialize_matrix(0);
    matD.initialize_matrix(0);
    matE.initialize_matrix(0);
    
    // converts the arrays generated previously into N x N matrices
    to_multidimension(A, matA, N);
    to_multidimension(B, matB, N);
    
    
    
    // Computes the average execution time of the sequential gemm()
    start_seq = clock();
    for (int i = 0; i < LOOP_COUNT; i++)
    {
        seq_gemm(matA, matB, matC, N, N, N, alpha, beta);
    }
    stop_seq = clock();
    
    
    // Computes the average execution time of the naive parallel gemm()
    start_parallel = clock();
    
    for (int i = 0; i < LOOP_COUNT; i++)
    {
        parallel_gemm(matA, matB, matD, N, N, N, alpha, beta);
    }
    stop_parallel = clock();

    
    // Computes the average execution time of parallel gemm tiled
    start_parallel_tiled = clock();
    for (int i = 0; i < LOOP_COUNT; i++) {

        parallel_gemm_tiled(matA, matB, matE, N, N, N, alpha, beta);
    }
    stop_parallel_tiled = clock();

    
    // Computes the average execution time of MKL Dgemm
    start_MKL = clock();
    for (int i = 0; i < LOOP_COUNT; i++)
    {
        cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,N,N,N,alpha,A,N,B,N,beta,C,N);
    }
    stop_MKL = clock();
    
    
    
    // Computes avg execution time of parallel gemm
    double parallel_time_avg = (stop_parallel - start_parallel) / ( LOOP_COUNT * (double)CLOCKS_PER_SEC );
    
    // Computes avg execution time of sequential gemm
    double seq_time_avg = (stop_seq - start_seq) / ( LOOP_COUNT * (double)CLOCKS_PER_SEC );

    // Computes avg execution time of parallel tiled gemm
    double parallel_tiled_time_avg = (stop_parallel_tiled - start_parallel_tiled) / (LOOP_COUNT * (double) CLOCKS_PER_SEC);


//    // Computes and records MKL dgemm execution time
    double MKL_time_avg = (stop_MKL - start_MKL) / ( LOOP_COUNT * (double)CLOCKS_PER_SEC );
    

    // Computes gFlop
    double gflop = (2.0 * N * N * N) * 1E-9;
    
    
    
    
//    // Prints matrices for testing purposes
//    //print_matrix<double>(N, (char *)"A", A);
//    matA.print((char *)"A");
//    //print_matrix<double>(N, (char *)"B", B);
//    matB.print((char *)"B");
//    print_matrix<double>(N, (char *)"C - DGEMM Cblas Implementation", C);
//    matC.print((char *)"C - DGEMM Sequential ");
//    matD.print((char *)"C - DGEMM Naive Parallel");
//    matE.print((char *)"C - DGEMM Parallel with Tiling");
    
    cout << "Number of concurrent threads supported are: "
         << MAX_THREADS << endl << endl;
    
    
    printf("\n\nComparing Performance between Parallel and Sequential GEMM\n");
    printf("-----------------------------------------------------------\n");
    printf("Number of reps             :  %d\n", LOOP_COUNT);
    printf("Matrix dimension           :  %d\n", N);
    printf ("\nPERFORMANCE RESULTS:\n");
    
    printf ("- BASELINE: MKL Implementation \n");
    printf("\tAvg execution time       :  %f secs\n" ,MKL_time_avg);
    // Calculating GFlops
    printf("\tGFLOP                    :  %.6f\n ", gflop);
    printf("\tGFLOP / sec              :  %.6f  GFlops\n", gflop / MKL_time_avg);
    
    printf ("1) Sequential Implementation\n");
    printf("\tSum of squared residual  :  %f\n", calc_residual(C, matC, N, N));
    printf("\tAvg execution time       :  %f secs\n" ,seq_time_avg);
    // Calculates GFlops
    printf("\tGFLOP                    :  %.6f\n ", gflop);
    printf("\tGFLOP / sec              :  %.6f  GFlops\n", gflop / seq_time_avg);
    
    printf ("2) Parallel Implementation (Naive)\n");
    printf("\tSum of squared residual  :  %f\n", calc_residual(C, matD, N, N));
    printf("\tAvg execution time       :  %f secs\n" ,parallel_time_avg);
    // Calculating GFlops
    printf("\tGFLOP                    :  %.6f\n ", gflop);
    printf("\tGFLOP / sec              :  %.6f  GFlops\n", gflop / parallel_time_avg);

    printf ("3) Parallel Implementation with Tiling\n");
    printf("\tSum of squared residual  :  %f\n", calc_residual(C, matE, N, N));
    printf("\tAvg execution time       :  %f secs\n" ,parallel_tiled_time_avg);
    // Calculating GFlops
    printf("\tGFLOP                    :  %.6f\n ", gflop);
    printf("\tGFLOP / sec              :  %.6f  GFlops\n", gflop / parallel_tiled_time_avg);
    

    
    // Deallocates memory
    free(A);
    free(B);
    free(C);
    matA.deallocate_matrix();
    matB.deallocate_matrix();
    matC.deallocate_matrix();
    matD.deallocate_matrix();
    matE.deallocate_matrix();
    
    printf ("\n*** Program completed. *** \n\n");
    
    return 0;
}


