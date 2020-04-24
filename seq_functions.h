/* *****************************************************************************
 CSS 490    : High Performance Computing
 Instructor : Dr. Parsons
 Written by : Camila Valdebenito
 Date       : April 20, 2020
 Lab 2
 
 *******************************************************************************
 
 Program Description: This is a template for the sequential gemm
 
 ********************************************************************************/


#define seq_functions_h
#include <chrono>
//#include <bits/stdc++.h>
#include <pthread.h>
#include <iostream>
#include <math.h> 
using namespace std;
extern int MAX_THREADS;
extern int step_i;

//template <class T>
struct Matrix {
    double ** elements;
    int size;

    // Initializes given matrix with given dimensions, M x N
    // If value is not given, then it will initialize the
    // matrix with random numbers
    void initialize_matrix(int value = 1)
    {
        elements = new double*[size];
        for ( int i = 0; i < size; i++ )
        {
            elements[i] = new double[size];
            for ( int j = 0; j < size; j++ )
            {
                if( value == 1 )
                {
                    elements[i][j] = rand( );
                }
                else
                {
                    elements[i][j] = 0.0;
                    
                }
            }
        }
    }
    
    // Prints out the matrix to the console
    void print(char * name) {
        cout <<"Printing matrix " << name << ":\n";
        for (int i = 0; i < size; ++i)
        {
            for (int j = 0; j < size; ++j)
            {
                //cout << elements[i][j] << "\t";
                printf("%g\t",elements[i][j]);
            }
            cout << endl;
        }
        cout << endl;
    }
};

// creates a struct to pass into threads_create
struct arg_struct {
    
    Matrix A;
    Matrix B;
    Matrix C;

};

template <class T>
void seq_gemm(T *A, T *B, T *C, int N, int M, int k, T alpha, T beta)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < M; j++)
        {

            C[i*N + j] = beta * C[i*N + j];
            // Compute an element of the output matrix
            for (int l = 0; l < k; l++)
            {
                C[i*N + j] = C[i*N + j] + alpha*A[i*N + l] * B[l*k + j];

            }
        }
    }
    
}

// matrix multiply helper method for naive parallel method
void *matrix_multiply(void *data)
{
    int core = step_i++;
    struct arg_struct *args = (struct arg_struct*)data;
    Matrix matA = args-> A;
    Matrix matB = args-> B;
    Matrix matC = args-> C;
    
    int matrix_Size = matA.size;
    
    
    // Each thread computes a fraction of matrix multiplication
    // This fraction depends on the number of cores of the device
    for (int i = core * matrix_Size / MAX_THREADS; i < (core + 1) * matrix_Size / MAX_THREADS; i++)
    {
        for (int j = 0; j < matrix_Size; j++)
        {
            double result = 0.0;
            for (int k = 0; k < matrix_Size; k++)
            {
                const double e1 = matA.elements[i][k];
                const double e2 = matB.elements[k][j];
                result += e1 * e2;
            }
            matC.elements[i][j] = result;
        }
    }
    pthread_exit(0);
}


// method computes the naive parallel matrix multiplication
template <class T>
void parallel_gemm(Matrix matA, Matrix matB, Matrix matC, int N, int M, int k, T alpha, T beta)
{
   
    struct arg_struct *data = (struct arg_struct *) malloc(sizeof(struct arg_struct));
    data->A = matA;
    data->B = matB;
    data->C = matC;

    // declares threads based on numbers of threads supported by hardware
    pthread_t threads[MAX_THREADS];

    // Creating four threads, each evaluating its own part
    for (int i = 0; i < MAX_THREADS; i++)
    {
        printf("Creating user thread: %d\n", i);
        pthread_create(&threads[i], NULL, *matrix_multiply, (void *)data );
    }
    cout << endl;
    
    // joining and waiting for all threads to complete
    for (int i = 0; i < MAX_THREADS; i++)
    {
        pthread_join(threads[i], NULL);
    }

    matA.print((char*) "matA via naive Parallel");
    matB.print((char*) "matB via naive Parallel");
    matC.print((char*) "matC via naive Parallel");
    
    
}

// Transforms a 1-d array into a multidimensional array
void to_multidimension ( double* flat_array, Matrix new_Matrix, int N)
{
        for (int i = 0; i < N; i++)
        {
            for (int j= 0; j < N; j++)
            {
                //printf("hey!\t");//,C[MAX*i+j]);
                new_Matrix.elements[i][j] = flat_array[N * i + j];
            }
        }
    
}



// Prints matrix to std out
template <class T>
void print_matrix(int N, char * name, double* array)
{
    cout <<"Printing matrix " << name << ":\n";
    for (int i = 0; i < N; i++)
    {
        for (int j= 0; j < N; j++)
        {
            printf("%g\t",array[N*i+j]);
        }
        cout << endl;
    }
    cout << endl;
}



// Initializes given matrix with given dimensions, M x N
// If value is not given, then it will initialize the
// matrix with random numbers
template <class T>
void initialize_matrix(int N, int M, T* A, T value = 1)
{

    for ( int i = 0; i < N * M; i++ )
    {
        if( value == 1 )
        {
            A[i] = rand( );

        }
        else
        {
            A[i] = 0;

        }

    }
}

// Calculate sum of squared difference for residual
template <class T>
T calc_residual(T* C, Matrix matC, int N, int M)
{
    Matrix matC_seq;
    matC_seq.size = N;
    matC_seq.initialize_matrix(0);
    to_multidimension(C, matC_seq, N);
    
    double sum = 0;
    for (int i = 0; i < N; i++)
    {
        for(int j = 0; j < N; j++)
        {
            sum += pow((matC_seq.elements[i][j] - matC.elements[i][j]), 2);
        }
        
    }
    return sum;
    
//    double sum = 0;
//    for (int i = 0; i < N * M; i++)
//    {
//        sum += pow((A[i] - B[i]), 2);
//    }
//    return sum;
}






#ifndef Header_h
#define Header_h


#endif /* Header_h */
