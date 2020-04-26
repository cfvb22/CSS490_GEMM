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
#include <deque>
#include <thread>
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
        double denominator = ( (double)( RAND_MAX ) + (double) (1) ) / (double) 500.0;
        elements = new double*[size];
        for ( int i = 0; i < size; i++ )
        {
            elements[i] = new double[size];
            for ( int j = 0; j < size; j++ )
            {
                if( value == 1 )
                {
                    elements[i][j] = rand( ) / denominator;
                }
                else
                {
                    elements[i][j] = 0.0;
                    
                }
            }
        }
    }
    void deallocate_matrix()
    {
        //cout << "Deallocating memory..." << endl;
        //Free each sub-array
        for(int i = 0; i < size; i++)
        {
            delete[] elements[i];
        }
        //Free the array of pointers
        delete[] elements;
    }
    
    // Prints out the matrix to the console
    void print(char * name) {
        cout <<"Printing matrix " << name << ":\n";
        for (int i = 0; i < size; ++i)
        {
            for (int j = 0; j < size; ++j)
            {
                //cout << elements[i][j] << "\t";
                printf("%.6f\t",elements[i][j]);
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

struct tile_arg_struct {
    Matrix A;
    Matrix B;
    Matrix C;
    int n_lower;
    int n_upper;
    int m_lower;
    int m_upper;
    int k_lower;
    int k_upper;
};

//template <class T>
//void seq_gemm(T *A, T *B, T *C, int N, int M, int k, T alpha, T beta)
//{
//    for (int i = 0; i < N; i++)
//    {
//        for (int j = 0; j < M; j++)
//        {
//
//            C[i*N + j] = beta * C[i*N + j];
//            // Compute an element of the output matrix
//            for (int l = 0; l < k; l++)
//            {
//                C[i*N + j] = C[i*N + j] + alpha*A[i*N + l] * B[l*k + j];
//
//            }
//        }
//    }
//
//}

template <class T>
void seq_gemm(Matrix A, Matrix B, Matrix C, int N, int M, int k, T alpha, T beta)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < M; j++)
        {
            
            C.elements[i][j] = beta * C.elements[i][j];
            // Compute an element of the output matrix
            for (int l = 0; l < k; l++)
            {
                C.elements[i][j] = C.elements[i][j] + alpha* A.elements[i][l] * B.elements[l][j];
                
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
    free(args);
    pthread_exit(0);
}


// method computes the naive parallel matrix multiplication
template <class T>
void parallel_gemm(Matrix matA, Matrix matB, Matrix matC, int N, int M, int k, T alpha, T beta)
{
    // declares threads based on numbers of threads supported by hardware
    pthread_t threads[MAX_THREADS];

    // Creating four threads, each evaluating its own part
    for (int i = 0; i < MAX_THREADS; i++)
    {
        struct arg_struct *data;
        data = (arg_struct*) malloc(sizeof(arg_struct));
        data->A = matA;
        data->B = matB;
        data->C = matC;
        //printf("Creating user thread: %d\n", i);
        pthread_attr_t attr;
        pthread_attr_init(&attr);
        pthread_create(&threads[i], &attr, *matrix_multiply,
                reinterpret_cast<void *>(data) );
    }
    cout << endl;
    
    // joining and waiting for all threads to complete
    for (int i = 0; i < MAX_THREADS; i++)
    {
        pthread_join(threads[i], NULL);
    }
    step_i = 0;
}

void* tile_multiply(void * arguments) {
    struct tile_arg_struct *args = (struct tile_arg_struct*)arguments;
    Matrix A = args -> A;
    Matrix B = args -> B;
    Matrix C = args -> C;
    int n_lower = args -> n_lower;
    int n_upper = args -> n_upper;
    int m_lower = args -> m_lower;
    int m_upper = args -> m_upper;
    int k_lower = args -> k_lower;
    int k_upper = args -> k_upper;
    for (int n = n_lower; n <= n_upper; n++) {
        for (int m = m_lower; m <= m_upper; m++) {
            double sum = 0;
            for (int k = k_lower; k <= k_upper; k++) {
                sum += A.elements[n][k] * B.elements[k][m];
            }
            C.elements[n][m] += sum;
        }
    }
    free(args);
    pthread_exit(0);
}

template <class T>
void parallel_gemm_tiled(Matrix A, Matrix B, Matrix C, int N, int M, int k, T alpha, T beta) {
    const int coreCount = MAX_THREADS;//thread::hardware_concurrency();
    const int blockSize = 4096; // This was the block size in my machine.
    // Number of elements in each block. Float = 1024 Double = 512
    const int elementsPerBlock = blockSize / sizeof(alpha);
    deque<pthread_t> tids;
    for (int i = 0; i < N; i+= elementsPerBlock) {
        for (int j = 0; j < N; j+= elementsPerBlock) {
            for (int k = 0; k < N; k+= elementsPerBlock) {

                // Spin off a new thread here to handle this execution
                // need to check that we havent gone over our max number of threads:
                struct tile_arg_struct *args;
                args = (tile_arg_struct*)malloc(sizeof(tile_arg_struct));
                args -> A = A; args -> B = B; args -> C = C;
                args -> n_upper = min(i + elementsPerBlock - 1, N - 1);
                args -> m_upper = min(j + elementsPerBlock - 1, N - 1);
                args -> k_upper = min(k + elementsPerBlock - 1, N - 1);
                args -> n_lower = i;
                args -> m_lower = j;
                args -> k_lower = k;

                if (tids.size() >= coreCount) {
                    pthread_join(tids.back(), NULL);
                    tids.pop_back();
                }
                pthread_t tid;
                pthread_attr_t attr;
                pthread_attr_init(&attr);
                pthread_create(&tid, &attr, tile_multiply,
                               reinterpret_cast<void *>(args));
                tids.push_front(tid);
            }
        }
    }
    int size = tids.size();
    for (int i = 0; i < size; i++) {
        pthread_join(tids.back(), NULL);
        tids.pop_back();
    }
}

// Transforms a 1-d array into a multidimensional array
void to_multidimension ( double* flat_array, Matrix new_Matrix, int N)
{
        for (int i = 0; i < N; i++)
        {
            for (int j= 0; j < N; j++)
            {
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
    double denominator = ( (double)( RAND_MAX ) + (double) (1) ) / (double) 500.0;

    for ( int i = 0; i < N * M; i++ )
    {
        if( value == 1 )
        {
            A[i] = rand( ) / denominator;

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
    matC_seq.deallocate_matrix();
    return sum;
    

}






#ifndef Header_h
#define Header_h


#endif /* Header_h */
