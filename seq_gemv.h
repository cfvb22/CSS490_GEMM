/* *****************************************************************************
 CSS 490    : High Performance Computing
 Instructor : Dr. Parsons
 Written by : Camila Valdebenito
 Date       : April 18, 2020
 Program 1- part 3
 Comparing performance between Intel MKL dgemv and sequential gemv
 *******************************************************************************

 Program Description: This is a template for the sequential gemv

 IMPLEMENTS    y = y + A * X
    A = input matrix (N x N)
    X = input vector (N x 1)
 
 
********************************************************************************/


#ifndef seq_gemv_h
#define seq_gemv_h
using namespace std;


template <class T>
void seq_gemv(T *a, T *x, T *y, int N, int M)
{
    for (int i = 0; i < M;i++)
    {
        y[i] = 0;
        for (int j = 0; j < N; j++)
        {
            y[i] += a[N*j+i] * x[j];
        }
    }
    
}

// Prints matrix to std out
template <class T>
void print_matrix(int N, char * name, double* array)
{
    cout << "\nPrinting matrix " << name << ":\n";
    for (int i = 0; i < N; i++)
    {
        for (int j= 0; j < N; j++)
        {
            printf("%g\t",array[N*j+i]);
        }
        cout << endl;
    }
}

// Prints vector to std out
template <class T>
void print_vector(int N, char * name, double* array)
{
    cout << "\nPrinting vector " << name << ":\n";
    for (int i= 0; i < N; i++)
    {
        printf("%g\t",array[i]);
    }
    cout << endl;
}



// Initializes matrix with given dimensions N x M
template <class T>
void initialize_matrix(int N, int M, double* array)
{
    
    for (int i = 0; i < N * M; i++)
    {
        array[i] = rand( );
        
    }
}

template <class T>
void initialize_y(int N, double* array)
{
    for(int i = 0; i < N; i++)
    {
        array[i] = 0;
    }
}




#endif /* seq_gemv_h */
