#include <iostream>
#include <chrono>   
#include <vector>
#include <thread>

#include <xmmintrin.h> 

using namespace std;
using namespace std::chrono;


void matTranspose(const float * src, float *dst, const int rows, const int cols)
{
    for (int i = 0; i < cols; i++)
    {
        for (int j = 0; j < rows; j++)
        {
            *(dst + i * rows + j) = *(src + j * cols + i);
        }
    }
}

void initiaMatWithValue(float *mat, const int rows, const int cols, int value)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            *(mat + i * cols + j) = value;
        }
    }
}

void matrixMul(float * A, float * B, float * C, const int Arows, const int Acols, const int Bcols)
{
    for(int i = 0; i < Arows; i ++)
    {
        for (int j = 0; j < Bcols; j ++)
        {
            int elem = 0;
            for (int k = 0; k < Acols; k ++)
            {
                elem += *(A + i * Acols + k) * *(B + k * Bcols + j);
            }
            *(C + i * Bcols + j) = elem;
        }
    }
}

void matrixMulSSE(float * A, float * B, float * C, const int Arows, const int Acols, const int Bcols)
{
    float * BTranspose = (float *)malloc(sizeof(float) * Acols * Bcols);   
    matTranspose(B, BTranspose, Acols, Bcols);
    float temp[4];
    __m128 sse_A_input, sse_B_input;

    for(int i = 0; i < Arows; i ++)
    {
        float * Aptr = A + i * Acols;
        for (int j = 0; j < Bcols; j ++)
        {
            float sum = 0;
            float * Bptr = BTranspose + j * Acols;
            __m128 sse_sum = _mm_setzero_ps();
            for (int k = 0; k < (Acols / 4) * 4; k += 4)
            {
                sse_A_input = _mm_loadu_ps(Aptr + k);
                sse_B_input = _mm_loadu_ps(Bptr + k);
                sse_sum = _mm_add_ps(sse_sum, _mm_mul_ps(sse_A_input, sse_B_input));
            }
            _mm_storeu_ps(temp, sse_sum);
            sum = temp[0] + temp[1] + temp[2] + temp[3];

            for(int k = (Acols / 4) * 4; k < Acols; k ++)
            {
                sum += *(A + i * Acols + k) * *(BTranspose + j * Acols + k);
            }
            *(C + i * Bcols + j) = sum;
        }
    }
    free(BTranspose);
}

void multiThreadMatMul(float * A, float * B, float * C, int Arows, int Acols, int Bcols, string sse, int thread_num=4)
{
    vector<thread> threadPool;
    for(int i = 0; i < thread_num; i ++)
    {   
        int task_scale =  i != thread_num - 1 ?  Arows / thread_num : Arows - (Arows / thread_num) * i;
        if (sse == "SSE")
            threadPool.push_back(thread(matrixMulSSE, A + (Arows / thread_num) * Acols * i, B, C + (Arows / thread_num) * Bcols * i, task_scale, Acols, Bcols));
        else
            threadPool.push_back(thread(matrixMul, A + (Arows / thread_num) * Acols * i, B, C + (Arows / thread_num) * Bcols * i, task_scale, Acols, Bcols));
    }
    for(auto& t : threadPool) t.join();
}
 
void showMat(float * mat, int cols, int show_rows, int show_cols)
{
    for(int i = 0; i < show_rows; i ++)
    {
        for(int j = 0; j < show_cols; j ++)
        {
            cout << *(mat + i * cols + j) << "\t";
        }
        cout << "..." << endl;
    }
    for(int i = 0; i < show_rows + 1; i ++) cout << "..." << "\t";
    cout << endl;
}

void testMatMul(float * A, float * B, float * C, const int Arows, const int Acols, const int Bcols, int flag)
{   
    cout << "Test type: 1: NORMAL, 2: SSE, 3:MULTI THREAD WITH NORMAL, 4: MULTI THREAD WITH SSE" << endl;
    initiaMatWithValue(C, Arows, Bcols, 0);
    auto start = system_clock::now();
    switch (flag)
    {
    case 1:
        matrixMul(A, B, C, Arows, Acols, Bcols);
        break;
    case 2:
        matrixMulSSE(A, B, C, Arows, Acols, Bcols);
        break;
    case 3:
        multiThreadMatMul(A, B, C, Arows, Acols, Bcols, "NORMAL");
        break;
    case 4:
        multiThreadMatMul(A, B, C, Arows, Acols, Bcols, "SSE");
        break;
    default:
        break;
    }
    auto end = system_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    cout << "==================TEST TYPE " << flag << "== " << "cost time:" <<
         double(duration.count()) * microseconds::period::num / microseconds::period::den <<" s ====================" << endl;
    showMat(C, Bcols, 10, 10);
    cout << endl;
}

int main()
{
    int Arows = 1024, Acols = 128, Bcols = 1024;
    float * matrixA = (float *)malloc(sizeof(float) * Arows * Acols);
    float * matrixB = (float *)malloc(sizeof(float) * Acols * Bcols);
    float * matrixC = (float *)malloc(sizeof(float) * Arows * Bcols);

    initiaMatWithValue(matrixA, Arows, Acols, 2);
    initiaMatWithValue(matrixB, Acols, Bcols, 2);

    testMatMul(matrixA, matrixB, matrixC, Arows, Acols, Bcols, 1);
    testMatMul(matrixA, matrixB, matrixC, Arows, Acols, Bcols, 2);
    testMatMul(matrixA, matrixB, matrixC, Arows, Acols, Bcols, 3);
    testMatMul(matrixA, matrixB, matrixC, Arows, Acols, Bcols, 4);
    
    free(matrixA);
    free(matrixB);
    free(matrixC);
    std::cout << "Modify to resolve conflict" << std::endl;
    return 0;
}

