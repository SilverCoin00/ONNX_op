#include "..\include\Core.h"


template <typename T>
void Norm(TensorMem<T> &X, TensorMem<T> &Y, TensorMem<T> &gamma, TensorMem<T> &beta, T epsilon, int axis, 
            TensorMem<T> &extra_mem_size_X_reduced_axis) {
    Shape temp_shape = {1, 1, 1, 1};
    TensorMem<T> temp(temp_shape);
    switch (axis) {
        case N_AXIS: temp.raw()[0] = X.shape.N - 1; break;
        case C_AXIS: temp.raw()[0] = X.shape.C - 1; break;
        case H_AXIS: temp.raw()[0] = X.shape.H - 1; break;
        case W_AXIS: temp.raw()[0] = X.shape.W - 1;
    }
    assert(temp.raw()[0] > 0);
    temp.raw()[0] = static_cast<T>(sqrt(temp.raw()[0]));

    ReduceMean(X, extra_mem_size_X_reduced_axis, axis);
    Sub(X, extra_mem_size_X_reduced_axis, X);
    Div(X, temp, Y);
    Mul(Y, Y, Y);
    ReduceSum(Y, extra_mem_size_X_reduced_axis, axis);
    temp.raw()[0] = epsilon;
    Add(extra_mem_size_X_reduced_axis, temp, extra_mem_size_X_reduced_axis);
    Sqrt(extra_mem_size_X_reduced_axis, extra_mem_size_X_reduced_axis);
    Div(X, extra_mem_size_X_reduced_axis, X);

    Mul(gamma, X, X);
    Add(X, beta, Y);
}
template <typename T>
void Norm(TensorMem<T> &X, TensorMem<T> &Y, TensorMem<T> &gamma, TensorMem<T> &beta, T epsilon, int axis, 
            TensorMem<T> &mean, TensorMem<T> &var, TensorMem<T> &std) {
    Shape temp_shape = {1, 1, 1, 1};
    TensorMem<T> temp(temp_shape);
    switch (axis) {
        case N_AXIS: temp.raw()[0] = X.shape.N - 1; break;
        case C_AXIS: temp.raw()[0] = X.shape.C - 1; break;
        case H_AXIS: temp.raw()[0] = X.shape.H - 1; break;
        case W_AXIS: temp.raw()[0] = X.shape.W - 1;
    }
    assert(temp.raw()[0] > 0);

    ReduceMean(X, mean, axis);
    Sub(X, mean, X);
    Mul(X, X, Y);
    ReduceSum(Y, var, axis);
    Div(var, temp, var);
    temp.raw()[0] = epsilon;
    Add(var, temp, var);
    Sqrt(var, std);
    Div(X, std, X);

    Mul(gamma, X, X);
    Add(X, beta, Y);
}

template <typename T>
void Channel_Norm(TensorMem<T> &X, TensorMem<T> &Y, TensorMem<T> &gamma, TensorMem<T> &beta, T epsilon, 
                    Shape start, Shape end) {
    int num = end.C - start.C;
    for (int n = start.N; n < end.N; n++)
    for (int h = start.H; h < end.H; h++)
    for (int w = start.W; w < end.W; w++) {
        T mean = 0, var = 0;
        for (int c = start.C; c < end.C; c++) 
            mean += X.get(n, h, w, c);
        mean /= num;
        for (int c = start.C; c < end.C; c++) {
            T mid = X.get(n, h, w, c) - mean;
            var += mid / (num - 1)* mid;
        }
        var = static_cast<T>(sqrt(var + epsilon));
        for (int c = start.C; c < end.C; c++) {
            int id = c - start.C;
            Y.at(n, h, w, c) = (X.get(n, h, w, c) - mean) / var* gamma.raw()[id] + beta.raw()[id];
        }
    }
}
template <typename T>
void Channel_Norm_2(TensorMem<T> &X, TensorMem<T> &Y, TensorMem<T> &gamma, TensorMem<T> &beta, T epsilon, 
                    Shape start, Shape end) {
    int num = end.C - start.C;
    T adjustment_scale = (T) num / (num - 1);
    T pre_div = (T) 1 / (T) (sqrtf(num - 1));
    for (int n = start.N; n < end.N; n++)
    for (int h = start.H; h < end.H; h++)
    for (int w = start.W; w < end.W; w++) {
        T mean = 0, var = 0;
        for (int c = start.C; c < end.C; c++) {
            T cur = X.get(n, h, w, c);
            mean += cur;
            T cur_n = cur* pre_div;
            var += cur_n* cur_n;
        }
        mean /= num;
        var = (T) 1 / static_cast<T>(sqrtf(var - mean* mean* adjustment_scale + epsilon));
        for (int c = start.C; c < end.C; c++) {
            int id = c - start.C;
            T weight = gamma.raw()[id]* var;
            T bias = beta.raw()[id] - mean* weight;
            Y.at(n, h, w, c) = X.get(n, h, w, c)* weight + bias;
        }
    }
}
