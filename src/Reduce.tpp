#include "..\include\Core.h"


template <typename T>
void ReduceSum(TensorMem<T> &X, TensorMem<T> &Y, int axis) {
    assert(axis >= 0 && axis <= 3);
    int N = X.shape.N, H = X.shape.H, W = X.shape.W, C = X.shape.C;
    int y_size = Y.shape.N* Y.shape.H* Y.shape.W* Y.shape.C;
    assert((Y.shape.N == N) + (Y.shape.H == H) + (Y.shape.W == W) + (Y.shape.C == C) == 3);
    int n, h, w, c, temp = 0;
    int* on = &n, *oh = &h, *ow = &w, *oc = &c;
    switch (axis) {
        case N_AXIS: on = &temp; break;
        case H_AXIS: oh = &temp; break;
        case W_AXIS: ow = &temp; break;
        case C_AXIS: oc = &temp;
    }
    
    memset(Y.raw(), 0, y_size* sizeof(T));
    for (n = 0; n < N; n++) 
    for (h = 0; h < H; h++)
    for (w = 0; w < W; w++)
    for (c = 0; c < C; c++) 
        Y.at(*on, *oh, *ow, *oc) += X.get(n, h, w, c);
}
template <typename T>
TensorMem<T>* ReduceSum(TensorMem<T> &X, int axis) {
    assert(axis >= 0 && axis <= 3);
    Shape shape;
    int N = X.shape.N, H = X.shape.H, W = X.shape.W, C = X.shape.C;
    shape.N = N, shape.H = H, shape.W = W, shape.C = C;
    int n, h, w, c, temp = 0;
    int* on = &n, *oh = &h, *ow = &w, *oc = &c;
    switch (axis) {
        case N_AXIS: shape.N = 1; on = &temp; break;
        case H_AXIS: shape.H = 1; oh = &temp; break;
        case W_AXIS: shape.W = 1; ow = &temp; break;
        case C_AXIS: shape.C = 1; oc = &temp;
    }
    TensorMem<T>* Y = new TensorMem<T>(shape);
    
    for (n = 0; n < N; n++) 
    for (h = 0; h < H; h++)
    for (w = 0; w < W; w++)
    for (c = 0; c < C; c++) 
        Y->at(*on, *oh, *ow, *oc) += X.get(n, h, w, c);
    
    return Y;
}

template <typename T>
void ReduceMean(TensorMem<T> &X, TensorMem<T> &Y, int axis) {
    assert(axis >= 0 && axis <= 3);
    int N = X.shape.N, H = X.shape.H, W = X.shape.W, C = X.shape.C;
    int y_size = Y.shape.N* Y.shape.H* Y.shape.W* Y.shape.C;
    assert((Y.shape.N == N) + (Y.shape.H == H) + (Y.shape.W == W) + (Y.shape.C == C) == 3);
    int n, h, w, c, size, temp = 0;
    int* on = &n, *oh = &h, *ow = &w, *oc = &c;
    switch (axis) {
        case N_AXIS: size = N; on = &temp; break;
        case H_AXIS: size = H; oh = &temp; break;
        case W_AXIS: size = W; ow = &temp; break;
        case C_AXIS: size = C; oc = &temp;
    }
    
    memset(Y.raw(), 0, y_size* sizeof(T));
    for (n = 0; n < N; n++) 
    for (h = 0; h < H; h++)
    for (w = 0; w < W; w++)
    for (c = 0; c < C; c++) 
        Y.at(*on, *oh, *ow, *oc) += X.get(n, h, w, c);

    for (int i = 0; i < y_size; i++)
        Y.raw()[i] /= size;
}
template <typename T>
TensorMem<T>* ReduceMean(TensorMem<T> &X, int axis) {
    assert(axis >= 0 && axis <= 3);
    Shape shape;
    int N = X.shape.N, H = X.shape.H, W = X.shape.W, C = X.shape.C;
    shape.N = N, shape.H = H, shape.W = W, shape.C = C;
    int n, h, w, c, size, temp = 0;
    int* on = &n, *oh = &h, *ow = &w, *oc = &c;
    switch (axis) {
        case N_AXIS: size = N; shape.N = 1; on = &temp; break;
        case H_AXIS: size = H; shape.H = 1; oh = &temp; break;
        case W_AXIS: size = W; shape.W = 1; ow = &temp; break;
        case C_AXIS: size = C; shape.C = 1; oc = &temp;
    }
    TensorMem<T>* Y = new TensorMem<T>(shape);
    
    for (n = 0; n < N; n++) 
    for (h = 0; h < H; h++)
    for (w = 0; w < W; w++)
    for (c = 0; c < C; c++) 
        Y->at(*on, *oh, *ow, *oc) += X.get(n, h, w, c);

    int y_size = shape.N* shape.H* shape.W* shape.C;
    for (int i = 0; i < y_size; i++)
        Y->raw()[i] /= size;
    
    return Y;
}

template <typename T>
void ReduceProd(TensorMem<T> &X, TensorMem<T> &Y, int axis) {
    assert(axis >= 0 && axis <= 3);
    int N = X.shape.N, H = X.shape.H, W = X.shape.W, C = X.shape.C;
    int y_size = Y.shape.N* Y.shape.H* Y.shape.W* Y.shape.C;
    assert((Y.shape.N == N) + (Y.shape.H == H) + (Y.shape.W == W) + (Y.shape.C == C) == 3);
    int n, h, w, c, size, temp = 0;
    int* on = &n, *oh = &h, *ow = &w, *oc = &c;
    switch (axis) {
        case N_AXIS: size = N; on = &temp; break;
        case H_AXIS: size = H; oh = &temp; break;
        case W_AXIS: size = W; ow = &temp; break;
        case C_AXIS: size = C; oc = &temp;
    }
    
    for (int i = 0; i < y_size; i++)
        Y.raw()[i] = static_cast<T>(1);

    for (n = 0; n < N; n++) 
    for (h = 0; h < H; h++)
    for (w = 0; w < W; w++)
    for (c = 0; c < C; c++) 
        Y.at(*on, *oh, *ow, *oc) *= X.get(n, h, w, c);
}
template <typename T>
TensorMem<T>* ReduceProd(TensorMem<T> &X, int axis) {
    Shape shape;
    if (axis == N_AXIS) shape.N = 1;
    else shape.N = X.shape.N;
    if (axis == H_AXIS) shape.H = 1;
    else shape.H = X.shape.H;
    if (axis == W_AXIS) shape.W = 1;
    else shape.W = X.shape.W;
    if (axis == C_AXIS) shape.C = 1;
    else shape.C = X.shape.C;
    TensorMem<T>* Y = new TensorMem<T>(shape);
    ReduceProd(X, *Y, axis);
    return Y;
}
