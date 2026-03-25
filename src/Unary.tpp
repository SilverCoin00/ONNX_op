#include "..\include\Core.h"


template <typename T>
void Identity(const TensorMem<T> &X, TensorMem<T> &Y) {
    assert(X.shape == Y.shape);
    int size = X.shape.N* X.shape.H* X.shape.W* X.shape.C;
    for (int i = 0; i < size; i++) 
        Y.raw()[i] = X.raw()[i];
}
template <typename T>
TensorMem<T>* Identity(const TensorMem<T> &X) {
    TensorMem<T>* Y = new TensorMem<T>(X.shape);
    Identity(X, *Y);
    return Y;
}

template <typename T>
void Constant(const T* X, TensorMem<T> &Y, int size) {
    for (int i = 0; i < size; i++)
        Y.raw()[i] = X[i];
}

template <typename T>
void Constant_of_shape(TensorMem<T> &Y, T val) {
    int size = Y.shape.N* Y.shape.H* Y.shape.W* Y.shape.C;
    for (int i = 0; i < size; i++)
        Y.raw()[i] = val;
}
template <typename T>
TensorMem<T>* Constant_of_shape(const Shape &shape, T val) {
    TensorMem<T>* Y = new TensorMem<T>(shape);
    Constant_of_shape(*Y, val);
    return Y;
}

template <typename T_IN, typename T_OUT>
void Cast(TensorMem<T_IN> &X, TensorMem<T_OUT> &Y) {
    assert(X.shape == Y.shape);
    int size = X.shape.N* X.shape.H* X.shape.W* X.shape.C;
    for (int i = 0; i < size; ++i) 
        Y.raw()[i] = static_cast<T_OUT>(X.raw()[i]);
}
template <typename T_OUT, typename T_IN>
auto Cast(TensorMem<T_IN> &X) -> TensorMem<T_OUT>* {
    TensorMem<T_OUT>* Y = new TensorMem<T_OUT>(X.shape);
    Cast(X, *Y);
    return Y;
}

template <typename T>
void Relu(TensorMem<T> &X, TensorMem<T> &Y) {
    assert(X.shape == Y.shape);
    int size = X.shape.N* X.shape.H* X.shape.W* X.shape.C;
    for (int i = 0; i < size; ++i) {
        T val = X.raw()[i];
        Y.raw()[i] = val > T(0) ? val : T(0);
    }
}
template <typename T>
TensorMem<T>* Relu(TensorMem<T> &X) {
    TensorMem<T>* Y = new TensorMem<T>(X.shape);
    Relu(X, *Y);
    return Y;
}

template <typename T>
void Sqrt(TensorMem<T> &X, TensorMem<T> &Y) {
    static_assert(is_float_point<T>::value, "Error: Floor requires float point type !!");
    assert(X.shape == Y.shape);
    int size = X.shape.N* X.shape.H* X.shape.W* X.shape.C;
    for (int i = 0; i < size; ++i) 
        Y.raw()[i] = static_cast<T>(sqrt(X.raw()[i]));
}
template <typename T>
TensorMem<T>* Sqrt(TensorMem<T> &X) {
    TensorMem<T>* Y = new TensorMem<T>(X.shape);
    Sqrt(X, *Y);
    return Y;
}

template <typename T>
void Floor(TensorMem<T> &X, TensorMem<T> &Y) {
    static_assert(is_float_point<T>::value, "Error: Floor requires float point type !!");
    assert(X.shape == Y.shape);
    int size = X.shape.N* X.shape.H* X.shape.W* X.shape.C;
    for (int i = 0; i < size; ++i) 
        Y.raw()[i] = static_cast<T>(floor(X.raw()[i]));
}
template <typename T>
TensorMem<T>* Floor(TensorMem<T> &X) {
    TensorMem<T>* Y = new TensorMem<T>(X.shape);
    Floor(X, *Y);
    return Y;
}

template <typename T>
void Round(TensorMem<T> &X, TensorMem<T> &Y) {
    static_assert(is_float_point<T>::value, "Error: Floor requires float point type !!");
    assert(X.shape == Y.shape);
    int size = X.shape.N* X.shape.H* X.shape.W* X.shape.C;
    for (int i = 0; i < size; ++i) 
        Y.raw()[i] = static_cast<T>(roundf(X.raw()[i]));
}
template <typename T>
TensorMem<T>* Round(TensorMem<T> &X) {
    TensorMem<T>* Y = new TensorMem<T>(X.shape);
    Round(X, *Y);
    return Y;
}

template <typename T>
void Clip(TensorMem<T> &X, TensorMem<T> &Y, T min, T max) {
    assert(X.shape == Y.shape);
    assert(min <= max);
    int size = X.shape.N* X.shape.H* X.shape.W* X.shape.C;
    for (int i = 0; i < size; ++i) {
        T val = X.raw()[i];
        if (val < min) val = min;
        else if (val > max) val = max;
        Y.raw()[i] = val;
    }
}
template <typename T>
TensorMem<T>* Clip(TensorMem<T> &X, T min, T max) {
    TensorMem<T>* Y = new TensorMem<T>(X.shape);
    Clip(X, *Y, min, max);
    return Y;
}
