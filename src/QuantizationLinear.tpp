#include "..\include\Core.h"


template <typename T_IN, typename T_OUT>
void QuantizeLinear(TensorMem<T_IN> &X, TensorMem<T_OUT> &Y, TensorMem<T_IN> scale, TensorMem<T_OUT> zero_point) {
    static_assert(is_float_point<T_IN>::value, "Error: Inappropriate type for quantize !!");
    static_assert(is_integral<T_OUT>::value, "Error: Inappropriate type for quantize !!");
    assert(X.shape == Y.shape && scale.shape == zero_point.shape);

    int dn, dh, dw, dc;
    dn = dh = dw = dc = 1;
    int n1, n2, h1, h2, w1, w2, c1, c2;
    if (X.shape.N != scale.shape.N) 
        if (scale.shape.N == 1) dn--;
        else assert(0 && "Error: Inappropriate N-axis size !!");
    if (X.shape.H != scale.shape.H) 
        if (scale.shape.H == 1) dh--;
        else assert(0 && "Error: Inappropriate H-axis size !!");
    if (X.shape.W != scale.shape.W) 
        if (scale.shape.W == 1) dw--;
        else assert(0 && "Error: Inappropriate W-axis size !!");
    if (X.shape.C != scale.shape.C) 
        if (scale.shape.C == 1) dc--;
        else assert(0 && "Error: Inappropriate C-axis size !!");
    
    for (n1 = 0, n2 = 0; n1 < Y.shape.N; n1++, n2 += dn)
    for (h1 = 0, h2 = 0; h1 < Y.shape.H; h1++, h2 += dh)
    for (w1 = 0, w2 = 0; w1 < Y.shape.W; w1++, w2 += dw)
    for (c1 = 0, c2 = 0; c1 < Y.shape.C; c1++, c2 += dc) {
        int32_t div = static_cast<int32_t>(roundf(X.get(n1, h1, w1, c1) / scale.get(n2, h2, w2, c2))) 
                                            + static_cast<int32_t>(zero_point.get(n2, h2, w2, c2));
        Y.at(n1, h1, w1, c1) = clamp<T_OUT>(div);
    }
}
template <typename T_IN, typename T_OUT>
void QuantizeLinear(TensorMem<T_IN> &X, TensorMem<T_OUT> &Y, T_IN scale, T_OUT zero_point) {
    static_assert(is_float_point<T_IN>::value, "Error: Inappropriate type for quantize !!");
    static_assert(is_integral<T_OUT>::value, "Error: Inappropriate type for quantize !!");
    assert(X.shape == Y.shape);

    int size = X.shape.N* X.shape.H* X.shape.W* X.shape.C;
    for (int i = 0; i < size; i++) {
        int32_t div = static_cast<int32_t>(roundf(X.raw()[i] / scale)) + static_cast<int32_t>(zero_point);
        Y.raw()[i] = clamp<T_OUT>(div);
    }
}

template <typename T_IN, typename T_OUT>
void DequantizeLinear(TensorMem<T_IN> &X, TensorMem<T_OUT> &Y, TensorMem<T_OUT> scale, TensorMem<T_IN> zero_point) {
    static_assert(is_integral<T_IN>::value, "Error: Inappropriate type for dequantize !!");
    static_assert(is_float_point<T_OUT>::value, "Error: Inappropriate type for dequantize !!");
    assert(X.shape == Y.shape && scale.shape == zero_point.shape);

    int dn, dh, dw, dc;
    dn = dh = dw = dc = 1;
    int n1, n2, h1, h2, w1, w2, c1, c2;
    if (X.shape.N != scale.shape.N) 
        if (scale.shape.N == 1) dn--;
        else assert(0 && "Error: Inappropriate N-axis size !!");
    if (X.shape.H != scale.shape.H) 
        if (scale.shape.H == 1) dh--;
        else assert(0 && "Error: Inappropriate H-axis size !!");
    if (X.shape.W != scale.shape.W) 
        if (scale.shape.W == 1) dw--;
        else assert(0 && "Error: Inappropriate W-axis size !!");
    if (X.shape.C != scale.shape.C) 
        if (scale.shape.C == 1) dc--;
        else assert(0 && "Error: Inappropriate C-axis size !!");
    
    for (n1 = 0, n2 = 0; n1 < Y.shape.N; n1++, n2 += dn)
    for (h1 = 0, h2 = 0; h1 < Y.shape.H; h1++, h2 += dh)
    for (w1 = 0, w2 = 0; w1 < Y.shape.W; w1++, w2 += dw)
    for (c1 = 0, c2 = 0; c1 < Y.shape.C; c1++, c2 += dc) {
        int32_t div = static_cast<int32_t>(X.get(n1, h1, w1, c1)) - static_cast<int32_t>(zero_point.get(n2, h2, w2, c2));
        Y.at(n1, h1, w1, c1) = static_cast<T_OUT>(div)* scale.get(n2, h2, w2, c2);
    }
}
template <typename T_IN, typename T_OUT>
void DequantizeLinear(TensorMem<T_IN> &X, TensorMem<T_OUT> &Y, T_OUT scale, T_IN zero_point) {
    static_assert(is_integral<T_IN>::value, "Error: Inappropriate type for dequantize !!");
    static_assert(is_float_point<T_OUT>::value, "Error: Inappropriate type for dequantize !!");
    assert(X.shape == Y.shape);

    int size = X.shape.N* X.shape.H* X.shape.W* X.shape.C;
    for (int i = 0; i < size; i++) {
        int32_t div = static_cast<int32_t>(X.raw()[i]) - static_cast<int32_t>(zero_point);
        Y.raw()[i] = static_cast<T_OUT>(div)* scale;
    }
}
