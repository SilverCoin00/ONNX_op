#include "..\include\Core.h"


static int32_t _mul_quantize_multiplier(int32_t x, int32_t M, int shift) {
    int64_t prod = static_cast<int64_t>(x)* static_cast<int64_t>(M);
    if (shift > 0) 
        // round(a / d) = (a + d/2) / d
        if (prod >= 0) prod = (prod + (1LL << (shift - 1))) >> shift;
        else prod = - ((-prod + (1LL << (shift - 1))) >> shift);
    else if (shift < 0) 
        prod <<= (-shift);
    return prod;
}
void cast_scale_to_M_pow_2_n(float scale, int32_t &M, int &n) {
    float f;
    int e;
    f = frexpf(scale, &e);
    M = static_cast<int32_t>(roundf(f* (1LL << 31)));
    if (M == (1LL << 31)) {
        M >>= 1;
        n--;
    }
    n = 31 - e;
}

template<typename T>
void QLinearAdd(TensorMem<T> &X1, TensorMem<T> &X2, TensorMem<T> &Y, 
                //float scale_1, float scale_2, float scale_y, 
                int32_t M1, int N1, int32_t M2, int N2, 
                T zero_point_1, T zero_point_2, T zero_point_y) {
    static_assert(is_integral<T>::value, "Error: Inappropriate type for QLAdd !!");
    assert(X1.shape == Y.shape || X2.shape == Y.shape);
    int dn1, dn2, dh1, dh2, dw1, dw2, dc1, dc2;
    dn1 = dn2 = dh1 = dh2 = dw1 = dw2 = dc1 = dc2 = 1;
    int n1, n2, h1, h2, w1, w2, c1, c2;
    int* yn = &n1, *yh = &h1, *yw = &w1, *yc = &c1;
    if (X1.shape.N != X2.shape.N) {
        if (X1.shape.N == 1) dn1--, yn = &n2;
        else if (X2.shape.N == 1) dn2--;
        else assert(0 && "Error: Inappropriate N-axis size !!");
    }
    if (X1.shape.H != X2.shape.H) {
        if (X1.shape.H == 1) dh1--, yh = &h2;
        else if (X2.shape.H == 1) dh2--;
        else assert(0 && "Error: Inappropriate H-axis size !!");
    }
    if (X1.shape.W != X2.shape.W) {
        if (X1.shape.W == 1) dw1--, yw = &w2;
        else if (X2.shape.W == 1) dw2--;
        else assert(0 && "Error: Inappropriate W-axis size !!");
    }
    if (X1.shape.C != X2.shape.C) {
        if (X1.shape.C == 1) dc1--, yc = &c2;
        else if (X2.shape.C == 1) dc2--;
        else assert(0 && "Error: Inappropriate C-axis size !!");
    }

    // scale_1 /= scale_y;
    // scale_2 /= scale_y;
    // int32_t M1, M2;
    // int N1, N2;
    // cast_scale_to_M_pow_2_n(scale_1, M1, N1);
    // cast_scale_to_M_pow_2_n(scale_2, M2, N2);

    for (n1 = 0, n2 = 0; n1 < Y.shape.N && n2 < Y.shape.N; n1 += dn1, n2 += dn2)
    for (h1 = 0, h2 = 0; h1 < Y.shape.H && h2 < Y.shape.H; h1 += dh1, h2 += dh2)
    for (w1 = 0, w2 = 0; w1 < Y.shape.W && w2 < Y.shape.W; w1 += dw1, w2 += dw2)
    for (c1 = 0, c2 = 0; c1 < Y.shape.C && c2 < Y.shape.C; c1 += dc1, c2 += dc2) {
        int32_t x1 = static_cast<int32_t>(static_cast<int16_t>(X1.get(n1, h1, w1, c1)) - static_cast<int16_t>(zero_point_1));
        int32_t x2 = static_cast<int32_t>(static_cast<int16_t>(X2.get(n2, h2, w2, c2)) - static_cast<int16_t>(zero_point_2));
        x1 = _mul_quantize_multiplier(x1, M1, N1);
        x2 = _mul_quantize_multiplier(x2, M2, N2);
        Y.at(*yn, *yh, *yw, *yc) = clamp<T>(static_cast<int32_t>(zero_point_y) + x1 + x2);
    }
}

template<typename T>
void QLinearMul(TensorMem<T> &X1, TensorMem<T> &X2, TensorMem<T> &Y, 
                //float scale_1, float scale_2, float scale_y, 
                int32_t M, int N, 
                T zero_point_1, T zero_point_2, T zero_point_y) {
    static_assert(is_integral<T>::value, "Error: Inappropriate type for QLMul !!");
    assert(X1.shape == Y.shape || X2.shape == Y.shape);
    int dn1, dn2, dh1, dh2, dw1, dw2, dc1, dc2;
    dn1 = dn2 = dh1 = dh2 = dw1 = dw2 = dc1 = dc2 = 1;
    int n1, n2, h1, h2, w1, w2, c1, c2;
    int* yn = &n1, *yh = &h1, *yw = &w1, *yc = &c1;
    if (X1.shape.N != X2.shape.N) {
        if (X1.shape.N == 1) dn1--, yn = &n2;
        else if (X2.shape.N == 1) dn2--;
        else assert(0 && "Error: Inappropriate N-axis size !!");
    }
    if (X1.shape.H != X2.shape.H) {
        if (X1.shape.H == 1) dh1--, yh = &h2;
        else if (X2.shape.H == 1) dh2--;
        else assert(0 && "Error: Inappropriate H-axis size !!");
    }
    if (X1.shape.W != X2.shape.W) {
        if (X1.shape.W == 1) dw1--, yw = &w2;
        else if (X2.shape.W == 1) dw2--;
        else assert(0 && "Error: Inappropriate W-axis size !!");
    }
    if (X1.shape.C != X2.shape.C) {
        if (X1.shape.C == 1) dc1--, yc = &c2;
        else if (X2.shape.C == 1) dc2--;
        else assert(0 && "Error: Inappropriate C-axis size !!");
    }

    // scale_y = scale_1* scale_2 / scale_y;
    // int32_t M;
    // int N;
    // cast_scale_to_M_pow_2_n(scale_y, M, N);

    for (n1 = 0, n2 = 0; n1 < Y.shape.N && n2 < Y.shape.N; n1 += dn1, n2 += dn2)
    for (h1 = 0, h2 = 0; h1 < Y.shape.H && h2 < Y.shape.H; h1 += dh1, h2 += dh2)
    for (w1 = 0, w2 = 0; w1 < Y.shape.W && w2 < Y.shape.W; w1 += dw1, w2 += dw2)
    for (c1 = 0, c2 = 0; c1 < Y.shape.C && c2 < Y.shape.C; c1 += dc1, c2 += dc2) {
        int32_t x = static_cast<int32_t>(static_cast<int16_t>(X1.get(n1, h1, w1, c1)) - static_cast<int16_t>(zero_point_1))
                    * static_cast<int32_t>(static_cast<int16_t>(X2.get(n2, h2, w2, c2)) - static_cast<int16_t>(zero_point_2));
        x = _mul_quantize_multiplier(x, M, N);
        Y.at(*yn, *yh, *yw, *yc) = clamp<T>(static_cast<int32_t>(zero_point_y) + x);
    }
}

static char _is_valid_pointq(Shape &x_shape, int height_p, int width_p) {
    return height_p >= 0 && width_p >= 0 && height_p < x_shape.H && width_p < x_shape.W;
}
template<typename T_IN, typename T_W>
static int32_t _conv_pointq(const Conv_Attributes &att, TensorMem<T_IN>* X, TensorMem<T_W>* W, 
                            T_IN x_zero_point, TensorMem<T_W>* w_zero_point, Shape &x_pos, Shape &w_pos) {
    int32_t y_point = 0;
    Shape x_shape = X->shape;
    int w_height = w_pos.H;
    int w_width = w_pos.W;

    int i, j, e, f;
    if (!x_zero_point) 
        for (i = e = 0; i < w_height; i++, e += att.dilations[0]) 
            for (j = f = 0; j < w_width; j++, f += att.dilations[1]) {
                int h = x_pos.H + e, w = x_pos.W + f;
                if (_is_valid_pointq(x_shape, h, w)) 
                    y_point += static_cast<int32_t>(static_cast<int16_t>(X->get(x_pos.N, h, w, x_pos.C))
                            * (static_cast<int16_t>(W->get(w_pos.N, i, j, w_pos.C)) 
                            - static_cast<int16_t>(w_zero_point->get(0, 0, 0, w_pos.C))));
            }
    else 
        for (i = e = 0; i < w_height; i++, e += att.dilations[0]) 
            for (j = f = 0; j < w_width; j++, f += att.dilations[1]) {
                int h = x_pos.H + e, w = x_pos.W + f;
                if (_is_valid_pointq(x_shape, h, w)) 
                    y_point += static_cast<int32_t>(static_cast<int16_t>(X->get(x_pos.N, h, w, x_pos.C) 
                            - static_cast<int16_t>(x_zero_point))
                            * (static_cast<int16_t>(W->get(w_pos.N, i, j, w_pos.C)) 
                            - static_cast<int16_t>(w_zero_point->get(0, 0, 0, w_pos.C))));
            }
    return y_point;
}
template<typename T_IN, typename T_W, typename T_OUT>
void QLinearConv(const Conv_Attributes &attributes, TensorMem<T_IN>* X, TensorMem<T_W>* W, TensorMem<int32_t>* B, TensorMem<T_OUT>* Y, 
                    TensorMem<int32_t>* M, TensorMem<int>* N, T_IN x_zero_point, TensorMem<T_W>* w_zero_point, T_OUT y_zero_point) {
    static_assert(is_integral<T_IN>::value, "Error: Inappropriate type for QLConv !!");
    static_assert(is_integral<T_W>::value, "Error: Inappropriate type for QLConv !!");
    static_assert(is_integral<T_OUT>::value, "Error: Inappropriate type for QLConv !!");
    assert(X && W && B);
    Shape x_shape = X->shape;
    Shape w_shape = W->shape;
    Shape b_shape = B->shape;
    Shape y_shape = Y->shape;

    if (x_shape.C % attributes.group || w_shape.N % attributes.group || x_shape.C / attributes.group != w_shape.C
        || attributes.kernel_shape[0] != w_shape.H || attributes.kernel_shape[1] != w_shape.W) 
        assert(0 && "Error: Unappropriate input sizes for QLConv !!\n");
    
    if (y_shape.H != (x_shape.H - (w_shape.H - 1)* attributes.dilations[0] 
        + attributes.pads[0] + attributes.pads[2] - 1) / attributes.strides[0] + 1 
        || y_shape.W != (x_shape.W - (w_shape.W - 1)* attributes.dilations[1] 
        + attributes.pads[1] + attributes.pads[3] - 1) / attributes.strides[1] + 1 
        || y_shape.N != x_shape.N || y_shape.C != w_shape.N)
        assert(0 && "Error: Unappropriate output sizes for QLConv !!\n");

    int i, j, l, m, n;
    int start_height = -attributes.pads[0], start_width = -attributes.pads[1];
    int batch = x_shape.N, C_in = w_shape.C, C_out = w_shape.N / attributes.group;
    b_shape.N = b_shape.H = b_shape.W = 0;
    for (i = 0; i < batch; i++) {
        x_shape.N = y_shape.N = i;
        for (j = 0; j < attributes.group; j++) 
            for (l = 0; l < C_out; l++) {
                y_shape.C = w_shape.N = b_shape.C = j* C_out + l;
                int k, h, e, f;
                for (k = e = 0; k < y_shape.H; k++, e += attributes.strides[0]) 
                    for (h = f = 0; h < y_shape.W; h++, f += attributes.strides[1]) {
                        int32_t accum = 0;
                        x_shape.H = start_height + e, x_shape.W = start_width + f;
                        for (m = 0; m < C_in; m++) {
                            x_shape.C = j* C_in + m;
                            w_shape.C = m;
                            accum += _conv_pointq<T_IN, T_W>(attributes, X, W, x_zero_point, w_zero_point, x_shape, w_shape);
                        }
                        accum += B->get(b_shape.N, b_shape.H, b_shape.W, b_shape.C);
                        accum = _mul_quantize_multiplier(accum, 
                                M->get(b_shape.N, b_shape.H, b_shape.W, b_shape.C), 
                                N->get(b_shape.N, b_shape.H, b_shape.W, b_shape.C));
                        Y->at(y_shape.N, k, h, y_shape.C) 
                            = clamp<T_OUT>(static_cast<int32_t>(y_zero_point) + accum);
                    }
            }
    }
}

// template <typename T>
// static void _convtranspose_pointq(const ConvTranspose_Attributes &att, T X, TensorMem<T>* W, TensorMem<T>* Y, 
//                                                                             Shape &w_pos, Shape &y_pos) {
//     Shape y_shape = Y->shape;
//     int w_height = w_pos.H;
//     int w_width = w_pos.W;

//     int i, j, e, f;
//     for (i = e = 0; i < w_height; i++, e += att.dilations[0]) 
//         for (j = f = 0; j < w_width; j++, f += att.dilations[1]) {
//             int h = y_pos.H + e, w = y_pos.W + f;
//             if (_is_valid_pointq(y_shape, h, w)) 
//                 Y->at(y_pos.N, h, w, y_pos.C) += X* W->get(w_pos.N, i, j, w_pos.C);
//         }
// }

// template<typename T_IN, typename T_W, typename T_OUT>
// void QLinearConvTranspose(const ConvTranspose_Attributes &attributes, TensorMem<T_IN>* X, TensorMem<T_W>* W, TensorMem<int32_t>* B, TensorMem<T_OUT>* Y, 
//                                 TensorMem<int32_t>* M, TensorMem<int>* N, T_IN x_zero_point, TensorMem<T_W>* w_zero_point, T_OUT y_zero_point) {
//     assert(X && W && B);
//     Shape x_shape = X->shape;
//     Shape w_shape = W->shape;
//     Shape b_shape = B->shape;
//     Shape y_shape = Y->shape;

//     if (x_shape.C % attributes.group || w_shape.N % attributes.group || x_shape.C / attributes.group != w_shape.C
//         || attributes.kernel_shape[0] != w_shape.H || attributes.kernel_shape[1] != w_shape.W 
//         || attributes.output_padding[0] >= attributes.dilations[0] && attributes.output_padding[0] >= attributes.strides[0] 
//         || attributes.output_padding[1] >= attributes.dilations[1] && attributes.output_padding[1] >= attributes.strides[1]) 
//         assert(0 && "Error: Unappropriate input sizes for QLConvTranspose !!\n");

//     if (y_shape.H != (x_shape.H - 1)* attributes.strides[0] + (w_shape.H - 1)* attributes.dilations[0]
//                 - attributes.pads[0] - attributes.pads[2] + attributes.output_padding[0] + 1 
//         || y_shape.W != (x_shape.W - 1)* attributes.strides[1] + (w_shape.W - 1)* attributes.dilations[1]
//                 - attributes.pads[1] - attributes.pads[3] + attributes.output_padding[1] + 1 
//         || y_shape.C != w_shape.N || y_shape.N != x_shape.N) 
//         assert(0 && "Error: Unappropriate output sizes for QLConvTranspose !!\n");

//     int i, j, l, m, n;
//     int start_height = -attributes.pads[0], start_width = -attributes.pads[1];
//     int batch = x_shape.N, C_in = w_shape.C, C_out = w_shape.N / attributes.group;
//     b_shape.N = b_shape.H = b_shape.W = 0;
//     for (i = 0; i < batch; i++) {
//         x_shape.N = y_shape.N = i;
//         for (j = 0; j < attributes.group; j++) 
//             for (l = 0; l < C_out; l++) {
//                 y_shape.C = w_shape.N = b_shape.C = j* C_out + l;
//                 int k, h, e, f;
//                 for (k = e = 0; k < x_shape.H; k++, e += attributes.strides[0]) 
//                     for (h = f = 0; h < x_shape.W; h++, f += attributes.strides[1]) {
//                         y_shape.H = start_height + e, y_shape.W = start_width + f;
//                         for (m = 0; m < C_in; m++) {
//                             x_shape.C = j* C_in + m;
//                             w_shape.C = m;
//                             _convtranspose_point(attributes, X->get(x_shape.N, k, h, x_shape.C), W, Y, w_shape, y_shape);
//                         }
//                         Y->at(y_shape.N, i, j, y_shape.C) += B->get(b_shape.N, b_shape.H, b_shape.W, b_shape.C);
//                     }
//             }
//     }
// }

// void QLinearNorm_hybrid_1() {
    
// }