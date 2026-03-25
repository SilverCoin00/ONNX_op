#include "..\include\Core.h"
#include <cassert>


static char _is_valid_point(Shape &x_shape, int height_p, int width_p) {
    return height_p >= 0 && width_p >= 0 && height_p < x_shape.H && width_p < x_shape.W;
}
template <typename T>
static T _conv_point(const Conv_Attributes &att, TensorMem<T>* X, TensorMem<T>* W, Shape &x_pos, Shape &w_pos) {
    T y_point = 0;
    Shape x_shape = X->shape;
    int w_height = w_pos.H;
    int w_width = w_pos.W;

    int i, j, e, f;
    for (i = e = 0; i < w_height; i++, e += att.dilations[0]) 
        for (j = f = 0; j < w_width; j++, f += att.dilations[1]) {
            int h = x_pos.H + e, w = x_pos.W + f;
            if (_is_valid_point(x_shape, h, w)) 
                y_point += X->get(x_pos.N, h, w, x_pos.C)* W->get(w_pos.N, i, j, w_pos.C);
        }
    return y_point;
}
template <typename T>
static void _conv_channel(const Conv_Attributes &att, TensorMem<T>* X, TensorMem<T>* W, TensorMem<T>* Y, 
                                                        Shape x_pos, Shape &w_pos, Shape &y_pos) {
    int start_height = -att.pads[0], start_width = -att.pads[1];
    int y_height = y_pos.H;
    int y_width = y_pos.W;
    int i, j, e, f;
    for (i = e = 0; i < y_height; i++, e += att.strides[0]) 
        for (j = f = 0; j < y_width; j++, f += att.strides[1]) {
            x_pos.H = start_height + e, x_pos.W = start_width + f;
            Y->at(y_pos.N, i, j, y_pos.C) += _conv_point(att, X, W, x_pos, w_pos);
        }
}
template <typename T>
static void _conv_plus_bias(TensorMem<T>* XW, TensorMem<T>* B, Shape &xw_pos, Shape &b_pos) {
    int xw_height = xw_pos.H;
    int xw_width = xw_pos.W;
    int i, j;
    for (i = 0; i < xw_height; i++) 
        for (j = 0; j < xw_width; j++) 
            XW->at(xw_pos.N, i, j, xw_pos.C) += B->get(b_pos.N, b_pos.H, b_pos.W, b_pos.C);
}

template <typename T>
TensorMem<T>* Conv(const Conv_Attributes &attributes, TensorMem<T>* X, TensorMem<T>* W, TensorMem<T>* B) {
    assert(X && W && B);
    Shape x_shape = X->shape;
    Shape w_shape = W->shape;
    Shape b_shape = B->shape;

    if (x_shape.C % attributes.group || w_shape.N % attributes.group || x_shape.C / attributes.group != w_shape.C
        || attributes.kernel_shape[0] != w_shape.H || attributes.kernel_shape[1] != w_shape.W) 
        assert(0 && "Error: Unappropriate input sizes for conv !!\n");
    
    Shape y_shape;
    y_shape.H = (x_shape.H - (w_shape.H - 1)* attributes.dilations[0] 
                + attributes.pads[0] + attributes.pads[2] - 1) / attributes.strides[0] + 1;
    y_shape.W = (x_shape.W - (w_shape.W - 1)* attributes.dilations[1] 
                + attributes.pads[1] + attributes.pads[3] - 1) / attributes.strides[1] + 1;
    y_shape.N = x_shape.N;
    y_shape.C = w_shape.N;
    TensorMem<T>* Y = new TensorMem<T>(y_shape);

    int i, j, l, m, n;
    int batch = x_shape.N, C_in = w_shape.C, C_out = w_shape.N / attributes.group;
    b_shape.N = b_shape.H = b_shape.W = 0;
    for (i = 0; i < batch; i++) {
        x_shape.N = y_shape.N = i;
        for (j = 0; j < attributes.group; j++) 
            for (l = 0; l < C_out; l++) {
                y_shape.C = w_shape.N = b_shape.C = j* C_out + l;
                for (m = 0; m < C_in; m++) {
                    x_shape.C = j* C_in + m;
                    w_shape.C = m;
                    _conv_channel(attributes, X, W, Y, x_shape, w_shape, y_shape);
                }
                _conv_plus_bias(Y, B, y_shape, b_shape);
            }
    }
    return Y;
}
template <typename T>
void Conv(const Conv_Attributes &attributes, TensorMem<T>* X, TensorMem<T>* W, TensorMem<T>* B, TensorMem<T>* Y) {
    assert(X && W && B);
    Shape x_shape = X->shape;
    Shape w_shape = W->shape;
    Shape b_shape = B->shape;
    Shape y_shape = Y->shape;

    if (x_shape.C % attributes.group || w_shape.N % attributes.group || x_shape.C / attributes.group != w_shape.C
        || attributes.kernel_shape[0] != w_shape.H || attributes.kernel_shape[1] != w_shape.W) 
        assert(0 && "Error: Unappropriate input sizes for conv !!\n");
    
    if (y_shape.H != (x_shape.H - (w_shape.H - 1)* attributes.dilations[0] 
        + attributes.pads[0] + attributes.pads[2] - 1) / attributes.strides[0] + 1 
        || y_shape.W != (x_shape.W - (w_shape.W - 1)* attributes.dilations[1] 
        + attributes.pads[1] + attributes.pads[3] - 1) / attributes.strides[1] + 1 
        || y_shape.N != x_shape.N || y_shape.C != w_shape.N)
        assert(0 && "Error: Unappropriate output sizes for conv !!\n");

    memset(Y->raw(), 0, y_shape.N* y_shape.H* y_shape.W* y_shape.C* sizeof(T));
    int i, j, l, m, n;
    int batch = x_shape.N, C_in = w_shape.C, C_out = w_shape.N / attributes.group;
    b_shape.N = b_shape.H = b_shape.W = 0;
    for (i = 0; i < batch; i++) {
        x_shape.N = y_shape.N = i;
        for (j = 0; j < attributes.group; j++) 
            for (l = 0; l < C_out; l++) {
                y_shape.C = w_shape.N = b_shape.C = j* C_out + l;
                for (m = 0; m < C_in; m++) {
                    x_shape.C = j* C_in + m;
                    w_shape.C = m;
                    _conv_channel(attributes, X, W, Y, x_shape, w_shape, y_shape);
                }
                _conv_plus_bias(Y, B, y_shape, b_shape);
            }
    }
}
