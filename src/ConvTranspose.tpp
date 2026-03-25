#include "..\include\Core.h"
#include <cassert>


static char _is_valid_pointt(Shape &x_shape, int height_p, int width_p) {
    return height_p >= 0 && width_p >= 0 && height_p < x_shape.H && width_p < x_shape.W;
}
template <typename T>
static void _convtranspose_point(const ConvTranspose_Attributes &att, T X, TensorMem<T>* W, TensorMem<T>* Y, 
                                                                            Shape &w_pos, Shape &y_pos) {
    Shape y_shape = Y->shape;
    int w_height = w_pos.H;
    int w_width = w_pos.W;

    int i, j, e, f;
    for (i = e = 0; i < w_height; i++, e += att.dilations[0]) 
        for (j = f = 0; j < w_width; j++, f += att.dilations[1]) {
            int h = y_pos.H + e, w = y_pos.W + f;
            if (_is_valid_pointt(y_shape, h, w)) 
                Y->at(y_pos.N, h, w, y_pos.C) += X* W->get(w_pos.N, i, j, w_pos.C);
        }
}
template <typename T>
static void _convtranspose_channel(const ConvTranspose_Attributes &att, TensorMem<T>* X, TensorMem<T>* W, TensorMem<T>* Y, 
                                                                    Shape &x_pos, Shape &w_pos, Shape y_pos) {
    int start_height = -att.pads[0], start_width = -att.pads[1];
    int x_height = x_pos.H;
    int x_width = x_pos.W;
    int i, j, e, f;
    for (i = e = 0; i < x_height; i++, e += att.strides[0]) 
        for (j = f = 0; j < x_width; j++, f += att.strides[1]) {
            y_pos.H = start_height + e, y_pos.W = start_width + f;
            _convtranspose_point(att, X->get(x_pos.N, i, j, x_pos.C), W, Y, w_pos, y_pos);
        }
}
template <typename T>
static void _convtranspose_plus_bias(TensorMem<T>* XW, TensorMem<T>* B, Shape &xw_pos, Shape &b_pos) {
    int xw_height = xw_pos.H;
    int xw_width = xw_pos.W;
    int i, j;
    for (i = 0; i < xw_height; i++) 
        for (j = 0; j < xw_width; j++) 
            XW->at(xw_pos.N, i, j, xw_pos.C) += B->get(b_pos.N, b_pos.H, b_pos.W, b_pos.C);
}

template <typename T>
TensorMem<T>* ConvTranspose(const ConvTranspose_Attributes &attributes, TensorMem<T>* X, TensorMem<T>* W, TensorMem<T>* B) {
    assert(X && W && B);
    Shape x_shape = X->shape;
    Shape w_shape = W->shape;
    Shape b_shape = B->shape;

    if (x_shape.C % attributes.group || w_shape.N % attributes.group || x_shape.C / attributes.group != w_shape.C
        || attributes.kernel_shape[0] != w_shape.H || attributes.kernel_shape[1] != w_shape.W 
        || attributes.output_padding[0] >= attributes.dilations[0] && attributes.output_padding[0] >= attributes.strides[0] 
        || attributes.output_padding[1] >= attributes.dilations[1] && attributes.output_padding[1] >= attributes.strides[1]) 
        assert(0 && "Error: Unappropriate input sizes for convtranspose !!\n");

    Shape y_shape;
    y_shape.H = (x_shape.H - 1)* attributes.strides[0] + (w_shape.H - 1)* attributes.dilations[0]
                - attributes.pads[0] - attributes.pads[2] + attributes.output_padding[0] + 1;
    y_shape.W = (x_shape.W - 1)* attributes.strides[1] + (w_shape.W - 1)* attributes.dilations[1]
                - attributes.pads[1] - attributes.pads[3] + attributes.output_padding[1] + 1;
    y_shape.C = w_shape.N;
    y_shape.N = x_shape.N;
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
                    _convtranspose_channel(attributes, X, W, Y, x_shape, w_shape, y_shape);
                }
                _convtranspose_plus_bias(Y, B, y_shape, b_shape);
            }
    }
    return Y;
}
template <typename T>
void ConvTranspose(const ConvTranspose_Attributes &attributes, TensorMem<T>* X, TensorMem<T>* W, TensorMem<T>* B, TensorMem<T>* Y) {
    assert(X && W && B);
    Shape x_shape = X->shape;
    Shape w_shape = W->shape;
    Shape b_shape = B->shape;
    Shape y_shape = Y->shape;

    if (x_shape.C % attributes.group || w_shape.N % attributes.group || x_shape.C / attributes.group != w_shape.C
        || attributes.kernel_shape[0] != w_shape.H || attributes.kernel_shape[1] != w_shape.W 
        || attributes.output_padding[0] >= attributes.dilations[0] && attributes.output_padding[0] >= attributes.strides[0] 
        || attributes.output_padding[1] >= attributes.dilations[1] && attributes.output_padding[1] >= attributes.strides[1]) 
        assert(0 && "Error: Unappropriate input sizes for convtranspose !!\n");
    
    if (y_shape.H != (x_shape.H - 1)* attributes.strides[0] + (w_shape.H - 1)* attributes.dilations[0]
                - attributes.pads[0] - attributes.pads[2] + attributes.output_padding[0] + 1 
        || y_shape.W != (x_shape.W - 1)* attributes.strides[1] + (w_shape.W - 1)* attributes.dilations[1]
                - attributes.pads[1] - attributes.pads[3] + attributes.output_padding[1] + 1 
        || y_shape.C != w_shape.N || y_shape.N != x_shape.N)
        assert(0 && "Error: Unappropriate output sizes for convtranspose !!\n");

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
                    _convtranspose_channel(attributes, X, W, Y, x_shape, w_shape, y_shape);
                }
                _convtranspose_plus_bias(Y, B, y_shape, b_shape);
            }
    }
}
template <typename T>
void ConvTranspose_2(const ConvTranspose_Attributes &attributes, TensorMem<T>* X, TensorMem<T>* W, TensorMem<T>* B, TensorMem<T>* Y) {
    assert(X && W && B);
    Shape x_shape = X->shape;
    Shape w_shape = W->shape;
    Shape b_shape = B->shape;
    Shape y_shape = Y->shape;

    if (x_shape.C % attributes.group || w_shape.N % attributes.group || x_shape.C / attributes.group != w_shape.C
        || attributes.kernel_shape[0] != w_shape.H || attributes.kernel_shape[1] != w_shape.W 
        || attributes.output_padding[0] >= attributes.dilations[0] && attributes.output_padding[0] >= attributes.strides[0] 
        || attributes.output_padding[1] >= attributes.dilations[1] && attributes.output_padding[1] >= attributes.strides[1]
        || attributes.group != 1) 
        assert(0 && "Error: Unappropriate input sizes for convtranspose !!\n");
    
    if (y_shape.H != (x_shape.H - 1)* attributes.strides[0] + (w_shape.H - 1)* attributes.dilations[0]
                - attributes.pads[0] - attributes.pads[2] + attributes.output_padding[0] + 1 
        || y_shape.W != (x_shape.W - 1)* attributes.strides[1] + (w_shape.W - 1)* attributes.dilations[1]
                - attributes.pads[1] - attributes.pads[3] + attributes.output_padding[1] + 1 
        || y_shape.C != w_shape.N || y_shape.N != x_shape.N)
        assert(0 && "Error: Unappropriate output sizes for convtranspose !!\n");

    int batch = x_shape.N, C_IN = w_shape.C, C_OUT = w_shape.N;
    int H_IN = x_shape.H, W_IN = x_shape.W;
    int H_R = w_shape.H, W_R = w_shape.W;
    int H_OUT = y_shape.H, W_OUT = y_shape.W;
    int start_height = -attributes.pads[0], start_width = -attributes.pads[1];
    int end_height = H_OUT - attributes.output_padding[0], end_width = W_OUT - attributes.output_padding[1];

    for (int n = 0; n < batch; n++) {
        for (int co = 0; co < C_OUT; co++) {
            for (int ci = 0; ci < C_IN; ci++) {
                for (int hi = 0, i = 0; hi < H_IN; hi++, i += attributes.strides[0]) 
                for (int wi = 0, j = 0; wi < W_IN; wi++, j += attributes.strides[1]) {
                    int start_wind_h = start_height + i, start_wind_w = start_width + j;

                    for (int hf = 0, e = 0; hf < H_R; hf++, e += attributes.dilations[0]) 
                    for (int wf = 0, f = 0; wf < W_R; wf++, f += attributes.dilations[1]) {
                        int h = start_wind_h + e, w = start_wind_w + f;
                        if (_is_valid_pointt(y_shape, h, w) )//&& h < end_height && w < end_width) 
                            Y->at(n, h, w, co) += X->get(n, hi, wi, ci)* W->get(co, hf, wf, ci);
                    }
                }
            }
            for (int ho = 0; ho < H_OUT; ho++) 
            for (int wo = 0; wo < W_OUT; wo++) 
                Y->at(n, ho, wo, co) += B->get(0, 0, 0, co);
        }
    }
}