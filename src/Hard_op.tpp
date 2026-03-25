#include "..\include\Core.h"
#include <cassert>


template<int SIZE, typename T>
static inline void load_buffer(T buffer[SIZE], T* data) {
    #pragma HLS INLINE
    for (int i = 0, j = 0; i < SIZE; i++, j++) {
        buffer[i] = data[j];
    }
}
template<int SIZE, typename T>
static inline void store_buffer(T buffer[SIZE], T* data) {
    #pragma HLS INLINE
    for (int i = 0, j = 0; i < SIZE; i++, j++) {
        data[j] = buffer[i];
    }
}
static inline int mod_3(int n) {
    #pragma HLS INLINE
    switch (n) {
        case 0: case 3: case 6: case 9: case 12: case 15: return 0;
        case 1: case 4: case 7: case 10: case 13: case 16: return 1;
        case 2: case 5: case 8: case 11: case 14: case 17: return 2;
    }
    return 0;
}

template<int PEs, int C_IN, int C_OUT, int BATCH, typename T>
void Resblock___Pad_ref_Conv_11133111111_CNorm_Relu___(TensorMem<T> &X, 
                                        TensorMem<T> &W_1, TensorMem<T> &B_1, 
                                        TensorMem<T> &gamma_1, TensorMem<T> &beta_1, 
                                        TensorMem<T> &W_2, TensorMem<T> &B_2, 
                                        TensorMem<T> &gamma_2, TensorMem<T> &beta_2, T epsilon, TensorMem<T> &Y) {
    Shape x_shape = X.shape;
    Shape w_shape = W_1.shape;
    Shape y_shape = Y.shape;

    if (3 != w_shape.H || 3 != w_shape.W) 
        assert(0 && "Error: Unappropriate input sizes for Conv !!\n");
    
    if (y_shape.H != (x_shape.H - (w_shape.H - 1)* 1 
        + 1 + 1 - 1) / 1 + 1 
        || y_shape.W != (x_shape.W - (w_shape.W - 1)* 1 
        + 1 + 1 - 1) / 1 + 1 
        || y_shape.N != x_shape.N || y_shape.C != w_shape.N || C_IN != w_shape.C || C_OUT != w_shape.N || BATCH != x_shape.N)
        assert(0 && "Error: Unappropriate output sizes for Conv !!\n");

    //assert(!(PEs % y_shape.W));
    int width = PEs / y_shape.W;
    int rolls = y_shape.H / width;
    int num = C_OUT;
    T adjustment_scale = (T) num / (num - 1);
    T pre_div = (T) 1 / (T) (sqrtf(num - 1));

    constexpr int vector_depth_size = PEs* C_IN;
    T skip_buffer[C_OUT* 256];
    T x_buffer[3* vector_depth_size];
    T w_buffer[C_IN << 1];
    T b_buffer[C_OUT];
    T y_buffer[PEs][C_OUT];
    T gamma_buffer[C_OUT];
    T beta_buffer[C_OUT];
    #pragma HLS ARRAY_PARTITION variable=x_buffer cyclic factor=4
    #pragma HLS ARRAY_PARTITION variable=w_buffer cyclic factor=4
    #pragma HLS ARRAY_PARTITION variable=b_buffer cyclic factor=4
    #pragma HLS ARRAY_PARTITION variable=y_buffer complete dim=2
    #pragma HLS ARRAY_PARTITION variable=gamma_buffer cyclic factor=4
    #pragma HLS ARRAY_PARTITION variable=beta_buffer cyclic factor=4


    Batch_loop:
    for (int n = 0; n < BATCH; n++) {
        int skip_buf_id = 0;
    Res_block_loop:
    for (int times = 0; times < 2; times++) {
        T* raw_ifm_data = X.raw_at(n, 0, 0, 0);
        load_buffer<vector_depth_size>(&x_buffer[vector_depth_size], raw_ifm_data);
        if (!times) {
            load_buffer<vector_depth_size>(&skip_buffer[skip_buf_id], raw_ifm_data);
            skip_buf_id += vector_depth_size;
            load_buffer<C_OUT>(b_buffer, B_1.raw());
            load_buffer<C_OUT>(gamma_buffer, gamma_1.raw());
            load_buffer<C_OUT>(beta_buffer, beta_1.raw());
        } else {
            load_buffer<C_OUT>(b_buffer, B_2.raw());
            load_buffer<C_OUT>(gamma_buffer, gamma_2.raw());
            load_buffer<C_OUT>(beta_buffer, beta_2.raw());
        }

    Slide_PEs_loop:
        for (int r = 0, load_id = 2; r < rolls; r++, load_id++) {
            if (load_id == 3) load_id = 0;
            if (r < rolls - 1) {
                T* raw_ifm_data_inner = X.raw_at(n, r + width, 0, 0);
                load_buffer<vector_depth_size>(&x_buffer[load_id* vector_depth_size], raw_ifm_data_inner);
                if (!times) {
                    load_buffer<vector_depth_size>(&skip_buffer[skip_buf_id], raw_ifm_data_inner);
                    skip_buf_id += vector_depth_size;
                }
            }

    #pragma HLS UNROLL
    PEs_loop:
        for (int pe = 0; pe < PEs; pe++) {
            int y_h = r* width, y_w = pe, i, j;
            int x_h = -1 + y_h, x_w = -1 + y_w;
            int* w_h, *w_w, _dh, _dw;
            T* x_points_1, *x_points_2;
            if (x_w < 0 || x_w + 2 == x_shape.W) {
                w_h = &j, w_w = &i;
                _dh = 0, _dw = 2;
            } else {
                w_h = &i, w_w = &j;
                _dh = 2, _dw = 0;
            }
    Channel_out_loop:
            for (int co = 0; co < C_OUT; co++) {
                T psum = 0;
                int d_h, d_w;
    Kernel_loop:
                for (i = 0, d_h = _dh, d_w = _dw; i < 2; i++, d_h = 2 - d_h, d_w = 2 - d_w) 
                for (j = 0; j < 3; j++) {
                    int h = x_h + *w_h, w = x_w + *w_w;
                    if (i == 1 && j == 1) {
                        if (co & 1) {
                            T pre_psum = 0;
                            x_points_1 = x_points_2 = &x_buffer[(mod_3(h + 1)* y_shape.W + w)* C_IN];
                            if (!times) {
                                load_buffer<C_IN>(w_buffer, W_1.raw_at(co, i, j, 0));
                                load_buffer<C_IN>(&w_buffer[C_IN], W_1.raw_at(co - 1, i, j, 0));
                            } else {
                                load_buffer<C_IN>(w_buffer, W_2.raw_at(co, i, j, 0));
                                load_buffer<C_IN>(&w_buffer[C_IN], W_2.raw_at(co - 1, i, j, 0));
                            }
                            for (int ci = 0; ci < C_IN; ci++) {
                                psum += x_points_1[ci]* w_buffer[ci];
                                pre_psum += x_points_2[ci]* w_buffer[ci + C_IN];
                            }
                            y_buffer[pe][co - 1] += pre_psum;
                        }
                        break;
                    }
                    char dup = 0;
                    int ops_h = h + d_h, ops_w = w + d_w;
                    int ops_w_h = *w_h + d_h, ops_w_w = *w_w + d_w;
                    if (h < 0) 
                        dup = 1, h = -h;
                    else if (h >= X.shape.H) 
                        dup = 1, h = (X.shape.H << 1) - h - 2;
                    if (w < 0) 
                        dup = 1, w = -w;
                    else if (w >= x_shape.W) 
                        dup = 1, w = (x_shape.W << 1) - w - 2;
                    if (ops_h < 0 || ops_h >= X.shape.H || ops_w < 0 || ops_w >= x_shape.W) dup = 1;


                    x_points_1 = &x_buffer[(mod_3(h + 1)* y_shape.W + w)* C_IN];
                    if (dup) x_points_2 = x_points_1;
                    else x_points_2 = &x_buffer[(mod_3(ops_h + 1)* y_shape.W + ops_w)* C_IN];

                    if (!times) {
                        load_buffer<C_IN>(w_buffer, W_1.raw_at(co, *w_h, *w_w, 0));
                        load_buffer<C_IN>(&w_buffer[C_IN], W_1.raw_at(co, ops_w_h, ops_w_w, 0));
                    } else {
                        load_buffer<C_IN>(w_buffer, W_2.raw_at(co, *w_h, *w_w, 0));
                        load_buffer<C_IN>(&w_buffer[C_IN], W_2.raw_at(co, ops_w_h, ops_w_w, 0));
                    }
    
    Channel_in_loop:
                    for (int ci = 0; ci < C_IN; ci++) {
    #pragma HLS PIPELINE II=1
                        psum += x_points_1[ci]* w_buffer[ci];
                        psum += x_points_2[ci]* w_buffer[ci + C_IN];
                    }
                }
                y_buffer[pe][co] = psum + b_buffer[co];
            }
        }

    Norm_loop:
        for (int non_pe = 0; non_pe < PEs; non_pe++) {
            int y_h = r* width, y_w = non_pe;
            T mean = 0, var = 0;
            for (int co = 0; co < C_OUT; co++) {
    #pragma HLS PIPELINE II=1
                T cur = y_buffer[non_pe][co];
                mean += cur;
                T cur_n = cur* pre_div;
                var += cur_n* cur_n;
            }
            mean /= num;
            var = (T) 1 / static_cast<T>(sqrtf(var - mean* mean* adjustment_scale + epsilon));
            for (int co = 0; co < C_OUT; co++) {
    #pragma HLS PIPELINE II=1
                T weight = gamma_buffer[co]* var;
                T bias = beta_buffer[co] - mean* weight;
                T &point = y_buffer[non_pe][co];
                point = point* weight + bias;
                if (times) point += skip_buffer[(y_h* y_shape.W + y_w)* C_OUT + co];
                else if (point < 0) point = 0;
            }
            if (times) store_buffer<C_OUT>(y_buffer[non_pe], Y.raw_at(n, y_h, y_w, 0));
            else store_buffer<C_OUT>(y_buffer[non_pe], X.raw_at(n, y_h, y_w, 0));
        }
        }
    }
    }
}



/*
template<int PEs, int C_IN, int C_OUT, char RELU, typename T>
void Pad_ref_Conv_11133111111_CNorm_Relu(TensorMem<T> &X, TensorMem<T> &W, TensorMem<T> &B, 
                                        TensorMem<T> &gamma, TensorMem<T> &beta, T epsilon, TensorMem<T> &Y) {
    Shape x_shape = X.shape;
    Shape w_shape = W.shape;
    Shape b_shape = B.shape;
    Shape y_shape = Y.shape;

    if (3 != w_shape.H || 3 != w_shape.W) 
        assert(0 && "Error: Unappropriate input sizes for Conv !!\n");
    
    if (y_shape.H != (x_shape.H - (w_shape.H - 1)* 1 
        + 1 + 1 - 1) / 1 + 1 
        || y_shape.W != (x_shape.W - (w_shape.W - 1)* 1 
        + 1 + 1 - 1) / 1 + 1 
        || y_shape.N != x_shape.N || y_shape.C != w_shape.N || C_IN != w_shape.C || C_OUT != w_shape.N)
        assert(0 && "Error: Unappropriate output sizes for Conv !!\n");

    int start_height = -1, start_width = -1;
    int batch = x_shape.N;
    //b_shape.N = b_shape.H = b_shape.W = 0;
    assert(!(PEs % y_shape.W));
    int width = PEs / y_shape.W;
    int rolls = y_shape.H / width;
    int num = y_shape.C;
    T adjustment_scale = (T) num / (num - 1);
    T pre_div = (T) 1 / (T) (sqrtf(num - 1));


    T x_buffer[C_IN << 1];
    T w_buffer[C_IN << 1];
    T y_buffer[PEs][C_OUT];
    #pragma HLS ARRAY_PARTITION variable=x_buffer cyclic factor=4
    #pragma HLS ARRAY_PARTITION variable=w_buffer cyclic factor=4
    #pragma HLS ARRAY_PARTITION variable=y_buffer complete dim=2

    
    for (int n = 0; n < batch; n++) {
        x_shape.N = y_shape.N = n;
        for (int r = 0; r < rolls; r++) {
    #pragma HLS UNROLL
        for (int pe = 0; pe < PEs; pe++) {
            int y_h = r* width + pe / y_shape.W, y_w = pe % y_shape.W;
            x_shape.H = start_height + y_h, x_shape.W = start_width + y_w;
            for (int co = 0; co < C_OUT; co++) {
                y_shape.C = w_shape.N = co;
                T psum = 0;
                for (int w_h = 0; w_h < 3; w_h++) 
                for (int w_w = 0; w_w < 3; w_w++) {
                    int h = x_shape.H + w_h, w = x_shape.W + w_w;
                    if (w_h == 1 && w_w == 1) {
                        if (co & 1) {
                            T pre_psum = 0;
                            load_buffer<C_IN>(x_buffer, X.raw_at(x_shape.N, h, w, 0));
                            load_buffer<C_IN>(w_buffer, W.raw_at(w_shape.N, w_h, w_w, 0));
                            load_buffer<C_IN>(&w_buffer[C_IN], W.raw_at(w_shape.N - 1, w_h, w_w, 0));
                            for (int ci = 0; ci < C_IN; ci++) {
                                psum += x_buffer[ci]* w_buffer[ci];
                                pre_psum += x_buffer[ci]* w_buffer[ci + C_IN];
                            }
                            y_buffer[pe][y_shape.C - 1] += pre_psum;
                        }
                        w_h++;
                        break;
                    }
                    int ops_h = x_shape.H + 2 - w_h, ops_w = x_shape.W + 2 - w_w;
                    int ops_w_h = 2 - w_h, ops_w_w = 2 - w_w;
                    if (h < 0) 
                        h = -h;
                    else if (h >= X.shape.H) 
                        h = (X.shape.H << 1) - h - 2;
                    if (w < 0) 
                        w = -w;
                    else if (w >= X.shape.W) 
                        w = (X.shape.W << 1) - w - 2;
                    if (ops_h < 0) 
                        ops_h = -ops_h;
                    else if (ops_h >= X.shape.H) 
                        ops_h = (X.shape.H << 1) - ops_h - 2;
                    if (ops_w < 0) 
                        ops_w = -ops_w;
                    else if (ops_w >= X.shape.W) 
                        ops_w = (X.shape.W << 1) - ops_w - 2;

                    load_buffer<C_IN>(x_buffer, X.raw_at(x_shape.N, h, w, 0));
                    load_buffer<C_IN>(&x_buffer[C_IN], X.raw_at(x_shape.N, ops_h, ops_w, 0));

                    load_buffer<C_IN>(w_buffer, W.raw_at(w_shape.N, w_h, w_w, 0));
                    load_buffer<C_IN>(&w_buffer[C_IN], W.raw_at(w_shape.N, ops_w_h, ops_w_w, 0));
                    
                    for (int ci = 0; ci < C_IN << 1; ci++) {
    #pragma HLS PIPELINE II=1
                        psum += x_buffer[ci]* w_buffer[ci];
                    }
                }
                y_buffer[pe][y_shape.C] = psum + B.raw()[co];
            }
        }
        for (int non_pe = 0; non_pe < PEs; non_pe++) {
            int y_h = r* width + non_pe / y_shape.W, y_w = non_pe % y_shape.W;
            T mean = 0, var = 0;
            for (int co = 0; co < C_OUT; co++) {
    #pragma HLS PIPELINE II=1
                T cur = y_buffer[non_pe][co];
                mean += cur;
                T cur_n = cur* pre_div;
                var += cur_n* cur_n;
            }
            mean /= num;
            var = (T) 1 / static_cast<T>(sqrtf(var - mean* mean* adjustment_scale + epsilon));
            for (int co = 0; co < C_OUT; co++) {
    #pragma HLS PIPELINE II=1
                T weight = gamma.raw()[co]* var;
                T bias = beta.raw()[co] - mean* weight;
                T &point = y_buffer[non_pe][co];
                point = point* weight + bias;
                if (RELU && point < 0) point = 0;
            }
            store_buffer<C_OUT>(y_buffer[non_pe], Y.raw_at(y_shape.N, y_h, y_w, 0));
        }
        }
    }
}*/