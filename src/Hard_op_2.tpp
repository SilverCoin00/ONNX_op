#include "..\include\Core.h"
#include <cassert>



template<int SIZE, typename T>
static inline void load_buffer(T buffer[SIZE], T* data) {
    #pragma HLS INLINE
    for (int i = 0; i < SIZE; i++) {
        buffer[i] = data[i];
    }
}
template<int SIZE, typename T>
static inline void store_buffer(T buffer[SIZE], T* data) {
    #pragma HLS INLINE
    for (int i = 0; i < SIZE; i++) {
        data[i] = buffer[i];
    }
}

template <int PEs, int BATCH, int C_IN, int C_OUT, 
            int H_IN, int W_IN, int H_R, int W_R, int H_OUT, int W_OUT, typename T>
void ConvTranspose_hw(TensorMem<T>* X, TensorMem<T>* W, TensorMem<T>* B, TensorMem<T>* Y) {
    assert(X && W && B);
    Shape x_shape = X->shape;
    Shape w_shape = W->shape;
    Shape b_shape = B->shape;
    Shape y_shape = Y->shape;

    int start_height = -1, start_width = -1;
    constexpr int TC_OUT = C_OUT / PEs;

    T x_buffer[C_IN];
    #pragma HLS ARRAY_PARTITION variable=x_buffer type=complete
    T w_buffer[PEs][H_R][W_R][C_IN];
    #pragma HLS ARRAY_PARTITION variable=w_buffer type=complete dim=1
    #pragma HLS ARRAY_PARTITION variable=w_buffer type=complete dim=2
    #pragma HLS ARRAY_PARTITION variable=w_buffer type=complete dim=3
    T b_buffer[C_OUT];
    #pragma HLS ARRAY_PARTITION variable=b_buffer type=complete
    T y_buffer[H_R][W_R][C_OUT];
    #pragma HLS ARRAY_PARTITION variable=y_buffer type=complete

    T col_buff[C_OUT][H_R][W_R];
    T row_buff[W_IN][C_OUT][H_R][W_R];
    #pragma HLS ARRAY_PARTITION variable=col_buff type=block factor=PEs dim=1
    #pragma HLS ARRAY_PARTITION variable=row_buff type=block factor=PEs dim=2


    load_buffer<C_OUT>(b_buffer, B->raw_at(0, 0, 0, 0));
    Batch_loop:
    for (int n = 0; n < BATCH; n++) {

        Layer_in_loop:
        for (int hi = 0, i = 0; hi < H_IN; hi++, i += 2) 
        for (int wi = 0, j = 0; wi < W_IN; wi++, j += 2) {
            int start_wind_h = start_height + i, start_wind_w = start_width + j;
            load_buffer<C_IN>(x_buffer, X->raw_at(n, hi, wi, 0));

            PEs_loop:
            for (int pe = 0; pe < PEs; pe++) {
                #pragma HLS UNROLL

                T patch_buff[H_R][W_R];
                T out_col[H_R][W_R];
                T out_row[H_R][W_R];

                #pragma HLS ARRAY_PARTITION variable=patch_buff type=complete dim=0
                #pragma HLS ARRAY_PARTITION variable=out_col type=complete dim=0
                #pragma HLS ARRAY_PARTITION variable=out_row type=complete dim=0


                Partial_channel_out_loop:
                for (int tco = 0; tco < TC_OUT; tco++) {
                    int co = pe* TC_OUT + tco;
                    load_buffer<H_R* W_R* C_IN>(w_buffer[pe], W->raw_at(co, 0, 0, 0));

                    // Reset patch_buff
                    for (int hf = 0; hf < H_R; hf++) {
                        #pragma HLS UNROLL
                        for (int wf = 0; wf < W_R; wf++) {
                            #pragma HLS UNROLL
                            patch_buff[hf][wf] = 0;
                        }
                    }

                    for (int ci = 0; ci < C_IN; ci++) {
                        #pragma HLS PIPELINE II=1
                        for (int hf = 0; hf < H_R; hf++) {
                            #pragma HLS UNROLL
                            for (int wf = 0; wf < W_R; wf++) {
                                #pragma HLS UNROLL
                                patch_buff[hf][wf] += x_buffer[ci]* w_buffer[hf][wf][ci];
                            }
                        }
                    }

                    for (int hf = 0; hf < H_R; hf++) {
                        #pragma HLS UNROLL
                        for (int wf = 0; wf < W_R; wf++) {
                            #pragma HLS UNROLL
                            if (wf < W_R - 2) 
                                if (wi == 0) 
                                    out_col[hf][wf] = patch_buff[hf][wf];
                                else 
                                    out_col[hf][wf] = patch_buff[hf][wf] + col_buff[co][hf][wf];
                            else 
                                out_col[hf][wf] = patch_buff[hf][wf];

                            if (wf >= 2) 
                                col_buff[co][hf][wf - 2] = out_col[hf][wf];

                            if (hf < H_R - 2) 
                                if (hi == 0) 
                                    out_row[hf][wf] = out_col[hf][wf];
                                else 
                                    out_row[hf][wf] = out_col[hf][wf] + row_buff[wi][co][hf][wf];
                            else 
                                out_row[hf][wf] = out_col[hf][wf];

                            if (hf >= 2) 
                                row_buff[wi][co][hf - 2][wf] = out_row[hf][wf];

                            if ((hf < 2 || hi == H_IN - 1) && (wf < 2 || wi == W_IN - 1)) 
                                y_buffer[hf][wf][co] = out_row[hf][wf] + b_buffer[co];
                        }
                    }
                }  // end tco
            }  // end pe

            Partial_output_window_loop:
            for (int hf = 0; hf < H_R; hf++) 
            for (int wf = 0; wf < W_R; wf++) {
                #pragma HLS PIPELINE II=1

                int h = start_wind_h + hf, w = start_wind_w + wf;
                if (h >= 0 && w >= 0 && h < H_OUT && w < W_OUT 
                        && (hf < 2 || hi == H_IN - 1) && (wf < 2 || wi == W_IN - 1)) 
                    store_buffer<C_OUT>(y_buffer[hf][wf], Y->raw_at(n, h, w, 0));
            }
        }  // end hi-wi
    }  // end batch
}


template<int SIZE, typename T>
static inline void shift_load_buffer(T buffer[2][SIZE], T* data) {
    #pragma HLS INLINE
    for (int i = 0; i < SIZE; i++) {
        buffer[0][i] = buffer[1][i];
        buffer[1][i] = data[i];
    }
}

template <int PEs, int BATCH, int C_IN, int C_OUT, 
            int H_IN, int W_IN, int H_OUT, int W_OUT, typename T>
void ConvTranspose_hw_2(TensorMem<T>* X, TensorMem<T>* W, TensorMem<T>* B, TensorMem<T>* Y) {
    assert(X && W && B);
    Shape x_shape = X->shape;
    Shape w_shape = W->shape;
    Shape b_shape = B->shape;
    Shape y_shape = Y->shape;

    
    if (y_shape.H != (x_shape.H - 1)* 2 + (w_shape.H - 1) 
                - 1 - 1 + 1 + 1 
        || y_shape.W != (x_shape.W - 1)* 2 + (w_shape.W - 1) 
                - 1 - 1 + 1 + 1 
        || y_shape.C != w_shape.N || y_shape.N != x_shape.N)
        assert(0 && "Error: Unappropriate output sizes for convtranspose !!\n");

    int start_height = -1, start_width = -1;
    int end_height = H_OUT - 1, end_width = W_OUT - 1;
    constexpr int TC_OUT = C_OUT / PEs;

    T x_buffer[2][2][C_IN];
    T w_1x1_buffer[PEs][C_IN];
    T w_1x2_buffer[PEs][2][C_IN];
    T w_2x1_buffer[PEs][2][C_IN];
    T w_2x2_buffer[PEs][2][2][C_IN];
    T b_buffer[C_OUT];
    T y_buffer[2][2][C_OUT];


    load_buffer<C_OUT>(b_buffer, B->raw());
    for (int n = 0; n < BATCH; n++) {
        for (int hi = 0, i = 0; hi < H_IN; hi++, i += 2) 
        for (int wi = 0, j = 0; wi < W_IN; wi++, j += 2) {
            int start_wind_h = start_height + i, start_wind_w = start_width + j;
            if (wi == 0) {
                load_buffer<C_IN>(x_buffer[0][0], X->raw_at(n, hi, 0, 0));
                load_buffer<C_IN>(x_buffer[0][1], X->raw_at(n, hi, 1, 0));
                if (hi != H_IN - 1) {
                    load_buffer<C_IN>(x_buffer[1][0], X->raw_at(n, hi + 1, 0, 0));
                    load_buffer<C_IN>(x_buffer[1][1], X->raw_at(n, hi + 1, 1, 0));
                }
            } else if (wi == W_IN - 1) {
                load_buffer<C_IN>(x_buffer[0][0], x_buffer[0][1]);
                if (hi != H_IN - 1) 
                    load_buffer<C_IN>(x_buffer[1][0], x_buffer[1][1]);
            } else {
                shift_load_buffer<C_IN>(x_buffer[0], X->raw_at(n, hi, wi + 1, 0));
                if (hi != H_IN - 1) 
                    shift_load_buffer<C_IN>(x_buffer[1], X->raw_at(n, hi + 1, wi + 1, 0));
            }

            for (int pe = 0; pe < PEs; pe++) {
                for (int tco = 0; tco < TC_OUT; tco++) {
                    int co = pe* TC_OUT + tco;
                    load_buffer<C_IN>(w_1x1_buffer[pe], W->raw_at(co, 1, 1, 0));
                    load_buffer<C_IN>(w_1x2_buffer[pe][0], W->raw_at(co, 1, 2, 0));
                    load_buffer<C_IN>(w_1x2_buffer[pe][1], W->raw_at(co, 1, 0, 0));
                    load_buffer<C_IN>(w_2x1_buffer[pe][0], W->raw_at(co, 2, 1, 0));
                    load_buffer<C_IN>(w_2x1_buffer[pe][1], W->raw_at(co, 0, 1, 0));
                    load_buffer<C_IN>(w_2x2_buffer[pe][0][0], W->raw_at(co, 2, 2, 0));
                    load_buffer<C_IN>(w_2x2_buffer[pe][0][1], W->raw_at(co, 2, 0, 0));
                    load_buffer<C_IN>(w_2x2_buffer[pe][1][0], W->raw_at(co, 0, 2, 0));
                    load_buffer<C_IN>(w_2x2_buffer[pe][1][1], W->raw_at(co, 0, 0, 0));

                    y_buffer[0][0][co] = 0;
                    y_buffer[0][1][co] = 0;
                    y_buffer[1][0][co] = 0;
                    y_buffer[1][1][co] = 0;
                    for (int ci = 0; ci < C_IN; ci++) {
                        T sub_1x2[2];
                        T sub_2x1[2];
                        T sub_2x2[2][2];

                        y_buffer[0][0][co] += x_buffer[0][0][ci]* w_1x1_buffer[pe][ci];

                        for (int sub_id = 0; sub_id < 2; sub_id++) {
                            if (wi + sub_id < W_IN) 
                                sub_1x2[sub_id] = x_buffer[0][sub_id][ci]* w_1x2_buffer[pe][sub_id][ci];
                            else 
                                sub_1x2[sub_id] = 0;
                            if (hi + sub_id < H_IN) 
                                sub_2x1[sub_id] = x_buffer[sub_id][0][ci]* w_2x1_buffer[pe][sub_id][ci];
                            else 
                                sub_2x1[sub_id] = 0;
                        }
                        y_buffer[0][1][co] += sub_1x2[0] + sub_1x2[1];
                        y_buffer[1][0][co] += sub_2x1[0] + sub_2x1[1];

                        for (int sid_0 = 0; sid_0 < 2; sid_0++) {
                            for (int sid_1 = 0; sid_1 < 2; sid_1++) {
                                if (hi + sid_0 < H_IN && wi + sid_1 < W_IN) 
                                    sub_2x2[sid_0][sid_1] = x_buffer[sid_0][sid_1][ci]* w_2x2_buffer[pe][sid_0][sid_1][ci];
                                else 
                                    sub_2x2[sid_0][sid_1] = 0;
                            }
                        }
                        y_buffer[1][1][co] += (sub_2x2[0][0] + sub_2x2[0][1]) + (sub_2x2[1][0] + sub_2x2[1][1]);
                    }
                    for (int sid_0 = 0; sid_0 < 2; sid_0++) {
                        for (int sid_1 = 0; sid_1 < 2; sid_1++) {
                            y_buffer[sid_0][sid_1][co] += b_buffer[co];
                        }
                    }
                }
            }
            for (int sho = 0; sho < 2; sho++) 
            for (int swo = 0; swo < 2; swo++) {
                int ho = i + sho, wo = j + swo;
                if (ho < H_IN && wo < W_IN) 
                    store_buffer<C_OUT>(y_buffer[sho][swo], Y->raw_at(n, ho, wo, 0));
            }
        }
    }
}