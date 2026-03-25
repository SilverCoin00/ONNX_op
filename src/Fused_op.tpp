#include "..\include\Core.h"
#include <cassert>
#ifndef INTRINMINH
#define INTRINMINH 0
#endif


template<typename T>
void Relu_Pad_reflect_hw(TensorMem<T> &X) {
    int XN = X.shape.N, XH = X.shape.H, XW = X.shape.W, XC = X.shape.C;
    XH--, XW--;
    for (int n = 0; n < XN; n++) 
    for (int c = 0; c < XC; c++) {
        for (int h = 1; h < XH; h++)
        for (int w = 1; w < XW; w++) {
            T &x = X.at(n, h, w, c);
            if (x < T(0)) x = T(0);
        }
        X.at(n, 0, 0, c) = X.get(n, 2, 2, c);
        X.at(n, 0, XW, c) = X.get(n, 2, XW - 2, c);
        X.at(n, XH, 0, c) = X.get(n, XH - 2, 2, c);
        X.at(n, XH, XW, c) = X.get(n, XH - 2, XW - 2, c);
        for (int w = 1; w < XW; w++) {
            X.at(n, 0, w, c) = X.get(n, 2, w, c);
            X.at(n, XH, w, c) = X.get(n, XH - 2, w, c);
        }
        for (int h = 1; h < XH; h++) {
            X.at(n, h, 0, c) = X.get(n, h, 2, c);
            X.at(n, h, XW, c) = X.get(n, h, XW - 2, c);
        }
    }
}
template<int Size, typename T>
void Relu_Pad_reflect_hw(TensorMem<T> &X) {
    int XN = X.shape.N, XH = X.shape.H - Size, XW = X.shape.W - Size, XC = X.shape.C;
    for (int n = 0; n < XN; n++) 
    for (int c = 0; c < XC; c++) {
        for (int h = Size; h < XH; h++)
        for (int w = Size; w < XW; w++) {
            T &x = X.at(n, h, w, c);
            if (x < T(0)) x = T(0);
        }
        for (int h = 0; h < Size; h++) 
        for (int w = 0; w < Size; w++) {
            X.at(n, h, w, c) = X.get(n, Size + Size - h, Size + Size - w, c);
            X.at(n, h, XW + w, c) = X.get(n, Size + Size - h, XW - w - 2, c);
            X.at(n, XH + h, w, c) = X.get(n, XH - h - 2, Size + Size - w, c);
            X.at(n, XH + h, XW + w, c) = X.get(n, XH - h - 2, XW - w - 2, c);
        }
        for (int h = 0; h < Size; h++) 
        for (int w = Size; w < XW; w++) {
            X.at(n, h, w, c) = X.get(n, Size + Size - h, w, c);
            X.at(n, XH + h, w, c) = X.get(n, XH - h - 2, w, c);
        }
        for (int w = 0; w < Size; w++) 
        for (int h = Size; h < XH; h++) {
            X.at(n, h, w, c) = X.get(n, h, Size + Size - w, c);
            X.at(n, h, XW + w, c) = X.get(n, h, XW - w - 2, c);
        }
    }
}

template<typename T>
void Add_Pad_reflect_hw(TensorMem<T> &X1, TensorMem<T> &X2, TensorMem<T> &Y) {
    int YN = Y.shape.N, YH = Y.shape.H, YW = Y.shape.W, YC = Y.shape.C;
    YH--, YW--;
    for (int n = 0; n < YN; n++) 
    for (int c = 0; c < YC; c++) {
        for (int h = 1; h < YH; h++)
        for (int w = 1; w < YW; w++) 
            Y.at(n, h, w, c) = X1.get(n, h, w, c) + X2.get(n, h, w, c);
        
        Y.at(n, 0, 0, c) = Y.get(n, 2, 2, c);
        Y.at(n, 0, YW, c) = Y.get(n, 2, YW - 2, c);
        Y.at(n, YH, 0, c) = Y.get(n, YH - 2, 2, c);
        Y.at(n, YH, YW, c) = Y.get(n, YH - 2, YW - 2, c);
        for (int w = 1; w < YW; w++) {
            Y.at(n, 0, w, c) = Y.get(n, 2, w, c);
            Y.at(n, YH, w, c) = Y.get(n, YH - 2, w, c);
        }
        for (int h = 1; h < YH; h++) {
            Y.at(n, h, 0, c) = Y.get(n, h, 2, c);
            Y.at(n, h, YW, c) = Y.get(n, h, YW - 2, c);
        }
    }
}

template<typename T, int K_SIZE, int PEs, char RELU>
void Conv_CNorm_RPad_reflect(const Conv_Attributes &attributes, 
                TensorMem<T> &X, TensorMem<T> &W, TensorMem<T> &B, 
                //TensorMem<T> &gamma, TensorMem<T> &beta, T epsilon, 
                TensorMem<T> &Y) {
    Shape x_shape = X.shape;
    Shape w_shape = W.shape;
    Shape b_shape = B.shape;
    Shape y_shape = Y.shape;

    if (x_shape.C % attributes.group || w_shape.N % attributes.group || x_shape.C / attributes.group != w_shape.C
        || attributes.kernel_shape[0] != w_shape.H || attributes.kernel_shape[1] != w_shape.W
        || K_SIZE != w_shape.H || K_SIZE != w_shape.W) 
        assert(0 && "Error: Unappropriate input sizes for Conv !!\n");
    
    if (y_shape.H != (x_shape.H - (w_shape.H - 1)* attributes.dilations[0] 
        + attributes.pads[0] + attributes.pads[2] - 1) / attributes.strides[0] + 1 
        || y_shape.W != (x_shape.W - (w_shape.W - 1)* attributes.dilations[1] 
        + attributes.pads[1] + attributes.pads[3] - 1) / attributes.strides[1] + 1 
        || y_shape.N != x_shape.N || y_shape.C != w_shape.N)
        assert(0 && "Error: Unappropriate output sizes for Conv !!\n");

    int start_height = -attributes.pads[0], start_width = -attributes.pads[1];
    int batch = x_shape.N, C_in = w_shape.C, C_out = w_shape.N / attributes.group;
    b_shape.N = b_shape.H = b_shape.W = 0;

    assert(!(C_in % PEs));
    int rolls = C_in / PEs;
    constexpr int buff_size = K_SIZE* K_SIZE;
    T y_points[buff_size];
    #pragma HLS ARRAY_PARTITION variable=y_points complete

    // int num = y_shape.C;
    // T adjustment_scale = (T) num / (num - 1);
    // T pre_div = (T) 1 / (T) (sqrtf(num - 1));
    
    for (int n = 0; n < batch; n++) {
        x_shape.N = y_shape.N = n;
        for (int g = 0; g < attributes.group; g++) 
            for (int y_h = 0, e = 0; y_h < y_shape.H; y_h++, e += attributes.strides[0]) 
            for (int y_w = 0, f = 0; y_w < y_shape.W; y_w++, f += attributes.strides[1]) {
                for (int co = 0; co < C_out; co++) {
                    y_shape.C = w_shape.N = b_shape.C = g* C_out + co;
                    x_shape.H = start_height + e, x_shape.W = start_width + f;
                    for (int i = 0; i < buff_size; i++) 
                        y_points[i] = 0;
                    
                    for (int w_h = 0, i = 0; w_h < K_SIZE; w_h++, i += attributes.dilations[0]) 
                    for (int w_w = 0, j = 0; w_w < K_SIZE; w_w++, j += attributes.dilations[1]) {
                        int id = w_h* K_SIZE + w_w;
                        int h = x_shape.H + i, w = x_shape.W + j;
                        //if (h >= 0 && w >= 0 && h < X.shape.H && w < X.shape.W) 
                        if (h < 0) h = -h;
                        else if (h >= X.shape.H) h = (X.shape.H << 1) - h - 2;
                        if (w < 0) w = -w;
                        else if (w >= X.shape.W) w = (X.shape.W << 1) - w - 2;
                        for (int r = 0; r < rolls; r++) {
                            if ((PEs & 7) || !INTRINMINH) {
                            for (int ci = 0; ci < PEs; ci++) {
                                x_shape.C = (g* rolls + r)* PEs + ci;
                                w_shape.C = r* PEs + ci;
                                
                                y_points[id] += X.get(x_shape.N, h, w, x_shape.C)* W.get(w_shape.N, w_h, w_w, w_shape.C);
                            }
                            } else {
                            // float32x4_t sum_low = vdupq_n_f32(0.0f);
                            // float32x4_t sum_high = vdupq_n_f32(0.0f);
                            __m256 sum_vec;
                            for (int ci = 0; ci < PEs; ci += 8) {
                                x_shape.C = (g* rolls + r)* PEs + ci;
                                w_shape.C = r* PEs + ci;

                                __m128i vec_x16 = _mm_loadu_si128((__m128i*) X.raw_at(x_shape.N, h, w, x_shape.C));
                                __m128i vec_w16 = _mm_loadu_si128((__m128i*) W.raw_at(w_shape.N, w_h, w_w, w_shape.C));

                                __m256 vec_x32 = _mm256_cvtph_ps(vec_x16);
                                __m256 vec_w32 = _mm256_cvtph_ps(vec_w16);

                                // float16x8_t vec_x16 = vld1q_f16(X.raw_at(x_shape.N, h, w, x_shape.C));
                                // float16x8_t vec_w16 = vld1q_f16(W.raw_at(w_shape.N, w_h, w_w, w_shape.C));

                                // float16x4_t vec_x16_low = vget_low_f16(vec_x16);
                                // float16x4_t vec_x16_high = vget_high_f16(vec_x16);
                                // float16x4_t vec_w16_low = vget_low_f16(vec_w16);
                                // float16x4_t vec_w16_high = vget_high_f16(vec_w16);

                                // float32x4_t vec_x32_low = vcvt_f32_f16(vec_x16_low);
                                // float32x4_t vec_x32_high = vcvt_f32_f16(vec_x16_high);
                                // float32x4_t vec_w32_low = vcvt_f32_f16(vec_w16_low);
                                // float32x4_t vec_w32_high = vcvt_f32_f16(vec_w16_high);

                                sum_vec = _mm256_fmadd_ps(vec_x32, vec_w32, sum_vec);
                                // Nếu CPU không hỗ trợ FMA, dùng 2 lệnh rời:
                                // __m256 mul_res = _mm256_mul_ps(vec_x16, vec_w16);
                                // sum_vec = _mm256_add_ps(sum_vec, mul_res);

                                // sum_low = vmlaq_f32(sum_low, vec_x32_low, vec_w32_low);
                                // sum_high = vmlaq_f32(sum_high, vec_x32_high, vec_w32_high);
                            }
                            // Bước 2.1: Tách thanh ghi 256-bit làm 2 nửa 128-bit (mỗi nửa 4 số float) và cộng lại với nhau
                            __m128 lo_128 = _mm256_castps256_ps128(sum_vec);       // Lấy 4 float thấp
                            __m128 hi_128 = _mm256_extractf128_ps(sum_vec, 1);    // Lấy 4 float cao
                            __m128 sum_128 = _mm_add_ps(lo_128, hi_128);          // Cộng lại -> Còn 4 float

                            // Bước 2.2: Dùng lệnh hadd (Horizontal Add) để gập 4 float thành 2 float, rồi thành 1 float
                            sum_128 = _mm_hadd_ps(sum_128, sum_128);              // Cộng các cặp cạnh nhau (còn 2 giá trị hữu ích)
                            sum_128 = _mm_hadd_ps(sum_128, sum_128);              // Cộng lần nữa (còn 1 giá trị hữu ích nằm ở vị trí đầu)

                            // Bước 2.3: Trích xuất số float duy nhất ra biến C++ thông thường
                            float final_sum = _mm_cvtss_f32(sum_128);

                            // float32x4_t final_sum_vec = vaddq_f32(sum_low, sum_high);
                            // float final_sum = vaddvq_f32(final_sum_vec);

                            y_points[id] += (T) final_sum;
                            }
                        }
                    }
                    for (int stride = 1; stride < buff_size; stride <<= 1) {
                        for (int i = 0; i + stride < buff_size; i += stride << 1) 
                            y_points[i] += y_points[i + stride];
                    }
                    Y.at(y_shape.N, y_h, y_w, y_shape.C) = y_points[0] + B.get(b_shape.N, b_shape.H, b_shape.W, b_shape.C);
                }

                // T mean = 0, var = 0;
                // for (int co = 0; co < C_out; co++) {
                //     T cur = Y.get(n, y_h, y_w, co);
                //     mean += cur;
                //     T cur_n = cur* pre_div;
                //     var += cur_n* cur_n;
                // }
                // mean /= num;
                // var = (T) 1 / static_cast<T>(sqrtf(var - mean* mean* adjustment_scale + epsilon));
                // for (int co = 0; co < C_out; co++) {
                //     T weight = gamma.raw()[co]* var;
                //     T bias = beta.raw()[co] - mean* weight;
                //     T &point = Y.at(n, y_h, y_w, co);
                //     point = point* weight + bias;
                //     if (RELU && point < 0) point = 0;
                // }
            }
    }
}
