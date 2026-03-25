#ifndef TENSOR_MEM_H
#define TENSOR_MEM_H


#if defined(USE_HLS) || defined(__SYNTHESIS__)
    // --- MÔI TRƯỜNG HLS (Hardware) ---
    #define HLS_INLINE_PRAGMA _Pragma("HLS INLINE")
    #define HLS_PIPELINE_PRAGMA _Pragma("HLS PIPELINE II=1")

    #include <hls_stream.h>
    #include <ap_axi_sdata.h>
    #include <ap_int.h>
    #include <ap_fixed.h>

    // Dùng ap_fixed cho phần cứng để tối ưu tài nguyên
    typedef ap_fixed<16, 8> data_t;

    // Alias stream của HLS
    template<typename T>
    using MyStream = hls::stream<T>;

#else
    // --- MÔI TRƯỜNG PURE C++ (Software Simulation) ---
    #define HLS_INLINE_PRAGMA
    #define HLS_PIPELINE_PRAGMA
    #define INTRINMINH 1

    #include <iostream>
    #include <vector>
    #include <queue>
    #include <cmath>
    #include <cassert>
    #include <inttypes.h>
    #include <immintrin.h>
    //#include <arm_neon.h>

    typedef float data_t;

    template<typename T>
    class MyStream {
    private:
        std::queue<T> q;
    public:
        void write(const T& val) { q.push(val); }
        T read() { 
            if(q.empty()) return 0; 
            T val = q.front(); 
            q.pop(); 
            return val; 
        }
        bool empty() { return q.empty(); }
        bool full() { return false; }
    };

#endif

#define N_AXIS 0
#define H_AXIS 1
#define W_AXIS 2
#define C_AXIS 3


template <typename T>
struct is_integral {
    enum { value = 0 };
};

template <> struct is_integral<int8_t>         { enum { value = 1 }; };
template <> struct is_integral<int16_t>         { enum { value = 1 }; };
template <> struct is_integral<int32_t>         { enum { value = 1 }; };
template <> struct is_integral<int64_t>         { enum { value = 1 }; };

template <> struct is_integral<uint8_t>         { enum { value = 1 }; };
template <> struct is_integral<uint16_t>         { enum { value = 1 }; };
template <> struct is_integral<uint32_t>         { enum { value = 1 }; };
template <> struct is_integral<uint64_t>         { enum { value = 1 }; };

template <typename T>
struct is_float_point {
    enum { value = 0 };
};

template <> struct is_float_point<float> { enum { value = 1 }; };
template <> struct is_float_point<_Float16> { enum { value = 1 }; };

template <typename T>
struct type_limits;

template <>
struct type_limits<int8_t> {
    static constexpr int8_t min = -128;
    static constexpr int8_t max = 127;
};

template <>
struct type_limits<uint8_t> {
    static constexpr uint8_t min = 0;
    static constexpr uint8_t max = 255;
};
template <>
struct type_limits<int16_t> {
    static constexpr int16_t min = -32768;
    static constexpr int16_t max = 32767;
};

template <typename T>
constexpr T clamp(int x) {
    if (x > type_limits<T>::max) return type_limits<T>::max;
    if (x < type_limits<T>::min) return type_limits<T>::min;
    return x;
}


struct Shape {
    int N; // Batch size
    int H; // Height
    int W; // Width
    int C; // Channels
    /**
     * @brief New Shape object with full 0-sizes
     * 
     */
    Shape() : N(0), H(0), W(0), C(0) {}
    /**
     * @brief New Shape object
     * 
     * @param n Batchsize
     * @param h Height
     * @param w Width
     * @param c Channels
     */
    Shape(int n, int h, int w, int c) : N(n), H(h), W(w), C(c) {}

    bool operator==(const Shape &other) const {
        return N == other.N && H == other.H && W == other.W && C == other.C;
    }
    bool operator>=(const Shape &other) const {
        return N >= other.N && H >= other.H && W >= other.W && C >= other.C;
    }
#if defined(USE_HLS) || defined(__SYNTHESIS__)
#else
    void print() {
        std::cout << "[" << N << " " << H << " " << W << " " << C << "]\n";
    }
#endif
};

template <typename T>
class TensorMem {
private:
    T* data;
    bool own_memory;
    inline int index(int n, int h, int w, int c);

public:
    Shape shape;

    /**
     * @brief New Tensor with no mem
     */
    TensorMem();
    /**
     * @brief New Tensor
     * 
     * @param data 
     * @param shape {N, H, W, C}
     * @param is_own_memory 
     */
    TensorMem(T* data, const Shape &shape, bool own_memory);
    /**
     * @brief New Tensor with own full 0s mem
     * 
     * @param shape {N, H, W, C}
     */
    TensorMem(const Shape &shape);
    ~TensorMem();

    /**
     * @brief print data of tensor
     * 
     */
    void print();

    /**
     * @brief get tensor value at position
     * 
     * @param n 
     * @param h 
     * @param w 
     * @param c 
     * @return (T) value 
     */
    inline T get(int n, int h, int w, int c);

    /**
     * @brief get tensor reference at position
     * 
     * @param n 
     * @param h 
     * @param w 
     * @param c 
     * @return (T) reference 
     */
    inline T &at(int n, int h, int w, int c);

    /**
     * @brief raw data
     * 
     * @return T* 
     */
    inline T* raw() { return data; }
    /**
     * @brief raw data
     * 
     * @return const T* 
     */
    inline const T* raw() const { return data; }
    inline T* raw_at(int n, int h, int w, int c);

    void load_tile_to_stream(const Shape &start, const Shape &size, MyStream<T>& out_stream);
    void store_stream_to_mem(const Shape &start, const Shape &size, MyStream<T>& in_stream);
};



struct Conv_Attributes {
    int dilations[2];          // [a channel filter spacing]
    int group;
    int kernel_shape[2];       // [H, W]
    int pads[4];               // [top, left, bottom, right]
    int strides[2];            // [H, W]
    Conv_Attributes(int dil_1, int dil_2, int group, int kernel_h, int kernel_w, 
                    int pad_top, int pad_left, int pad_bottom, int pad_right, 
                    int stride_h, int stride_w) : group(group) {
        dilations[0] = dil_1, dilations[1] = dil_2;
        kernel_shape[0] = kernel_h, kernel_shape[1] = kernel_w;
        pads[0] = pad_top, pads[1] = pad_left, pads[2] = pad_bottom, pads[3] = pad_right;
        strides[0] = stride_h, strides[1] = stride_w;
    }
};
struct ConvTranspose_Attributes {
    int dilations[2];          // [a channel filter spacing]
    int group;
    int kernel_shape[2];       // [H, W]
    int output_padding[2];     // [bottom, right]
    int pads[4];               // [top, left, bottom, right]
    int strides[2];            // [H, W]
    ConvTranspose_Attributes(int dil_1, int dil_2, int group, int kernel_h, int kernel_w, 
                                int output_pad_bottom, int output_pad_right, 
                                int pad_top, int pad_left, int pad_bottom, int pad_right, 
                                int stride_h, int stride_w) : group(group) {
        dilations[0] = dil_1, dilations[1] = dil_2;
        kernel_shape[0] = kernel_h, kernel_shape[1] = kernel_w;
        output_padding[0] = output_pad_bottom, output_padding[1] = output_pad_right;
        pads[0] = pad_top, pads[1] = pad_left, pads[2] = pad_bottom, pads[3] = pad_right;
        strides[0] = stride_h, strides[1] = stride_w;
    }
};

/**
 * @brief 
 * 
 * @param attributes parameters of conv -- {[0], [1], [2], [3], [4]}
 * @param attributes[0] -- {dilations_h, dilations_w}
 * @param attributes[1] -- group
 * @param attributes[2] -- {kernel_size_h, kernel_size_w}
 * @param attributes[3] -- {pad_top, pad_left, pad_bottom, pad_right}
 * @param attributes[4] -- {stride_h, stride_w}
 * 
 * @param X [Batch, H, W, Cin]
 * @param W [Cout,  H, W, Cin]
 * @param B [  1,   1, 1, Cout]
 * @return TensorMem<T>* -- [Batch, H, W, Cout]
 */
template <typename T>
TensorMem<T>* Conv(const Conv_Attributes &attributes, TensorMem<T>* X, TensorMem<T>* W, TensorMem<T>* B);
/**
 * @brief 
 * 
 * @param attributes parameters of conv -- {[0], [1], [2], [3], [4]}
 * @param attributes[0] -- {dilations_h, dilations_w}
 * @param attributes[1] -- group
 * @param attributes[2] -- {kernel_size_h, kernel_size_w}
 * @param attributes[3] -- {pad_top, pad_left, pad_bottom, pad_right}
 * @param attributes[4] -- {stride_h, stride_w}
 * 
 * @param X [Batch, H, W, Cin]
 * @param W [Cout,  H, W, Cin]
 * @param B [  1,   1, 1, Cout]
 * @param Y [Batch, H, W, Cout]
 */
template <typename T>
void Conv(const Conv_Attributes &attributes, TensorMem<T>* X, TensorMem<T>* W, TensorMem<T>* B, TensorMem<T>* Y);
/**
 * @brief 
 * 
 * @param attributes parameters of convtranspose -- {[0], [1], [2], [3], [4], [5]}
 * @param attributes[0] -- {dilations_h, dilations_w}
 * @param attributes[1] -- group
 * @param attributes[2] -- {kernel_size_h, kernel_size_w}
 * @param attributes[3] -- {output_pad_h, output_pad_w}
 * @param attributes[4] -- {pad_top, pad_left, pad_bottom, pad_right}
 * @param attributes[5] -- {stride_h, stride_w}
 * 
 * @param X [Batch, H, W, Cin]
 * @param W [Cout,  H, W, Cin]
 * @param B [  1,   1, 1, Cout]
 * @return TensorMem<T>* -- [Batch, H, W, Cout]
 */
template <typename T>
TensorMem<T>* ConvTranspose(const ConvTranspose_Attributes &attributes, TensorMem<T>* X, TensorMem<T>* W, TensorMem<T>* B);
/**
 * @brief 
 * 
 * @param attributes parameters of convtranspose -- {[0], [1], [2], [3], [4], [5]}
 * @param attributes[0] -- {dilations_h, dilations_w}
 * @param attributes[1] -- group
 * @param attributes[2] -- {kernel_size_h, kernel_size_w}
 * @param attributes[3] -- {output_pad_h, output_pad_w}
 * @param attributes[4] -- {pad_top, pad_left, pad_bottom, pad_right}
 * @param attributes[5] -- {stride_h, stride_w}
 * 
 * @param X [Batch, H, W, Cin]
 * @param W [Cout,  H, W, Cin]
 * @param B [  1,   1, 1, Cout]
 * @param Y [Batch, H, W, Cout]
 */
template <typename T>
void ConvTranspose(const ConvTranspose_Attributes &attributes, TensorMem<T>* X, TensorMem<T>* W, TensorMem<T>* B, TensorMem<T>* Y);

/**
 * @brief Concat X(s) into reference parameter Y
 * 
 * @param X {X1, X2, ..., Xn}
 * @param Y reference to res place
 * @param num_inputs n
 * @param axis -- N_AXIS || H_AXIS || W_AXIS || C_AXIS
 */
template <typename T>
void Concat(TensorMem<T>** X, TensorMem<T> &Y, int num_inputs, int axis);
/**
 * @brief Concat X(s) and return res Y
 * 
 * @param X {X1, X2, ..., Xn}
 * @param num_inputs n
 * @param axis -- N_AXIS || H_AXIS || W_AXIS || C_AXIS
 * @return TensorMem<T>*
 */
template <typename T>
TensorMem<T>* Concat(TensorMem<T>** X, int num_inputs, int axis);

/**
 * @brief gather tensors chosen by indices[i] follow an axis
 * 
 * @param X tensor to choose X[ii]
 * @param Y reference to res place
 * @param indices {i1, i2, ..., in}
 * @param idx_count n
 * @param axis -- N_AXIS || H_AXIS || W_AXIS || C_AXIS
 */
template <typename T>
void Gather(TensorMem<T> &X, TensorMem<T> &Y, const int* indices, int idx_count, int axis);
/**
 * @brief gather tensors chosen by indices[i] follow an axis
 * 
 * @param X tensor to choose X[ii]
 * @param indices {i1, i2, ..., in}
 * @param idx_count n
 * @param axis -- N_AXIS || H_AXIS || W_AXIS || C_AXIS
 * @return TensorMem<T>*
 */
template <typename T>
TensorMem<T>* Gather(TensorMem<T> &X, const int* indices, int idx_count, int axis);

/**
 * @brief copy X data to Y
 * 
 * @tparam T 
 * @param X 
 * @param Y = X
 */
template <typename T>
void Identity(const TensorMem<T> &X, TensorMem<T> &Y);
/**
 * @brief return a copy of X
 * 
 * @tparam T 
 * @param X 
 * @return TensorMem<T>* 
 */
template <typename T>
TensorMem<T>* Identity(const TensorMem<T> &X);

/**
 * @brief add 0s around X to Y
 * 
 * @param X 
 * @param Y 
 * @param pad [begin_N, begin_H, begin_W, begin_C,  end_N, end_H, end_W, end_C]
 * @param mode 
 * constant: [val, val, A, B, C, D, val];
 * reflect:  [C,   B,   A, B, C, D,  C];
 * edge:     [A,   A,   A, B, C, D,  D];
 * @param const_value only work with constant mode
 */
template <typename T>
void Pad(TensorMem<T> &X, TensorMem<T> &Y, const int* pad, const char* mode, T const_value);
/**
 * @brief return X added 0s around
 * 
 * @param X 
 * @param pad [begin_N, begin_H, begin_W, begin_C,  end_N, end_H, end_W, end_C]
 * @param mode 
 * constant: [val, val, A, B, C, D, val];
 * reflect:  [C,   B,   A, B, C, D,  C];
 * edge:     [A,   A,   A, B, C, D,  D];
 * @param const_value only work with constant mode
 * @return TensorMem<T>* 
 */
template <typename T>
TensorMem<T>* Pad(TensorMem<T> &X, T const_value, const int* pad, const char* mode);
/**
 * @brief add 0s around X
 * 
 * @param X 
 * @param Y 
 * @param pad [begin_N, begin_H, begin_W, begin_C,  end_N, end_H, end_W, end_C]
 * @param mode 
 * constant: [val, val, A, B, C, D, val];
 * reflect:  [C,   B,   A, B, C, D,  C];
 * edge:     [A,   A,   A, B, C, D,  D];
 * @param const_value only work with constant mode
 */
template <typename T>
void Pad(TensorMem<T> &X, const int* pad, const char* mode, T const_value);

/**
 * @brief copy X data to reshaped reference Y
 * 
 * @param X 
 * @param Y 
 */
template <typename T>
void Reshape(TensorMem<T> &X, TensorMem<T> &Y);
/**
 * @brief reshape X
 * 
 * @param X 
 * @param shape 
 * @return TensorMem<T>* 
 */
template <typename T>
TensorMem<T>* Reshape(TensorMem<T> &X, const Shape &shape);

/**
 * @brief return shape of X
 * 
 * @param X 
 * @return TensorMem<int64_t>* [N, H, W, C]
 */
template <typename T>
TensorMem<int64_t>* Shapeof(TensorMem<T> &X);

/**
 * @brief slice part of X from pos_0 to pos_1 to Y
 * 
 * @param X 
 * @param Y 
 * @param pos_0 [N0, H0, W0, C0]
 * @param pos_1 [N1, H1, W1, C1]
 * @param step can be > 0 || < 0
 */
template <typename T>
void Slice(TensorMem<T> &X, TensorMem<T> &Y, const Shape &pos_0, const Shape &pos_1, const int* step);
/**
 * @brief return a slice part of X from pos_0 to pos_1
 * 
 * @param X 
 * @param pos_0 [N0, H0, W0, C0]
 * @param pos_1 [N1, H1, W1, C1]
 * @param step can be > 0 || < 0
 * @return TensorMem<T>* 
 */
template <typename T>
TensorMem<T>* Slice(TensorMem<T> &X, const Shape &pos_0, const Shape &pos_1, const int* step);

/**
 * @brief fill X permuted to Y
 * 
 * @param X 
 * @param Y 
 * @param perm [H_AXIS, C_AXIS, W_AXIS, N_AXIS] or any order else
 */
template <typename T>
void Transpose(TensorMem<T> &X, TensorMem<T> &Y, int perm[]);
/**
 * @brief return X permuted
 * 
 * @param X 
 * @param perm [H_AXIS, C_AXIS, W_AXIS, N_AXIS] or any order else
 * @return TensorMem<T>* 
 */
template <typename T>
TensorMem<T>* Transpose(TensorMem<T> &X, int perm[]);

/**
 * @brief load const data node
 * 
 * @tparam T 
 * @param X 
 * @param Y = X
 * @param size 
 */
template <typename T>
void Constant(const T* X, TensorMem<T> &Y, int size);
/**
 * @brief write a value full to tensor
 * 
 * @param Y 
 * @param val 
 */
template <typename T>
void Constant_of_shape(TensorMem<T> &Y, T val);
/**
 * @brief return a tensor with full value
 * 
 * @param shape 
 * @param val 
 * @return TensorMem<T>* 
 */
template <typename T>
TensorMem<T>* Constant_of_shape(const Shape &shape, T val);

/**
 * @brief <X_type, Y_type> load cast data from X to Y
 * 
 * @tparam T_IN 
 * @tparam T_OUT 
 * @param X 
 * @param Y = Cast<type_t>(X)
 */
template <typename T_IN, typename T_OUT>
void Cast(TensorMem<T_IN> &X, TensorMem<T_OUT> &Y);
/**
 * @brief <output_type> return cast tensor from X
 * 
 * @tparam T_OUT 
 * @tparam T_IN 
 * @param X 
 * @return TensorMem<T_OUT>* Cast<type_t>(X)
 */
template <typename T_OUT, typename T_IN>
auto Cast(TensorMem<T_IN> &X) -> TensorMem<T_OUT>*;

/**
 * @brief load relu of X to Y
 * @brief ( can self apply )
 * 
 * @tparam T 
 * @param X 
 * @param Y = relu(X)
 */
template <typename T>
void Relu(TensorMem<T> &X, TensorMem<T> &Y);
/**
 * @brief return relu of X
 * 
 * @tparam T 
 * @param X 
 * @return TensorMem<T>* 
 */
template <typename T>
TensorMem<T>* Relu(TensorMem<T> &X);

/**
 * @brief load sqrt of X to Y
 * @brief ( can self apply )
 * 
 * @tparam T 
 * @param X 
 * @param Y 
 */
template <typename T>
void Sqrt(TensorMem<T> &X, TensorMem<T> &Y);
/**
 * @brief return sqrt of X
 * 
 * @tparam T 
 * @param X 
 * @return TensorMem<T>* 
 */
template <typename T>
TensorMem<T>* Sqrt(TensorMem<T> &X);

/**
 * @brief load X clamped min, max to Y
 * @brief ( can self apply )
 * 
 * @tparam T 
 * @param X 
 * @param Y 
 * @param min 
 * @param max 
 */
template <typename T>
void Clip(TensorMem<T> &X, TensorMem<T> &Y, T min, T max);
/**
 * @brief return X clamped min, max
 * @brief ( can self apply )
 * 
 * @tparam T 
 * @param X 
 * @param min 
 * @param max 
 * @return TensorMem<T>* 
 */
template <typename T>
TensorMem<T>* Clip(TensorMem<T> &X, T min, T max);

/**
 * @brief load sum X1, X2 to Y
 * @brief ( can self apply )
 * 
 * @param X1 
 * @param X2 
 * @param Y 
 */
template <typename T>
void Add(TensorMem<T> &X1, TensorMem<T> &X2, TensorMem<T> &Y);
/**
 * @brief return sum of X1, X2
 * 
 * @param X1 
 * @param X2 
 * @return TensorMem<T>* 
 */
template <typename T>
TensorMem<T>* Add(TensorMem<T> &X1, TensorMem<T> &X2);
/**
 * @brief load X1 - X2 to Y
 * @brief ( can self apply )
 * 
 * @param X1 
 * @param X2 
 * @param Y 
 */
template <typename T>
void Sub(TensorMem<T> &X1, TensorMem<T> &X2, TensorMem<T> &Y);
/**
 * @brief return X1 - X2
 * 
 * @param X1 
 * @param X2 
 * @return TensorMem<T>* 
 */
template <typename T>
TensorMem<T>* Sub(TensorMem<T> &X1, TensorMem<T> &X2);
/**
 * @brief load X1 elewise mul X2 to Y
 * @brief ( can self apply )
 * 
 * @param X1 
 * @param X2 
 * @param Y 
 */
template <typename T>
void Mul(TensorMem<T> &X1, TensorMem<T> &X2, TensorMem<T> &Y);
/**
 * @brief return elewise mul of X1, X2
 * 
 * @param X1 
 * @param X2 
 * @return TensorMem<T>* 
 */
template <typename T>
TensorMem<T>* Mul(TensorMem<T> &X1, TensorMem<T> &X2);
/**
 * @brief load X1 elewise divide X2 to Y 
 * 
 * @param X1 
 * @param X2 
 * @param Y 
 */
template <typename T>
void Div(TensorMem<T> &X1, TensorMem<T> &X2, TensorMem<T> &Y);
/**
 * @brief return elewise divide of X1, X2
 * 
 * @param X1 
 * @param X2 
 * @return TensorMem<T>* 
 */
template <typename T>
TensorMem<T>* Div(TensorMem<T> &X1, TensorMem<T> &X2);

/**
 * @brief load floor data X to Y
 * @brief ( can self apply )
 * 
 * @tparam T 
 * @param X 
 * @param Y 
 */
template <typename T>
void Floor(TensorMem<T> &X, TensorMem<T> &Y);
/**
 * @brief return floor of X
 * @brief ( can self apply )
 * 
 * @tparam T 
 * @param X 
 * @return TensorMem<T>* 
 */
template <typename T>
TensorMem<T>* Floor(TensorMem<T> &X);
template <typename T>
void Round(TensorMem<T> &X, TensorMem<T> &Y);
template <typename T>
TensorMem<T>* Round(TensorMem<T> &X);

/**
 * @brief load a sum-1_axis tensor of X to Y
 * 
 * @tparam T 
 * @param X 
 * @param Y 
 * @param axis 
 */
template <typename T>
void ReduceSum(TensorMem<T> &X, TensorMem<T> &Y, int axis);
/**
 * @brief return a sum-1_axis of X
 * 
 * @tparam T 
 * @param X 
 * @param axis 
 * @return TensorMem<T>* 
 */
template <typename T>
TensorMem<T>* ReduceSum(TensorMem<T> &X, int axis);
/**
 * @brief load a mean-1_axis tensor of X to Y
 * 
 * @tparam T 
 * @param X 
 * @param Y 
 * @param axis -- N_AXIS || H_AXIS || W_AXIS || C_AXIS
 */
template <typename T>
void ReduceMean(TensorMem<T> &X, TensorMem<T> &Y, int axis);
/**
 * @brief return a mean-1_axis of X
 * 
 * @tparam T 
 * @param X 
 * @param axis -- N_AXIS || H_AXIS || W_AXIS || C_AXIS
 * @return TensorMem<T>* 
 */
template <typename T>
TensorMem<T>* ReduceMean(TensorMem<T> &X, int axis);

/**
 * @brief load a product-1_axis tensor of X to Y
 * 
 * @tparam T 
 * @param X 
 * @param Y 
 * @param axis -- N_AXIS || H_AXIS || W_AXIS || C_AXIS
 */
template <typename T>
void ReduceProd(TensorMem<T> &X, TensorMem<T> &Y, int axis);
/**
 * @brief return a product-1_axis of X
 * 
 * @tparam T 
 * @param X 
 * @param axis -- N_AXIS || H_AXIS || W_AXIS || C_AXIS
 * @return TensorMem<T>* 
 */
template <typename T>
TensorMem<T>* ReduceProd(TensorMem<T> &X, int axis);

/**
 * @brief load X quantized to Y
 * @brief Y = clamp(round(X / scale) + zero_point)
 * 
 * @tparam T_IN 
 * @tparam T_OUT 
 * @param X 
 * @param Y 
 * @param scale (tensor)
 * @param zero_point (tensor)
 */
template <typename T_IN, typename T_OUT>
void QuantizeLinear(TensorMem<T_IN> &X, TensorMem<T_OUT> &Y, TensorMem<T_IN> scale, TensorMem<T_OUT> zero_point);
/**
 * @brief load X quantized to Y
 * @brief Y = clamp(round(X / scale) + zero_point)
 * 
 * @tparam T_IN 
 * @tparam T_OUT 
 * @param X 
 * @param Y 
 * @param scale 
 * @param zero_point 
 */
template <typename T_IN, typename T_OUT>
void QuantizeLinear(TensorMem<T_IN> &X, TensorMem<T_OUT> &Y, T_IN scale, T_OUT zero_point);
/**
 * @brief load X dequantized to Y
 * @brief Y = (X - zero_point)* scale
 * 
 * @tparam T_IN 
 * @tparam T_OUT 
 * @param X 
 * @param Y 
 * @param scale (tensor)
 * @param zero_point (tensor)
 */
template <typename T_IN, typename T_OUT>
void DequantizeLinear(TensorMem<T_IN> &X, TensorMem<T_OUT> &Y, TensorMem<T_OUT> scale, TensorMem<T_IN> zero_point);
/**
 * @brief load X dequantized to Y
 * @brief Y = (X - zero_point)* scale
 * 
 * @tparam T_IN 
 * @tparam T_OUT 
 * @param X 
 * @param Y 
 * @param scale 
 * @param zero_point 
 */
template <typename T_IN, typename T_OUT>
void DequantizeLinear(TensorMem<T_IN> &X, TensorMem<T_OUT> &Y, T_OUT scale, T_IN zero_point);

/**
 * @brief quantize add X1, X2 to Y
 * @brief |: Sy* (Y - Zy) = S1* (X1 - Z1) + S2* (X2 - Z2)
 * @brief ->  Y = Zy + (X1 - Z1)* S1/Sy + (X2 - Z2)* S2/Sy
 * @brief |->  S1/Sy = M1.2^(-N1), S2/Sy = M2.2^(-N2)
 * 
 * @tparam T 
 * @param X1 
 * @param X2 
 * @param Y 
 * @param M1 
 * @param N1 
 * @param M2 
 * @param N2 
 * @param zero_point_1 
 * @param zero_point_2 
 * @param zero_point_y 
 */
template<typename T>
void QLinearAdd(TensorMem<T> &X1, TensorMem<T> &X2, TensorMem<T> &Y, 
                int32_t M1, int N1, int32_t M2, int N2, 
                T zero_point_1, T zero_point_2, T zero_point_y);
/**
 * @brief quantize mul X1, X2 to Y
 * @brief |: Sy* (Y - Zy) = S1* (X1 - Z1)* S2* (X2 - Z2)
 * @brief ->  Y = Zy + (X1 - Z1)* (X2 - Z2)* S1.S2/Sy 
 * @brief |->  S1.S2/Sy = M.2^(-N)
 * 
 * @tparam T 
 * @param X1 
 * @param X2 
 * @param Y  
 * @param M 
 * @param N 
 * @param zero_point_1 
 * @param zero_point_2 
 * @param zero_point_y 
 */
template<typename T>
void QLinearMul(TensorMem<T> &X1, TensorMem<T> &X2, TensorMem<T> &Y, 
                int32_t M, int N, 
                T zero_point_1, T zero_point_2, T zero_point_y);
/**
 * @brief quantize conv X to Y
 * @brief |: Sy* (Y - Zy) = Sx* (X - Zx) @ Sw* (W - Zw) + Sb* (B - Zb)
 * @brief ->  Y = Zy + (X - Zx) @ (W - Zw) * Sx.Sw/Sy + (B - Zb)* Sb/Sy = Zy + ((X - Zx) @ (W - Zw) + B)* Sx.Sw/Sy
 * @brief |->  Sx.Sw/Sy = M.2^(-N)
 * 
 * @tparam T 
 * @param attributes parameters of conv -- {[0], [1], [2], [3], [4]}
 * @param attributes[0] -- {dilations_h, dilations_w}
 * @param attributes[1] -- group
 * @param attributes[2] -- {kernel_size_h, kernel_size_w}
 * @param attributes[3] -- {pad_top, pad_left, pad_bottom, pad_right}
 * @param attributes[4] -- {stride_h, stride_w}
 * @param X [Batch, H, W, Cin]
 * @param W [Cout,  H, W, Cin]
 * @param B [  1,   1, 1, Cout]
 * @param Y [Batch, H, W, Cout]
 * @param M [  1,   1, 1, Cout]
 * @param N [  1,   1, 1, Cout]
 * @param x_zero_point 
 * @param w_zero_point 
 * @param y_zero_point 
 */
template<typename T_IN, typename T_W, typename T_OUT>
void QLinearConv(const Conv_Attributes &attributes, TensorMem<T_IN>* X, TensorMem<T_W>* W, TensorMem<int32_t>* B, TensorMem<T_OUT>* Y, 
                    TensorMem<int32_t>* M, TensorMem<int>* N, T_IN x_zero_point, TensorMem<T_W>* w_zero_point, T_OUT y_zero_point);

/**
 * @brief load X normalized to Y
 * @brief | mean = Reducemean(X, axis)
 * @brief | X = X - mean
 * @brief | var = Reducemean(X^2, axis)
 * @brief | std = sqrt(var + epsilon)
 * @brief | Y = gamma * (X / std) + beta
 * 
 * @tparam T 
 * @param X 
 * @param Y 
 * @param gamma 
 * @param beta 
 * @param epsilon 
 * @param axis 
 * @param extra_mem_size_X_reduced_axis 
 */
template <typename T>
void Norm(TensorMem<T> &X, TensorMem<T> &Y, TensorMem<T> &gamma, TensorMem<T> &beta, T epsilon, int axis, 
            TensorMem<T> &extra_mem_size_X_reduced_axis);

#include "..\src\Arithmetic.tpp"
#include "..\src\class_tensor.tpp"
#include "..\src\Manipulation.tpp"
#include "..\src\Reduce.tpp"
#include "..\src\Unary.tpp"
#include "..\src\Conv.tpp"
#include "..\src\ConvTranspose.tpp"
#include "..\src\QuantizationLinear.tpp"
#include "..\src\Quantize_Fused_op.tpp"
#include "..\src\Norm.tpp"

#endif 
