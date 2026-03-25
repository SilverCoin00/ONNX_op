#include "..\include\Core.h"



template <typename T>
TensorMem<T>::TensorMem() : data(NULL), own_memory(false) { HLS_INLINE_PRAGMA }
template <typename T>
TensorMem<T>::TensorMem(T* data, const Shape &shape, bool own_memory) 
            : data(data), shape(shape), own_memory(own_memory) { HLS_INLINE_PRAGMA }
template <typename T>
TensorMem<T>::TensorMem(const Shape &shape) : shape(shape), own_memory(true) {
    HLS_INLINE_PRAGMA
    data = new T[shape.N* shape.H* shape.W* shape.C]{};
}
template <typename T>
TensorMem<T>::~TensorMem() {
    HLS_INLINE_PRAGMA
    if (own_memory && data)
        delete[] data;
}
template <typename T>
inline int TensorMem<T>::index(int n, int h, int w, int c) {
    HLS_INLINE_PRAGMA
    return ((n* shape.H + h)* shape.W + w)* shape.C + c;
}
template <typename T>
void TensorMem<T>::print() {
    int size = shape.N* shape.H* shape.W* shape.C;
    for (int i = 0; i < size; i++) std::cout << static_cast<float>(data[i]) << " ";
    std::cout << "\n";
}
template <typename T>
inline T TensorMem<T>::get(int n, int h, int w, int c) {
    HLS_INLINE_PRAGMA
    return data[index(n, h, w, c)];
}
template <typename T>
inline T &TensorMem<T>::at(int n, int h, int w, int c) {
    HLS_INLINE_PRAGMA
    return data[index(n, h, w, c)];
}
template <typename T>
inline T* TensorMem<T>::raw_at(int n, int h, int w, int c) {
    HLS_INLINE_PRAGMA
    return &data[index(n, h, w, c)];
}
template <typename T>
void TensorMem<T>::load_tile_to_stream(const Shape &start, const Shape &size, MyStream<T>& out_stream) {
    // Vòng lặp tính toán
    for (int n = 0; n < size.N; ++n)
    for (int h = 0; h < size.H; ++h) 
    for (int w = 0; w < size.W; ++w) 
    for (int c = 0; c < size.C; ++c) {
        
        // Chỉ bật PIPELINE khi chạy HLS
        HLS_PIPELINE_PRAGMA

        // Logic tính toán vị trí bộ nhớ
        T val = data[index(n + start.N, h + start.H, w + start.W, c + start.C)];
        out_stream.write(val);
    }
}
template <typename T>
void TensorMem<T>::store_stream_to_mem(const Shape &start, const Shape &size, MyStream<T>& in_stream) {
    for (int n = 0; n < size.N; ++n)
    for (int h = 0; h < size.H; ++h) 
    for (int w = 0; w < size.W; ++w) 
    for (int c = 0; c < size.C; ++c) {
        
        HLS_PIPELINE_PRAGMA

        if (!in_stream.empty()) {
            T val = in_stream.read();
            data[index(n + start.N, h + start.H, w + start.W, c + start.C)] = val;
        }
    }
}
