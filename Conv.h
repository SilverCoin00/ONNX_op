#include "class_tensor.cpp"
#include <cassert>

/*
          [N,     H, W,  C]
X.shape = [Batch, H, W, Cin]
W.shape = [Cout,  H, W, Cin]
B.shape = [  1,   1, 1, Cout]
Y.shape = [Batch, H, W, Cout]
*/

struct Conv_Attributes {
    int dilations[2];          // [a channel filter spacing]
    int group;
    int kernel_shape[2];       // [H, W]
    int pads[4];               // [top, left, bottom, right]
    int strides[2];            // [H, W]
};

struct ConvTranspose_Attributes {
    int dilations[2];          // [a channel filter spacing]
    int group;
    int kernel_shape[2];
    int output_padding[2];     // [bottom, right]
    int pads[4];               // [top, left, bottom, right]
    int strides[2];
};

/**
 * @brief 
 * 
 * @param attributes 
 * @param X 
 * @param W 
 * @param B 
 * @return TensorMem<float>* 
 * @details njncjnij
 * @example hha.cpp
 */
TensorMem<float>* Conv(Conv_Attributes &attributes, TensorMem<float>* X, TensorMem<float>* W, TensorMem<float>* B);
TensorMem<float>* ConvTranspose(ConvTranspose_Attributes &attributes, TensorMem<float>* X, TensorMem<float>* W, TensorMem<float>* B);
