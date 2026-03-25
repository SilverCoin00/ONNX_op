#include "..\include\Core.h"
#include <cstring>


template <typename T>
void Concat(TensorMem<T>** X, TensorMem<T> &Y, int num_inputs, int axis) {
    assert(axis >= 0 && axis <= 3);
    int n_offset = 0, h_offset = 0, w_offset = 0, c_offset = 0;
    int* offset, offs_del;
    switch (axis) {
        case N_AXIS: offset = &n_offset; break;
        case H_AXIS: offset = &h_offset; break;
        case W_AXIS: offset = &w_offset; break;
        case C_AXIS: offset = &c_offset;
    }

    for (int k = 0; k < num_inputs; k++) {
        TensorMem<T>& t = *X[k];
        int N = t.shape.N, H = t.shape.H, W = t.shape.W, C = t.shape.C;
        switch (axis) {
            case N_AXIS: offs_del = N; break;
            case H_AXIS: offs_del = H; break;
            case W_AXIS: offs_del = W; break;
            case C_AXIS: offs_del = C;
        }

        for (int n = 0; n < N; n++)
        for (int h = 0; h < H; h++)
        for (int w = 0; w < W; w++)
        for (int c = 0; c < C; c++) {
            Y.at(n + n_offset, h + h_offset, w + w_offset, c + c_offset) = t.get(n, h, w, c);
        }
        *offset += offs_del;
    }
}
template <typename T>
TensorMem<T>* Concat(TensorMem<T>** X, int num_inputs, int axis) {
    TensorMem<T>* Y;
    Shape shape;
    if (axis == N_AXIS) 
        for (int i = 0; i < num_inputs; i++) shape.N += X[i]->shape.N;
    else shape.N = X[0]->shape.N;
    if (axis == H_AXIS) 
        for (int i = 0; i < num_inputs; i++) shape.H += X[i]->shape.H;
    else shape.H = X[0]->shape.H;
    if (axis == W_AXIS) 
        for (int i = 0; i < num_inputs; i++) shape.W += X[i]->shape.W;
    else shape.W = X[0]->shape.W;
    if (axis == C_AXIS) 
        for (int i = 0; i < num_inputs; i++) shape.C += X[i]->shape.C;
    else shape.C = X[0]->shape.C;

    Y = new TensorMem<T>(shape);
    Concat(X, *Y, num_inputs, axis);
    return Y;
}

template <typename T>
void Gather(TensorMem<T> &X, TensorMem<T> &Y, const int* indices, int idx_count, int axis) {
    assert(axis >= 0 && axis <= 3);
    int N = X.shape.N, H = X.shape.H, W = X.shape.W, C = X.shape.C;
    int n, h, w, c, id, max_idx;
    int* ind_id, *nn = &n, *hh = &h, *ww = &w, *cc = &c;
    switch (axis) {
        case N_AXIS: max_idx = N; N = idx_count; ind_id = &n; nn = &id; break;
        case H_AXIS: max_idx = H; H = idx_count; ind_id = &h; hh = &id; break;
        case W_AXIS: max_idx = W; W = idx_count; ind_id = &w; ww = &id; break;
        case C_AXIS: max_idx = C; C = idx_count; ind_id = &c; cc = &id;
    }

    for (n = 0; n < N; n++) 
    for (h = 0; h < H; h++)
    for (w = 0; w < W; w++)
    for (c = 0; c < C; c++) {
        id = indices[*ind_id];
        assert(id < max_idx && id >= 0);
        Y.at(n, h, w, c) = X.get(*nn, *hh, *ww, *cc);
    }
}
template <typename T>
TensorMem<T>* Gather(TensorMem<T> &X, const int* indices, int idx_count, int axis) {
    TensorMem<T>* Y;
    Shape shape;
    if (axis == N_AXIS) shape.N = idx_count;
    else shape.N = X.shape.N;
    if (axis == H_AXIS) shape.H = idx_count;
    else shape.H = X.shape.H;
    if (axis == W_AXIS) shape.W = idx_count;
    else shape.W = X.shape.W;
    if (axis == C_AXIS) shape.C = idx_count;
    else shape.C = X.shape.C;

    Y = new TensorMem<T>(shape);
    Gather(X, *Y, indices, idx_count, axis);
    return Y;
}

template <typename T>
static void _constant_pad(TensorMem<T> &X, TensorMem<T> &Y, const int* pad, T const_value) {
    int YN = Y.shape.N, YH = Y.shape.H, YW = Y.shape.W, YC = Y.shape.C;
    int XN = X.shape.N, XH = X.shape.H, XW = X.shape.W, XC = X.shape.C;
    for (int n = 0; n < YN; n++)
    for (int h = 0; h < YH; h++)
    for (int w = 0; w < YW; w++)
    for (int c = 0; c < YC; c++) {
        int in = n - pad[0];
        int ih = h - pad[1];
        int iw = w - pad[2];
        int ic = c - pad[3];

        T v = const_value;
        if (in >= 0 && in < XN && ih >= 0 && ih < XH && iw >= 0 && iw < XW && ic >= 0 && ic < XC)
            v = X.get(in, ih, iw, ic);

        Y.at(n, h, w, c) = v;
    }
}
template <typename T>
static void _constant_pad(TensorMem<T> &X, const int* pad, T const_value) {
    int XN = X.shape.N, XH = X.shape.H, XW = X.shape.W, XC = X.shape.C;
    int end_N = X.shape.N - pad[4], end_H = X.shape.H - pad[5], 
        end_W = X.shape.W - pad[6], end_C = X.shape.C - pad[7];
    for (int n = 0; n < XN; n++)
    for (int h = 0; h < XH; h++)
    for (int w = 0; w < XW; w++)
    for (int c = 0; c < XC; c++) {
        if (h < pad[1] || h >= end_H || w < pad[2] || w >= end_W 
            || c < pad[3] || c >= end_C || n < pad[0] || n >= end_N) 
            X.at(n, h, w, c) = const_value;
    }
}
template <typename T>
static void _reflect_pad(TensorMem<T> &X, TensorMem<T> &Y, const int* pad) {
    int YN = Y.shape.N, YH = Y.shape.H, YW = Y.shape.W, YC = Y.shape.C;
    int XN = X.shape.N, XH = X.shape.H, XW = X.shape.W, XC = X.shape.C;
    assert(pad[0] < XN && pad[4] < XN);
    assert(pad[1] < XH && pad[5] < XH);
    assert(pad[2] < XW && pad[6] < XW);
    assert(pad[3] < XC && pad[7] < XC);

    for (int n = 0; n < YN; n++)
    for (int h = 0; h < YH; h++)
    for (int w = 0; w < YW; w++)
    for (int c = 0; c < YC; c++) {
        int in = n - pad[0];
        int ih = h - pad[1];
        int iw = w - pad[2];
        int ic = c - pad[3];

        if (in < 0) in = -in;
        else if (in >= XN) in = 2* XN - in - 2;
        if (ih < 0) ih = -ih;
        else if (ih >= XH) ih = 2* XH - ih - 2;
        if (iw < 0) iw = -iw;
        else if (iw >= XW) iw = 2* XW - iw - 2;
        if (ic < 0) ic = -ic;
        else if (ic >= XC) ic = 2* XC - ic - 2;
        T v = X.get(in, ih, iw, ic);

        Y.at(n, h, w, c) = v;
    }
}
template <typename T>
static void _reflect_pad(TensorMem<T> &X, const int* pad) {
    int XN = X.shape.N, XH = X.shape.H, XW = X.shape.W, XC = X.shape.C;
    int end_N = X.shape.N - pad[4], end_H = X.shape.H - pad[5], 
        end_W = X.shape.W - pad[6], end_C = X.shape.C - pad[7];
    for (int n = 0; n < XN; n++)
    for (int h = 0; h < XH; h++)
    for (int w = 0; w < XW; w++)
    for (int c = 0; c < XC; c++) {
        int in = n, ih = h, iw = w, ic = c;
        if (n < pad[0]) in = pad[0] - in + pad[0];
        else if (n >= end_N) in = end_N - in + end_N - 2;
        if (h < pad[1]) ih = pad[1] - ih + pad[1];
        else if (h >= end_H) ih = end_H - ih + end_H - 2;
        if (w < pad[2]) iw = pad[2] - iw + pad[2];
        else if (w >= end_W) iw = end_W - iw + end_W - 2;
        if (c < pad[3]) ic = pad[3] - ic + pad[3];
        else if (c >= end_C) ic = end_C - ic + end_C - 2;
            X.at(n, h, w, c) = X.get(in, ih, iw, ic);
    }
}
template <typename T>
static void _edge_pad(TensorMem<T> &X, TensorMem<T> &Y, const int* pad) {
    int YN = Y.shape.N, YH = Y.shape.H, YW = Y.shape.W, YC = Y.shape.C;
    int XN = X.shape.N, XH = X.shape.H, XW = X.shape.W, XC = X.shape.C;
    for (int n = 0; n < YN; n++)
    for (int h = 0; h < YH; h++)
    for (int w = 0; w < YW; w++)
    for (int c = 0; c < YC; c++) {
        int in = n - pad[0];
        int ih = h - pad[1];
        int iw = w - pad[2];
        int ic = c - pad[3];

        if (in < 0) in = 0;
        if (in >= XN) in = XN - 1;
        if (ih < 0) ih = 0;
        if (ih >= XH) ih = XH - 1;
        if (iw < 0) iw = 0;
        if (iw >= XW) iw = XW - 1;
        if (ic < 0) ic = 0;
        if (ic >= XC) ic = XC - 1;
        T v = X.get(in, ih, iw, ic);

        Y.at(n, h, w, c) = v;
    }
}
template <typename T>
static void _edge_pad(TensorMem<T> &X, const int* pad) {
    int XN = X.shape.N, XH = X.shape.H, XW = X.shape.W, XC = X.shape.C;
    int end_N = X.shape.N - pad[4], end_H = X.shape.H - pad[5], 
        end_W = X.shape.W - pad[6], end_C = X.shape.C - pad[7];
    for (int n = 0; n < XN; n++)
    for (int h = 0; h < XH; h++)
    for (int w = 0; w < XW; w++)
    for (int c = 0; c < XC; c++) {
        int in = n, ih = h, iw = w, ic = c;
        if (n < pad[0]) in = 0;
        else if (n >= end_N) in = end_N - 1;
        if (h < pad[1]) ih = 0;
        else if (h >= end_H) ih = end_H - 1;
        if (w < pad[2]) iw = 0;
        else if (w >= end_W) iw = end_W - 1;
        if (c < pad[3]) ic = 0;
        else if (c >= end_C) ic = end_C - 1;
            X.at(n, h, w, c) = X.get(in, ih, iw, ic);
    }
}
template <typename T>
void Pad(TensorMem<T> &X, TensorMem<T> &Y, const int* pad, const char* mode, T const_value) {
    for (int i = 0; i < 8; i++) assert(pad[i] >= 0);
    if (!strcmp(mode, "constant")) _constant_pad(X, Y, pad, const_value);
    else if (!strcmp(mode, "reflect")) _reflect_pad(X, Y, pad);
    else if (!strcmp(mode, "edge")) _edge_pad(X, Y, pad);
    else assert(0 && "Error: Pad mode not found !!");
}
template <typename T>
TensorMem<T>* Pad(TensorMem<T> &X, T const_value, const int* pad, const char* mode) {
    TensorMem<T>* Y;
    Shape shape = {X.shape.N + pad[0] + pad[4], X.shape.H + pad[1] + pad[5], 
                    X.shape.W + pad[2] + pad[6], X.shape.C + pad[3] + pad[7]};
    Y = new TensorMem<T>(shape);
    Pad(X, *Y, pad, mode, const_value);
    return Y;
}
template <typename T>
void Pad(TensorMem<T> &X, const int* pad, const char* mode, T const_value) {
    for (int i = 0; i < 8; i++) assert(pad[i] >= 0);
    if (!strcmp(mode, "constant")) _constant_pad(X, pad, const_value);
    else if (!strcmp(mode, "reflect")) _reflect_pad(X, pad);
    else if (!strcmp(mode, "edge")) _edge_pad(X, pad);
    else assert(0 && "Error: Pad mode not found !!");
}

template <typename T>
void Reshape(TensorMem<T> &X, TensorMem<T> &Y) {
    int size = X.shape.N* X.shape.H* X.shape.W* X.shape.C;
    int check_size = Y.shape.N* Y.shape.H* Y.shape.W* Y.shape.C;
    assert(size == check_size);
    for (int i = 0; i < size; i++) 
        Y.raw()[i] = X.raw()[i];
}
template <typename T>
TensorMem<T>* Reshape(TensorMem<T> &X, const Shape &shape) {
    TensorMem<T>* Y = new TensorMem<T>(shape);
    Reshape(X, *Y);
    return Y;
}

template <typename T>
TensorMem<int64_t>* Shapeof(TensorMem<T> &X) {
    return new TensorMem<int64_t>(new int64_t[4]{X.shape.N, X.shape.H, X.shape.W, X.shape.C}, {1, 1, 1, 4}, true);
}

static int absolute(int val) {
    return val >= 0 ? val : -val;
}
template <typename T>
void Slice(TensorMem<T> &X, TensorMem<T> &Y, const Shape &pos_0, const Shape &pos_1, const int* step) {
    assert(step[0] && step[1] && step[2] && step[3]);
    for (int n = pos_0.N, on = 0; step[0] > 0 && n < pos_1.N || step[0] < 0 && n > pos_1.N; n += step[0], on++) 
    for (int h = pos_0.H, oh = 0; step[1] > 0 && h < pos_1.H || step[1] < 0 && h > pos_1.H; h += step[1], oh++) 
    for (int w = pos_0.W, ow = 0; step[2] > 0 && w < pos_1.W || step[2] < 0 && w > pos_1.W; w += step[2], ow++) 
    for (int c = pos_0.C, oc = 0; step[3] > 0 && c < pos_1.C || step[3] < 0 && c > pos_1.C; c += step[3], oc++) 
        Y.at(on, oh, ow, oc) = X.get(n, h, w, c);
}
template <typename T>
TensorMem<T>* Slice(TensorMem<T> &X, const Shape &pos_0, const Shape &pos_1, const int* step) {
    assert(step[0] && step[1] && step[2] && step[3]);
    TensorMem<T>* Y;
    Shape shape;
    shape.N = (absolute(pos_1.N - pos_0.N) - 1) / absolute(step[0]) + 1;
    shape.H = (absolute(pos_1.H - pos_0.H) - 1) / absolute(step[1]) + 1;
    shape.W = (absolute(pos_1.W - pos_0.W) - 1) / absolute(step[2]) + 1;
    shape.C = (absolute(pos_1.C - pos_0.C) - 1) / absolute(step[3]) + 1;
    Y = new TensorMem<T>(shape);
    Slice(X, *Y, pos_0, pos_1, step);
    return Y;
}

template <typename T>
void Transpose(TensorMem<T> &X, TensorMem<T> &Y, int perm[]) {
    int N = X.shape.N, H = X.shape.H, W = X.shape.W, C = X.shape.C;
    int n, h, w, c;
    int* per[4] = {NULL, NULL, NULL, NULL};
    for (int i = 0; i < 4; i++) 
        switch (perm[i]) {
            case N_AXIS: per[i] = &n; break;
            case H_AXIS: per[i] = &h; break;
            case W_AXIS: per[i] = &w; break;
            case C_AXIS: per[i] = &c;
        }
    for (n = 0; n < N; n++)
    for (h = 0; h < H; h++)
    for (w = 0; w < W; w++)
    for (c = 0; c < C; c++) 
        Y.at(*per[0], *per[1], *per[2], *per[3]) = X.get(n, h, w, c);
}
template <typename T>
TensorMem<T>* Transpose(TensorMem<T> &X, int perm[]) {
    TensorMem<T>* Y;
    Shape shape;
    int x_shape[4] = {X.shape.N, X.shape.H, X.shape.W, X.shape.C};
    shape.N = x_shape[perm[0]];
    shape.H = x_shape[perm[1]];
    shape.W = x_shape[perm[2]];
    shape.C = x_shape[perm[3]];
    Y = new TensorMem<T>(shape);
    Transpose(X, *Y, perm);
    return Y;
}
