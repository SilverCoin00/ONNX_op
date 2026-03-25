#include "..\include\Core.h"


template <typename T>
void Add(TensorMem<T> &X1, TensorMem<T> &X2, TensorMem<T> &Y) {
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
    for (n1 = 0, n2 = 0; n1 < Y.shape.N && n2 < Y.shape.N; n1 += dn1, n2 += dn2)
    for (h1 = 0, h2 = 0; h1 < Y.shape.H && h2 < Y.shape.H; h1 += dh1, h2 += dh2)
    for (w1 = 0, w2 = 0; w1 < Y.shape.W && w2 < Y.shape.W; w1 += dw1, w2 += dw2)
    for (c1 = 0, c2 = 0; c1 < Y.shape.C && c2 < Y.shape.C; c1 += dc1, c2 += dc2) 
        Y.at(*yn, *yh, *yw, *yc) = X1.get(n1, h1, w1, c1) + X2.get(n2, h2, w2, c2);
}
template <typename T>
void Add(TensorMem<T> &X1, TensorMem<T> &X2, TensorMem<T> &Y, Shape &start, Shape &end) {
    assert(X1.shape == Y.shape || X2.shape == Y.shape);
    int dn1, dn2, dh1, dh2, dw1, dw2, dc1, dc2;
    dn1 = dn2 = dh1 = dh2 = dw1 = dw2 = dc1 = dc2 = 1;
    int n1, n2, h1, h2, w1, w2, c1, c2;
    n1 = n2 = start.N, h1 = h2 = start.H, w1 = w2 = start.W, c1 = c2 = start.C;
    int* yn = &n1, *yh = &h1, *yw = &w1, *yc = &c1;
    if (X1.shape.N != X2.shape.N) {
        if (X1.shape.N == 1) dn1--, yn = &n2, n1 = 0;
        else if (X2.shape.N == 1) dn2--, n2 = 0;
        else assert(0 && "Error: Inappropriate N-axis size !!");
    }
    if (X1.shape.H != X2.shape.H) {
        if (X1.shape.H == 1) dh1--, yh = &h2, h1 = 0;
        else if (X2.shape.H == 1) dh2--, h2 = 0;
        else assert(0 && "Error: Inappropriate H-axis size !!");
    }
    if (X1.shape.W != X2.shape.W) {
        if (X1.shape.W == 1) dw1--, yw = &w2, w1 = 0;
        else if (X2.shape.W == 1) dw2--, w2 = 0;
        else assert(0 && "Error: Inappropriate W-axis size !!");
    }
    if (X1.shape.C != X2.shape.C) {
        if (X1.shape.C == 1) dc1--, yc = &c2, c1 = 0;
        else if (X2.shape.C == 1) dc2--, c2 = 0;
        else assert(0 && "Error: Inappropriate C-axis size !!");
    }
    for ( ; n1 < end.N && n2 < end.N; n1 += dn1, n2 += dn2)
    for ( ; h1 < end.H && h2 < end.H; h1 += dh1, h2 += dh2)
    for ( ; w1 < end.W && w2 < end.W; w1 += dw1, w2 += dw2)
    for ( ; c1 < end.C && c2 < end.C; c1 += dc1, c2 += dc2) 
        Y.at(*yn, *yh, *yw, *yc) = X1.get(n1, h1, w1, c1) + X2.get(n2, h2, w2, c2);
}
template <typename T>
TensorMem<T>* Add(TensorMem<T> &X1, TensorMem<T> &X2) {
    TensorMem<T>* Y;
    if (X1.shape >= X2.shape) Y = new TensorMem<T>(X1.shape);
    else Y = new TensorMem<T>(X2.shape);
    Add(X1, X2, *Y);
    return Y;
}

template <typename T>
void Sub(TensorMem<T> &X1, TensorMem<T> &X2, TensorMem<T> &Y) {
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
    for (n1 = 0, n2 = 0; n1 < Y.shape.N && n2 < Y.shape.N; n1 += dn1, n2 += dn2)
    for (h1 = 0, h2 = 0; h1 < Y.shape.H && h2 < Y.shape.H; h1 += dh1, h2 += dh2)
    for (w1 = 0, w2 = 0; w1 < Y.shape.W && w2 < Y.shape.W; w1 += dw1, w2 += dw2)
    for (c1 = 0, c2 = 0; c1 < Y.shape.C && c2 < Y.shape.C; c1 += dc1, c2 += dc2) 
        Y.at(*yn, *yh, *yw, *yc) = X1.get(n1, h1, w1, c1) - X2.get(n2, h2, w2, c2);
}
template <typename T>
TensorMem<T>* Sub(TensorMem<T> &X1, TensorMem<T> &X2) {
    TensorMem<T>* Y;
    if (X1.shape >= X2.shape) Y = new TensorMem<T>(X1.shape);
    else Y = new TensorMem<T>(X2.shape);
    Sub(X1, X2, *Y);
    return Y;
}

template <typename T>
void Mul(TensorMem<T> &X1, TensorMem<T> &X2, TensorMem<T> &Y) {
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
    for (n1 = 0, n2 = 0; n1 < Y.shape.N && n2 < Y.shape.N; n1 += dn1, n2 += dn2)
    for (h1 = 0, h2 = 0; h1 < Y.shape.H && h2 < Y.shape.H; h1 += dh1, h2 += dh2)
    for (w1 = 0, w2 = 0; w1 < Y.shape.W && w2 < Y.shape.W; w1 += dw1, w2 += dw2)
    for (c1 = 0, c2 = 0; c1 < Y.shape.C && c2 < Y.shape.C; c1 += dc1, c2 += dc2) 
        Y.at(*yn, *yh, *yw, *yc) = X1.get(n1, h1, w1, c1)* X2.get(n2, h2, w2, c2);
}
template <typename T>
TensorMem<T>* Mul(TensorMem<T> &X1, TensorMem<T> &X2) {
    TensorMem<T>* Y;
    if (X1.shape >= X2.shape) Y = new TensorMem<T>(X1.shape);
    else Y = new TensorMem<T>(X2.shape);
    Mul(X1, X2, *Y);
    return Y;
}

template <typename T>
void Div(TensorMem<T> &X1, TensorMem<T> &X2, TensorMem<T> &Y) {
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
    for (n1 = 0, n2 = 0; n1 < Y.shape.N && n2 < Y.shape.N; n1 += dn1, n2 += dn2)
    for (h1 = 0, h2 = 0; h1 < Y.shape.H && h2 < Y.shape.H; h1 += dh1, h2 += dh2)
    for (w1 = 0, w2 = 0; w1 < Y.shape.W && w2 < Y.shape.W; w1 += dw1, w2 += dw2)
    for (c1 = 0, c2 = 0; c1 < Y.shape.C && c2 < Y.shape.C; c1 += dc1, c2 += dc2) {
        T val = X2.get(n2, h2, w2, c2);
        if (abs(static_cast<float>(val)) < 1e-7) val = static_cast<T>(copysign(1e-7, val));
        Y.at(*yn, *yh, *yw, *yc) = X1.get(n1, h1, w1, c1) / val;
    }
}
template <typename T>
TensorMem<T>* Div(TensorMem<T> &X1, TensorMem<T> &X2) {
    TensorMem<T>* Y;
    if (X1.shape >= X2.shape) Y = new TensorMem<T>(X1.shape);
    else Y = new TensorMem<T>(X2.shape);
    Div(X1, X2, *Y);
    return Y;
}
