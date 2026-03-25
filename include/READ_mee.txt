Instruction:

Tensor:
        Shape() : N(0), H(0), W(0), C(0)
        Shape(int n, int h, int w, int c) : N(n), H(h), W(w), C(c)

        template <typename T>
        class TensorMem {
            Shape shape;
            TensorMem();
            TensorMem(T* data, const Shape &shape, bool own_memory);
            TensorMem(const Shape &shape);

            void print();

            inline T get(int n, int h, int w, int c);
            inline T &at(int n, int h, int w, int c);

            inline T* raw() { return data; }
            inline const T* raw() const { return data; }

            void load_tile_to_stream(const Shape &start, const Shape &size, MyStream<T>& out_stream);
            void store_stream_to_mem(const Shape &start, const Shape &size, MyStream<T>& in_stream);
        };



Convolution:
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
            int kernel_shape[2];       // [H, W]
            int output_padding[2];     // [bottom, right]
            int pads[4];               // [top, left, bottom, right]
            int strides[2];            // [H, W]
        };

    Conv:
        TensorMem<float>* Conv(Conv_Attributes &attributes, TensorMem<float>* X, TensorMem<float>* W, TensorMem<float>* B);
        @example: Y = Conv({{1, 1}, 1, {3, 3}, {0, 0, 0, 0}, {1, 1}}, X, W, B)

    ConvTranspose:
        TensorMem<float>* ConvTranspose(ConvTranspose_Attributes &attributes, TensorMem<float>* X, TensorMem<float>* W, TensorMem<float>* B);
        @example: Y = Conv({{1, 1}, 1, {3, 3}, {0, 0, 0, 0}, {1, 1}}, X, W, B)



Mem mainpulation:
    Concat:
            void Concat(TensorMem<T>** X, TensorMem<T> &Y, int num_inputs, int axis);
            TensorMem<T>* Concat(TensorMem<T>** X, int num_inputs, int axis);

    Gather:
            void Gather(TensorMem<T> &X, TensorMem<T> &Y, const int* indices, int idx_count, int axis);
            TensorMem<T>* Gather(TensorMem<T> &X, const int* indices, int idx_count, int axis);

    Identity:
            void Identity(const TensorMem<T> &X, TensorMem<T> &Y);
            TensorMem<T>* Identity(const TensorMem<T> &X);

    Pad:
            void Pad(TensorMem<T> &X, TensorMem<T> &Y, const int* pad, const char* mode, T const_value);
            TensorMem<T>* Pad(TensorMem<T> &X, const int* pad, const char* mode, T const_value);

    Reshape:
            void Reshape(TensorMem<T> &X, TensorMem<T> &Y);
            TensorMem<T>* Reshape(TensorMem<T> &X, const Shape &shape);

    Shapeof:
            TensorMem<int64_t>* Shapeof(TensorMem<T> &X);

    Slice:
            void Slice(TensorMem<T> &X, TensorMem<T> &Y, const Shape &pos_0, const Shape &pos_1, const int* step);
            TensorMem<T>* Slice(TensorMem<T> &X, const Shape &pos_0, const Shape &pos_1, const int* step);

    Transpose:
            void Transpose(TensorMem<T> &X, TensorMem<T> &Y, int perm[]);
            TensorMem<T>* Transpose(TensorMem<T> &X, int perm[]);



Constant:
            void Constant(const T* X, TensorMem<T> &Y, int size);
            void Constant_of_shape(TensorMem<T> &Y, T val);
            TensorMem<T>* Constant_of_shape(const Shape &shape, T val);



Unary:
    Cast:
        void Cast(TensorMem<T_IN> &X, TensorMem<T_OUT> &Y);
        auto Cast(TensorMem<T_IN> &X) -> TensorMem<T_OUT>*;

    Relu:
        void Relu(TensorMem<T> &X, TensorMem<T> &Y);
        TensorMem<T>* Relu(TensorMem<T> &X);

    Sqrt:
        void Sqrt(TensorMem<float> &X, TensorMem<float> &Y);
        TensorMem<float>* Sqrt(TensorMem<float> &X);

    Clip:
        void Clip(TensorMem<T> &X, TensorMem<T> &Y, T min, T max);
        TensorMem<T>* Clip(TensorMem<T> &X, T min, T max);

    Floor:
        void Floor(TensorMem<T> &X, TensorMem<T> &Y);
        TensorMem<T>* Floor(TensorMem<T> &X);



Arithmatic:
    Add:
        void Add(TensorMem<T> &X1, TensorMem<T> &X2, TensorMem<T> &Y);
        TensorMem<T>* Add(TensorMem<T> &X1, TensorMem<T> &X2);

    Sub:
        void Sub(TensorMem<T> &X1, TensorMem<T> &X2, TensorMem<T> &Y);
        TensorMem<T>* Sub(TensorMem<T> &X1, TensorMem<T> &X2);

    Mul:
        void Mul(TensorMem<T> &X1, TensorMem<T> &X2, TensorMem<T> &Y);
        TensorMem<T>* Mul(TensorMem<T> &X1, TensorMem<T> &X2);

    Div:
        void Div(TensorMem<T> &X1, TensorMem<T> &X2, TensorMem<T> &Y);
        TensorMem<T>* Div(TensorMem<T> &X1, TensorMem<T> &X2);



Reduce:
    ReduceMean:
        void ReduceMean(TensorMem<T> &X, TensorMem<T> &Y, int axis);
        TensorMem<T>* ReduceMean(TensorMem<T> &X, int axis);
    ReduceProd:
        void ReduceProd(TensorMem<T> &X, TensorMem<T> &Y, int axis);
        TensorMem<T>* ReduceProd(TensorMem<T> &X, int axis);
