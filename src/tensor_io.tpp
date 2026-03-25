#include "..\include\Core.h"
#include <iomanip>
#include <fstream>

template<typename T>
void read_tensor(const char* file_path, TensorMem<T> &tens) {
    std::ifstream file(file_path);
    int size = tens.shape.N* tens.shape.H* tens.shape.W* tens.shape.C;
    float val;
    int i = 0;
    while (file >> val && i < size) {
        tens.raw()[i++] = static_cast<T>(val);
    }
}
template<typename T>
void write_tensor(const char* file_path, TensorMem<T> &X, unsigned int precision) {
    std::ofstream file(file_path);
    int size = X.shape.N* X.shape.H* X.shape.W* X.shape.C;
    float val;
    for (int i = 0; i < size; i++) {
        val = (float)X.raw()[i];
        file << std::fixed << std::setprecision(precision) << val << std::endl;
    }
    file.close();
}
