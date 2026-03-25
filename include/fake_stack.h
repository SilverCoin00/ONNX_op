#include <inttypes.h>
#define byte char


struct Arena {
    byte* base;
    size_t size;
    size_t max_runtime_size;
    size_t offset;
    struct List {
        size_t data[30];
        int top;
    } manage;

    Arena(void* mem, size_t sz) : base((byte*)mem), size(sz), offset(0) {
        manage.top = -1;
        max_runtime_size = 0;
    }

    void reset() {
        offset = 0;
        manage.top = -1;
    }
    void pop() {
        if (manage.top < 0) 
            assert(0 && "Error: Fake stack underflow !!\n");
        offset = manage.data[manage.top--];
    }

    template<typename T>
    T* alloc(size_t count) {
        constexpr size_t align = alignof(T);
        uintptr_t current = (uintptr_t)(base + offset);
        uintptr_t aligned = (current + align - 1) & ~(align - 1);

        size_t new_offset = (aligned - (uintptr_t)base) + count* sizeof(T);

        if (new_offset > size || manage.top >= 29)
            assert(0 && "Error: Fake stack overflow !!\n");

        manage.data[++manage.top] = offset;
        offset = new_offset;

        if (offset > max_runtime_size) max_runtime_size = offset;

        return (T*)aligned;
    }
};
