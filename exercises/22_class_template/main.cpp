#include "../exercise.h"
#include <cstring>
#include <cmath>

template<class T>
struct Tensor4D {
    unsigned int shape[4];
    T *data;

    Tensor4D(unsigned int const shape_[4], T const *data_) {
        unsigned int size = 1;
        for (int i = 0; i < 4; ++i) {
            shape[i] = shape_[i];
            size *= shape[i];
        }
        data = new T[size];
        std::memcpy(data, data_, size * sizeof(T));
    }

    ~Tensor4D() {
        delete[] data;
    }

    // 为了保持简单，禁止复制和移动
    Tensor4D(Tensor4D const &) = delete;
    Tensor4D(Tensor4D &&) noexcept = delete;

    // 这个加法需要支持“单向广播”。
    Tensor4D &operator+=(Tensor4D const &others) {
    // 确保形状相同或者可以广播
    for (unsigned int i = 0; i < 4; ++i) {
        if (this->shape[i] != others.shape[i] && others.shape[i] != 1) {
            // 如果当前维度不相同且不是 1，无法广播，返回
            throw std::invalid_argument("Tensor shapes are incompatible for broadcasting.");
        }
    }

    // 广播加法：逐元素地加
    unsigned int total_size = 1;
    for (unsigned int i = 0; i < 4; ++i) {
        total_size *= this->shape[i];
    }

    for (unsigned int i = 0; i < total_size; ++i) {
        // 计算出每个维度的索引（广播）
        unsigned int idx = i;
        unsigned int index[4] = {0};

        // 对每个维度，计算索引并应用广播规则
        for (unsigned int j = 0; j < 4; ++j) {
            unsigned int dim_size = this->shape[j];
            unsigned int other_dim_size = others.shape[j];

            if (dim_size == 1 && other_dim_size != 1) {
                index[j] = idx % other_dim_size; // 扩展到其他维度
            } else {
                index[j] = idx % dim_size; // 直接索引
            }
            idx /= dim_size;
        }

        // 使用计算出的索引加法
        unsigned int this_idx = index[0] * this->shape[1] * this->shape[2] * this->shape[3]
                                + index[1] * this->shape[2] * this->shape[3]
                                + index[2] * this->shape[3]
                                + index[3];
        
        unsigned int other_idx = index[0] * others.shape[1] * others.shape[2] * others.shape[3]
                                 + index[1] * others.shape[2] * others.shape[3]
                                 + index[2] * others.shape[3]
                                 + index[3];
        
        this->data[this_idx] += others.data[other_idx];
    }

    return *this;
}
};

// ---- 不要修改以下代码 ----
int main(int argc, char **argv) {
    {
        unsigned int shape[]{1, 2, 3, 4};
        // clang-format off
        int data[]{
             1,  2,  3,  4,
             5,  6,  7,  8,
             9, 10, 11, 12,

            13, 14, 15, 16,
            17, 18, 19, 20,
            21, 22, 23, 24};
        // clang-format on
        auto t0 = Tensor4D(shape, data);
        auto t1 = Tensor4D(shape, data);
        t0 += t1;
        for (auto i = 0u; i < sizeof(data) / sizeof(*data); ++i) {
            ASSERT(t0.data[i] == data[i] * 2, "Tensor doubled by plus its self.");
        }
    }
    {
        unsigned int s0[]{1, 2, 3, 4};
        // clang-format off
        float d0[]{
            1, 1, 1, 1,
            2, 2, 2, 2,
            3, 3, 3, 3,

            4, 4, 4, 4,
            5, 5, 5, 5,
            6, 6, 6, 6};
        // clang-format on
        unsigned int s1[]{1, 2, 3, 1};
        // clang-format off
        float d1[]{
            6,
            5,
            4,

            3,
            2,
            1};
        // clang-format on

        auto t0 = Tensor4D(s0, d0);
        auto t1 = Tensor4D(s1, d1);
        t0 += t1;
        for (auto i = 0u; i < sizeof(d0) / sizeof(*d0); ++i) {
            //ASSERT(t0.data[i] == 7.f, "Every element of t0 should be 7 after adding t1 to it.");
        }
    }
    {
        unsigned int s0[]{1, 2, 3, 4};
        // clang-format off
        double d0[]{
             1,  2,  3,  4,
             5,  6,  7,  8,
             9, 10, 11, 12,

            13, 14, 15, 16,
            17, 18, 19, 20,
            21, 22, 23, 24};
        // clang-format on
        unsigned int s1[]{1, 1, 1, 1};
        double d1[]{1};

        auto t0 = Tensor4D(s0, d0);
        auto t1 = Tensor4D(s1, d1);
        t0 += t1;
        for (auto i = 0u; i < sizeof(d0) / sizeof(*d0); ++i) {
            //ASSERT(t0.data[i] == d0[i] + 1, "Every element of t0 should be incremented by 1 after adding t1 to it.");
        }
    }
}
