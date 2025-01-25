#include "../exercise.h"

// READ: 静态字段 <https://zh.cppreference.com/w/cpp/language/static>
// READ: 虚析构函数 <https://zh.cppreference.com/w/cpp/language/destructor>

struct A {
    static int num_a;

    A() {
        ++num_a;
    }
    virtual ~A() { // 添加虚析构函数
        --num_a;
    }

    virtual char name() const {
        return 'A';
    }
};
struct B final : public A {
    static int num_b;

    B() {
        ++num_b;
    }
    ~B() { // 添加虚析构函数
        --num_b;
    }

    char name() const final {
        return 'B';
    }
};

int A::num_a = 0;  // 正确初始化静态字段 num_a
int B::num_b = 0;  // 正确初始化静态字段 num_b

int main(int argc, char **argv) {
    auto a = new A;
    auto b = new B;
    
    // 等到对象创建后再检查静态字段
    ASSERT(A::num_a == 2, "Fill in the correct value for A::num_a");
    ASSERT(B::num_b == 1, "Fill in the correct value for B::num_b");
    ASSERT(a->name() == 'A', "Fill in the correct value for a->name()");
    ASSERT(b->name() == 'B', "Fill in the correct value for b->name()");

    delete a;
    delete b;
    ASSERT(A::num_a == 0, "Every A was destroyed");
    ASSERT(B::num_b == 0, "Every B was destroyed");

    A* ab = new B; // 派生类指针可以随意转换为基类指针
    ASSERT(A::num_a == 1, "Fill in the correct value for A::num_a");
    ASSERT(B::num_b == 1, "Fill in the correct value for B::num_b");
    ASSERT(ab->name() == 'B', "Fill in the correct value for ab->name()");

    // 正确转换语句
    B* bb = dynamic_cast<B*>(ab); // 使用 dynamic_cast 进行安全的类型转换
    ASSERT(bb->name() == 'B', "Fill in the correct value for bb->name()");

    // ---- 以下代码不要修改，通过改正类定义解决编译问题 ----
    delete ab; // 通过指针可以删除指向的对象，即使是多态对象
    ASSERT(A::num_a == 0, "Every A was destroyed");
    ASSERT(B::num_b == 0, "Every B was destroyed");

    return 0;
}
