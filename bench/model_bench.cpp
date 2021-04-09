#include <RTNeural.h>
#include <string>
#include <tuple>
#include <iostream>

class A {
public:
    A (int x, int y)
    {
        ax = x;
        ay = y;
        std::cout << "Constructing A with x=" << x << std::endl;

        tt = new float[4];
        tt[0] = 100.0f;
    }

    ~A()
    {
        std::cout << "Destructing A with x=" << ax << std::endl;
        tt[0] = 0.0f;
        delete[] tt;
    }

    A(const A& other) : A(other.ax, other.ay)
    {
        tt = new float[4];
        std::copy (other.tt, other.tt + 4, tt);
    }

    A(A&& other) noexcept :
        ax(other.ax),
        ay (other.ay),
        tt (std::exchange(other.tt, nullptr))
    {
    }

    int ax;
    int ay;
    float* tt;
};

class B {
public:
    B (int x, int y)
    {
        ax = x;
        ay = y;
        std::cout << "Constructing B with x=" << x << std::endl;
    }

    int ax;
    int ay;
};

/* Forward declaration. */
template <typename... T>
struct FuncImpl;

/* Base case. */
template <typename T>
struct FuncImpl<T> {
    using Liter = std::initializer_list<int>::const_iterator;
    std::tuple<T> operator()(Liter begin, Liter /*end*/) const
    {
        std::cout << "Base case" << std::endl;
        return std::tuple<T> (T (*begin, *(begin + 1)));
    }
};  // FuncImpl<>

/* Recursive case. */
template <typename First, typename... Rest>
struct FuncImpl<First, Rest...> {
    using Liter = std::initializer_list<int>::const_iterator;
    std::tuple<First, Rest...> operator()(Liter begin, Liter end) const
    {
        std::cout << "Recursive case" << std::endl;
        return std::tuple_cat(FuncImpl<First>()(begin, begin + 1),
                              FuncImpl<Rest...>()(begin + 1, end));
    }
};  // FuncImpl<First, Rest...>

/* Delegate function. */
template <typename... T>
std::tuple<T...> Func(std::initializer_list<int> l)
{
    return FuncImpl<T...>()(l.begin(), l.end());
}

std::tuple<A> createA(int x, int y)
{
    // A a (x, y);
    return std::forward_as_tuple (A { x, y });
}

int main (int argc, char* argv[])
{
    // auto aa = createA(1, 2);
    // std::cout << std::get<0>(aa).ax << std::endl;
    // std::cout << std::get<0>(aa).tt[0] << std::endl;
    // return 0;

    // std::tuple<A, B, A> test_tuple = Func<A, B, A>({1, 2, 3, 4});
    // std::cout << "Done construction!" << std::endl;

    // std::cout << std::get<0> (test_tuple).ax << std::endl;
    // std::cout << std::get<0> (test_tuple).ay << std::endl;
    // std::cout << std::get<1> (test_tuple).ax << std::endl;
    // std::cout << std::get<1> (test_tuple).ay << std::endl;
    // std::cout << std::get<2> (test_tuple).ax << std::endl;
    // std::cout << std::get<2> (test_tuple).ay << std::endl;

    RTNeural::ModelT<float,
        RTNeural::Dense<float>,
        RTNeural::Dense<float>> model ({2, 4, 1});

    // model.reset();
    
    // float x[] = { 2.0f, 4.0f };
    // model.forward (x);
}
