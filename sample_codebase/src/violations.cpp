#include <stdexcept>

void bad_name(int a, int b, int c, int d, int e, int f, int g, int h) {
    int* value = new int(a);
    if (a > 0) { b++; }
    if (b > 0) { c++; }
    if (c > 0) { d++; }
    if (d > 0) { e++; }
    if (e > 0) { f++; }
    if (f > 0) { g++; }
    if (g > 0) { h++; }
    if (h > 0) { goto done; }
    while (true) { break; }
    done:
    delete value;
}

void ThrowError() {
    try {
        throw std::runtime_error("failure");
    } catch (...) {
    }
}

int CountDown(int value) {
    if (value <= 0) {
        return 0;
    }
    return CountDown(value - 1);
}

void UseRtti(void* object) {
    (void)dynamic_cast<int*>(static_cast<int*>(object));
}
