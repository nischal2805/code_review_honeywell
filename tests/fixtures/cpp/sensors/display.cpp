#include "display.h"

void Display::update(float v) {
    shown = v;
}

void Display::clear(int mode) {
    shown = 0.0f;
}
