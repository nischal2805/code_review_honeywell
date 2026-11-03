#include "validator.h"

bool Validator::check(float v) {
    return v > -100.0f && v < 100.0f;
}
