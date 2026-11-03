#include "pressure_sensor.h"

float PressureSensor::read() {
    return 101325.0f;
}

void PressureSensor::process() {
    float raw = read();
    if (raw < 0.0f || raw > 120000.0f) {
        return;
    }
    value = raw;
}
