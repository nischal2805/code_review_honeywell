#include <cmath>
#include "temperature_sensor.h"

float TemperatureSensor::read() {
    return 25.0f;
}

bool TemperatureSensor::isStale(unsigned long nowMs) const {
    return (nowMs - lastUpdateMs) > TEMP_STALE_MS;
}

bool TemperatureSensor::inRange(float v) const {
    return v >= TEMP_MIN_C && v <= TEMP_MAX_C;
}

float TemperatureSensor::debounce(float v) {
    if (std::fabs(v - lastValue) < TEMP_DEBOUNCE_C) {
        return lastValue;
    }
    return v;
}

void TemperatureSensor::process() {
    float raw = read();
    if (!inRange(raw)) {
        return;
    }
    if (isStale(lastUpdateMs + 100)) {
        return;
    }
    lastValue = value;
    value = debounce(raw);
    lastUpdateMs = lastUpdateMs + 100;
}
