#ifndef TEMPERATURE_SENSOR_H
#define TEMPERATURE_SENSOR_H
#include "sensor.h"

const unsigned long TEMP_STALE_MS = 500;
const float TEMP_MIN_C = -54.0f;
const float TEMP_MAX_C = 54.0f;
const float TEMP_DEBOUNCE_C = 2.0f;

class TemperatureSensor : public Sensor {
public:
    void process() override;
    float read() override;
    bool isStale(unsigned long nowMs) const;
    bool inRange(float v) const;
    float debounce(float v);
};

#endif
