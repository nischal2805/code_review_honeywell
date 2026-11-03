#ifndef PRESSURE_SENSOR_H
#define PRESSURE_SENSOR_H
#include "sensor.h"

class PressureSensor : public Sensor {
public:
    void process() override;
    float read() override;
};

#endif
