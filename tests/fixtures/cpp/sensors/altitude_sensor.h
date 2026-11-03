#ifndef ALTITUDE_SENSOR_H
#define ALTITUDE_SENSOR_H
#include "sensor.h"

// Not instantiated anywhere in this build.
class AltitudeSensor : public Sensor {
public:
    float read() override { return 1500.0f; }
    bool selfTest() override { return false; }
};

#endif
