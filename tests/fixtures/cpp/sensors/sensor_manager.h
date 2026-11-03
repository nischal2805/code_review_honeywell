#ifndef SENSOR_MANAGER_H
#define SENSOR_MANAGER_H
#include "temperature_sensor.h"
#include "pressure_sensor.h"
#include "validator.h"
#include "display.h"

class SensorManager {
public:
    void init();
    void update();
    void updateTemperature();
    void unusedDiagnostic();

private:
    TemperatureSensor sensor;
    PressureSensor pressureSensor;
    Validator validator;
    Display display;
    float temperature = 0.0f;
    Sensor* sensors[2];
};

#endif
