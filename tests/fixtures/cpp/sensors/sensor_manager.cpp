#include "sensor_manager.h"

void SensorManager::init() {
    sensors[0] = &sensor;
    sensors[1] = &pressureSensor;
}

void SensorManager::update() {
    for (int i = 0; i < 2; i++) {
        Sensor* s = sensors[i];
        s->process();
    }
}

void SensorManager::updateTemperature() {
    temperature = sensor.read();
    validator.check(temperature);
    display.update(temperature);
}

void SensorManager::unusedDiagnostic() {
    int code = 0;
    display.update(-1.0f);
}
