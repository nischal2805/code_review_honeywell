#include "sensor.h"

void Sensor::process() {
    lastValue = value;
    value = read();
}
