#ifndef SENSOR_H
#define SENSOR_H

// Base class for every sensor.
class Sensor {
public:
    virtual ~Sensor() {}
    // DO-178C-REQ: SRS-SNS-001
    virtual void process();
    virtual float read() = 0;
    virtual bool selfTest() { return true; }
    float getValue() const { return value; }

protected:
    float value = 0.0f;
    float lastValue = 0.0f;
    unsigned long lastUpdateMs = 0;
};

#endif
