#include "sensor_manager.h"

static int cycleCount = 0;

typedef void (*Callback)();

void registerCallback(Callback cb) {
    cb();
}

void onTimerTick() {
    cycleCount++;
}

// Interrupt handler: invoked by hardware, never called from code.
void TIMER0_IRQHandler() {
    cycleCount = 0;
}

void legacyCalibrate() {
    cycleCount = -1;
}

void orphanHelper() {
    int unused = 5;
}

int computeChecksum(int x) {
    return x * 31;
    x = 0;
}

int main() {
    SensorManager mgr;
    mgr.init();
    registerCallback(onTimerTick);
    while (cycleCount < 10) {
        mgr.update();
        mgr.updateTemperature();
        cycleCount++;
    }
    if (0) {
        legacyCalibrate();
    }
    return computeChecksum(cycleCount);
}

#if 0
void disabledFeature() {
    cycleCount = 99;
}
#endif
