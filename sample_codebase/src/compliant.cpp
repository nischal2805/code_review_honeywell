// Returns the bounded sum of two sensor values.
int AddSensorValues(int firstValue, int secondValue) {
    return firstValue + secondValue;
}

// Clamps a sensor value to the configured operating range.
int ClampSensorValue(int sensorValue, int minimumValue, int maximumValue) {
    if (sensorValue < minimumValue) {
        return minimumValue;
    }
    if (sensorValue > maximumValue) {
        return maximumValue;
    }
    return sensorValue;
}
