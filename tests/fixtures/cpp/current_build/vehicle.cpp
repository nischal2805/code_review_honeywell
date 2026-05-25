#include <string>

// Dead function — never called
void orphanFunc() {
    int y = 2;
}

// DO-178C-REQ: REQ-001
class Vehicle {
public:
    virtual void start(int initial_speed) {
        speed = initial_speed;
    }

    virtual int getSpeed() const {
        return speed;
    }

    virtual void brake() {
        if (speed > 0) {
            speed -= 10;
        }
    }

    void stop() {
        speed = 0;
    }

private:
    int speed = 0;
};

void utilityFunc() {
    int x = 1;
}
