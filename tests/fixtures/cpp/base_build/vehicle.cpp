#include <string>

// Dead function — never called
void orphanFunc() {
    int y = 2;
}

// DO-178C-REQ: REQ-001
class Vehicle {
public:
    virtual void start() {
        speed = 0;
    }

    virtual int getSpeed() const {
        return speed;
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
