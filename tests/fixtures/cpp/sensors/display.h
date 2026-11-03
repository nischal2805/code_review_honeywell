#ifndef DISPLAY_H
#define DISPLAY_H

class Display {
public:
    void update(float v);
    void clear(int mode);

private:
    float shown = 0.0f;
};

#endif
