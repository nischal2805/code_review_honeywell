# Virtual Analysis Report — DO-178C §12
Generated: 2026-05-25
DAL Level: B

## Summary
- Base virtual count:    2
- Current virtual count: 3
- Added:    1
- Removed:  0
- Modified: 1
- Unchanged: 1

## Changes

### brake
- Change Type: ADDED
- DO-178C Category: Category 2
- File: tests\fixtures\cpp\current_build\vehicle.cpp:19
- Current Signature: `void brake()`
- Reverification Scope: Full reverification required — new virtual function brake

### getSpeed
- Change Type: UNCHANGED
- DO-178C Category: Category 1
- File: tests\fixtures\cpp\current_build\vehicle.cpp:15
- Base Signature: `int getSpeed()`
- Current Signature: `int getSpeed()`
- Reverification Scope: No reverification required

### start
- Change Type: MODIFIED
- DO-178C Category: Category 2
- File: tests\fixtures\cpp\current_build\vehicle.cpp:11
- Base Signature: `void start()`
- Current Signature: `void start(int initial_speed)`
- Reverification Scope: Signature/implementation changed — reverify callers of start

