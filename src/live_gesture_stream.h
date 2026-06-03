#pragma once
#include "tflite_gesture_classifier.h"
#include <functional>

void RunLiveStream(
    TfliteGestureClassifier* classifier,
    std::function<void(int gesture_index)> on_gesture_detected
);