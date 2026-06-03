#include "live_gesture_stream.h"
#include <array>
#include <deque>
#include <algorithm>
#include <iostream>

#include "sense_hat_imu.h"
#include <thread>

constexpr int kStride = 10; // Run inference every N samples to reduce CPU load.
constexpr int kVoteWindow = 5; // Number of recent predictions to consider for majority voting.
constexpr int kVoteThreshold = 3; // Minimum votes in the window to confirm a gesture (e.g. 3 out of 5).
constexpr int kCooldownSamples = 30; // to avoid overlapping gestures

void RunLiveStream(
    TfliteGestureClassifier* classifier,
    std::function<void(int)> on_gesture_detected)
{
    // circular buffer holding the last 50 IMU samples (each with the 9 features)
    std::deque<std::array<float, kGestureFeatures>> buffer;
    // recent predictions for majoirty voting - stabilizing
    std::deque<int> votes;
    int stride_counter  = 0;
    int cooldown        = 0;

    SenseHatImu imu;
    if (!imu.available()) {
        std::cerr << "IMU unavailable: " << imu.error_message() << "\n";
        return;
    }
    
    // Called every IMU reading, with the latest sample. Runs inference every kStride samples and calls on_gesture_detected when a gesture is detected.
    auto on_sample = [&](std::array<float, kGestureFeatures> sample) {
        // appends new samples - gets rid of the oldest if the window is full
        buffer.push_back(sample);
        if (buffer.size() > kGestureTimesteps)
            buffer.pop_front();

        // only run inference every kStride samples to reduce CPU load 
        if (++stride_counter < kStride) return;
        stride_counter = 0;

        // only run if full sample window is there
        if ((int)buffer.size() < kGestureTimesteps) return;

        // flatten the 2D buffer into a 1D input vector for the classifier
        std::vector<float> input;
        input.reserve(kGestureInputSize);
        for (auto& s : buffer)
            for (float v : s) input.push_back(v);

        // run inference + record the predicted class idx
        auto prediction = classifier->Predict(input);
        votes.push_back(prediction.gesture_index);
        if ((int)votes.size() > kVoteWindow)
            votes.pop_front();

        if (cooldown > 0) { cooldown--; return; }

        // count votes of rectent predictions - then find most common class
        std::array<int, kGestureNumClasses> counts{};
        for (int p : votes) counts[p]++;
        int best = std::max_element(counts.begin(), counts.end()) - counts.begin();

        // confirm if enough preds agree on the same one
        if (counts[best] >= kVoteThreshold) {
            on_gesture_detected(best);
            cooldown = kCooldownSamples;
            votes.clear();
        }
    };

    // continuously read IMU samples and feed them to the on_sample callback
    while (true) {
        ImuSample s;
        imu.Read(&s);

        on_sample({s.accel_x, s.accel_y, s.accel_z,
                   s.gyro_x,  s.gyro_y,  s.gyro_z,
                   s.mag_x,   s.mag_y,   s.mag_z});

        std::this_thread::sleep_for(std::chrono::milliseconds(16));
    }
}