#ifndef YOLO_MODEL_HPP
#define YOLO_MODEL_HPP

#include "model.hpp" // The one with constants

// Holds all parameters loaded from .bin files
struct ModelWeights {
    // Preprocessing
    std::vector<float> pp_scale, pp_bias;
    
    // Layer 0
    std::vector<float> conv0_w, bn0_s, bn0_b, bn0_m, bn0_v;
    // Layer 1
    std::vector<float> conv1_w, bn1_s, bn1_b, bn1_m, bn1_v;
    // Layer 2
    std::vector<float> conv2_w, bn2_s, bn2_b, bn2_m, bn2_v;
    // Layer 3
    std::vector<float> conv3_w, bn3_s, bn3_b, bn3_m, bn3_v;
    // Layer 4
    std::vector<float> conv4_w, bn4_s, bn4_b, bn4_m, bn4_v;
    // Layer 5
    std::vector<float> conv5_w, bn5_s, bn5_b, bn5_m, bn5_v;
    // Layer 6
    std::vector<float> conv6_w, bn6_s, bn6_b, bn6_m, bn6_v;
    // Layer 7
    std::vector<float> conv7_w, bn7_s, bn7_b, bn7_m, bn7_v;
    // Layer 8 (Final)
    std::vector<float> conv8_w, conv8_b;
};

// Main inference function
std::vector<BoundingBox> yolo_model_inference(
    const ModelWeights& weights,
    const std::vector<float>& input_image // 1*3*416*416
);

// Helper to load all weights from the directory
void load_all_weights(ModelWeights& weights, const std::string& weight_dir);

#endif // YOLO_MODEL_HPP