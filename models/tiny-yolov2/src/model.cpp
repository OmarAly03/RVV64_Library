// model.cpp
#include "yolo_model.hpp"
#include "kernels.hpp"
#include <opencv2/opencv.hpp> // For BoundingBox::get_rect() and IOU

// --- 1. Post-processing functions (copied from our previous example) ---
// We make them static so they are private to this file.

static inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

static inline void softmax(std::vector<float>& x) {
    float max_val = *std::max_element(x.begin(), x.end());
    float sum = 0.0f;
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] = std::exp(x[i] - max_val);
        sum += x[i];
    }
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] /= sum;
    }
}

// BoundingBox::get_rect() needs cv::Rect
static cv::Rect get_rect(const BoundingBox& box) {
    float x_min = box.x - box.w / 2.0f;
    float y_min = box.y - box.h / 2.0f;
    float x_max = box.x + box.w / 2.0f;
    float y_max = box.y + box.h / 2.0f;
    return cv::Rect(cv::Point((int)x_min, (int)y_min), cv::Point((int)x_max, (int)y_max));
}

static float iou(const BoundingBox& box1, const BoundingBox& box2) {
    cv::Rect rect1 = get_rect(box1);
    cv::Rect rect2 = get_rect(box2);

    cv::Rect intersection = rect1 & rect2;
    float inter_area = (float)intersection.area();
    float union_area = (float)rect1.area() + (float)rect2.area() - inter_area;

    if (union_area <= 0.0f) return 0.0f;
    return inter_area / union_area;
}

static std::vector<BoundingBox> decode_output(const float* net_output, const std::vector<std::vector<float>>& anchors) {
    std::vector<BoundingBox> boxes;
    for (int y = 0; y < GRID_H; ++y) {
        for (int x = 0; x < GRID_W; ++x) {
            for (int a = 0; a < NUM_ANCHORS; ++a) {
                int data_per_anchor = 5 + NUM_CLASSES;
                int anchor_offset = a * data_per_anchor * GRID_H * GRID_W;
                int cell_offset = (y * GRID_W) + x;
                
                int tx_idx = anchor_offset + (0 * GRID_H * GRID_W) + cell_offset;
                int ty_idx = anchor_offset + (1 * GRID_H * GRID_W) + cell_offset;
                int tw_idx = anchor_offset + (2 * GRID_H * GRID_W) + cell_offset;
                int th_idx = anchor_offset + (3 * GRID_H * GRID_W) + cell_offset;
                int obj_idx = anchor_offset + (4 * GRID_H * GRID_W) + cell_offset;

                float objectness = sigmoid(net_output[obj_idx]);
                if (objectness < OBJECT_THRESHOLD) continue;

                float center_x = (x + sigmoid(net_output[tx_idx]));
                float center_y = (y + sigmoid(net_output[ty_idx]));
                float width = anchors[a][0] * std::exp(net_output[tw_idx]);
                float height = anchors[a][1] * std::exp(net_output[th_idx]);

                std::vector<float> class_scores;
                int class_start_idx = anchor_offset + (5 * GRID_H * GRID_W) + cell_offset;
                for (int c = 0; c < NUM_CLASSES; ++c) {
                    int class_idx = class_start_idx + (c * GRID_H * GRID_W);
                    class_scores.push_back(net_output[class_idx]);
                }
                
                softmax(class_scores);
                int class_id = std::distance(class_scores.begin(), std::max_element(class_scores.begin(), class_scores.end()));
                float class_score = class_scores[class_id];
                
                float final_score = objectness * class_score;
                if (final_score < OBJECT_THRESHOLD) continue;

                boxes.push_back({center_x, center_y, width, height, final_score, class_id});
            }
        }
    }
    return boxes;
}

static std::vector<BoundingBox> non_max_suppression(std::vector<BoundingBox>& boxes) {
    std::sort(boxes.begin(), boxes.end(), [](const BoundingBox& a, const BoundingBox& b) {
        return a.score > b.score;
    });

    std::vector<BoundingBox> suppressed_boxes;
    std::vector<bool> suppressed(boxes.size(), false);

    for (size_t i = 0; i < boxes.size(); ++i) {
        if (suppressed[i]) continue;
        suppressed_boxes.push_back(boxes[i]);
        suppressed[i] = true;
        for (size_t j = i + 1; j < boxes.size(); ++j) {
            if (suppressed[j]) continue;
            if (boxes[i].class_id == boxes[j].class_id && iou(boxes[i], boxes[j]) > NMS_THRESHOLD) {
                suppressed[j] = true;
            }
        }
    }
    return suppressed_boxes;
}


// --- 2. Main Inference Function ---

std::vector<BoundingBox> yolo_model_inference(
    const ModelWeights& w,
    const std::vector<float>& input_image)
{
    // --- 1. Setup Buffers ---
    // We use two "ping-pong" buffers for activations
    std::vector<float> buf_a, buf_b;
    const float* in_ptr;
    float* out_ptr;

    // --- 2. Preprocessing ---
    buf_a = input_image; // Copy input image to buf_a
    preprocess_image(buf_a.data(), w.pp_scale, w.pp_bias, 3, 416, 416);
    in_ptr = buf_a.data(); // Input for next layer
    
    // --- 3. Model Body ---

    // Layer 0: Conv(16) -> BN -> Leaky
    buf_b.resize(16 * 416 * 416); out_ptr = buf_b.data();
    conv2d(in_ptr, out_ptr, w.conv0_w, 3, 416, 416, 16, 416, 416, 3, 1, 1, 1);
    batch_normalization(out_ptr, w.bn0_s, w.bn0_b, w.bn0_m, w.bn0_v, 16, 416, 416);
    leaky_relu(out_ptr, buf_b.size(), 0.1f);
    in_ptr = buf_b.data();

    // Layer 1: MaxPool(k=2, s=2)
    buf_a.resize(16 * 208 * 208); out_ptr = buf_a.data();
    max_pool_2d(in_ptr, out_ptr, 16, 416, 416, 208, 208, 2, 2, 0, 0);
    in_ptr = buf_a.data();

    // Layer 2: Conv(32) -> BN -> Leaky
    buf_b.resize(32 * 208 * 208); out_ptr = buf_b.data();
    conv2d(in_ptr, out_ptr, w.conv1_w, 16, 208, 208, 32, 208, 208, 3, 1, 1, 1);
    batch_normalization(out_ptr, w.bn1_s, w.bn1_b, w.bn1_m, w.bn1_v, 32, 208, 208);
    leaky_relu(out_ptr, buf_b.size(), 0.1f);
    in_ptr = buf_b.data();

    // Layer 3: MaxPool(k=2, s=2)
    buf_a.resize(32 * 104 * 104); out_ptr = buf_a.data();
    max_pool_2d(in_ptr, out_ptr, 32, 208, 208, 104, 104, 2, 2, 0, 0);
    in_ptr = buf_a.data();

    // Layer 4: Conv(64) -> BN -> Leaky
    buf_b.resize(64 * 104 * 104); out_ptr = buf_b.data();
    conv2d(in_ptr, out_ptr, w.conv2_w, 32, 104, 104, 64, 104, 104, 3, 1, 1, 1);
    batch_normalization(out_ptr, w.bn2_s, w.bn2_b, w.bn2_m, w.bn2_v, 64, 104, 104);
    leaky_relu(out_ptr, buf_b.size(), 0.1f);
    in_ptr = buf_b.data();

    // Layer 5: MaxPool(k=2, s=2)
    buf_a.resize(64 * 52 * 52); out_ptr = buf_a.data();
    max_pool_2d(in_ptr, out_ptr, 64, 104, 104, 52, 52, 2, 2, 0, 0);
    in_ptr = buf_a.data();

    // Layer 6: Conv(128) -> BN -> Leaky
    buf_b.resize(128 * 52 * 52); out_ptr = buf_b.data();
    conv2d(in_ptr, out_ptr, w.conv3_w, 64, 52, 52, 128, 52, 52, 3, 1, 1, 1);
    batch_normalization(out_ptr, w.bn3_s, w.bn3_b, w.bn3_m, w.bn3_v, 128, 52, 52);
    leaky_relu(out_ptr, buf_b.size(), 0.1f);
    in_ptr = buf_b.data();

    // Layer 7: MaxPool(k=2, s=2)
    buf_a.resize(128 * 26 * 26); out_ptr = buf_a.data();
    max_pool_2d(in_ptr, out_ptr, 128, 52, 52, 26, 26, 2, 2, 0, 0);
    in_ptr = buf_a.data();

    // Layer 8: Conv(256) -> BN -> Leaky
    buf_b.resize(256 * 26 * 26); out_ptr = buf_b.data();
    conv2d(in_ptr, out_ptr, w.conv4_w, 128, 26, 26, 256, 26, 26, 3, 1, 1, 1);
    batch_normalization(out_ptr, w.bn4_s, w.bn4_b, w.bn4_m, w.bn4_v, 256, 26, 26);
    leaky_relu(out_ptr, buf_b.size(), 0.1f);
    in_ptr = buf_b.data();

    // Layer 9: MaxPool(k=2, s=2)
    buf_a.resize(256 * 13 * 13); out_ptr = buf_a.data();
    max_pool_2d(in_ptr, out_ptr, 256, 26, 26, 13, 13, 2, 2, 0, 0);
    in_ptr = buf_a.data();

    // Layer 10: Conv(512) -> BN -> Leaky
    buf_b.resize(512 * 13 * 13); out_ptr = buf_b.data();
    conv2d(in_ptr, out_ptr, w.conv5_w, 256, 13, 13, 512, 13, 13, 3, 1, 1, 1);
    batch_normalization(out_ptr, w.bn5_s, w.bn5_b, w.bn5_m, w.bn5_v, 512, 13, 13);
    leaky_relu(out_ptr, buf_b.size(), 0.1f);
    in_ptr = buf_b.data();

    // Layer 11: MaxPool(k=2, s=1, p=1) -- NOTE THE PADDING
    buf_a.resize(512 * 13 * 13); out_ptr = buf_a.data();
    max_pool_2d(in_ptr, out_ptr, 512, 13, 13, 13, 13, 2, 1, 0, 0);
    in_ptr = buf_a.data();

    // Layer 12: Conv(1024) -> BN -> Leaky
    buf_b.resize(1024 * 13 * 13); out_ptr = buf_b.data();
    conv2d(in_ptr, out_ptr, w.conv6_w, 512, 13, 13, 1024, 13, 13, 3, 1, 1, 1);
    batch_normalization(out_ptr, w.bn6_s, w.bn6_b, w.bn6_m, w.bn6_v, 1024, 13, 13);
    leaky_relu(out_ptr, buf_b.size(), 0.1f);
    in_ptr = buf_b.data();

    // Layer 13: Conv(1024) -> BN -> Leaky
    buf_a.resize(1024 * 13 * 13); out_ptr = buf_a.data();
    conv2d(in_ptr, out_ptr, w.conv7_w, 1024, 13, 13, 1024, 13, 13, 3, 1, 1, 1);
    batch_normalization(out_ptr, w.bn7_s, w.bn7_b, w.bn7_m, w.bn7_v, 1024, 13, 13);
    leaky_relu(out_ptr, buf_a.size(), 0.1f);
    in_ptr = buf_a.data();

    // Layer 14: Final Conv(125) + Bias
    buf_b.resize(125 * 13 * 13); out_ptr = buf_b.data();
    conv2d(in_ptr, out_ptr, w.conv8_w, 1024, 13, 13, 125, 13, 13, 1, 1, 0, 0);
    add_bias(out_ptr, w.conv8_b, 125, 13, 13);
    
    // out_ptr (pointing to buf_b) now holds the final (125, 13, 13) grid

    // --- 4. Post-processing ---
    std::vector<BoundingBox> boxes = decode_output(out_ptr, ANCHORS);
    return non_max_suppression(boxes);
}


// --- 3. Weight Loading Function ---

void load_all_weights(ModelWeights& w, const std::string& weight_dir) {
    std::cout << "Loading weights from " << weight_dir << "..." << std::endl;
    // Preprocessing
    w.pp_scale = load_weights_from_bin(weight_dir + "scalerPreprocessor_scale.bin", 1);
    w.pp_bias = load_weights_from_bin(weight_dir + "scalerPreprocessor_bias.bin", 3);
    
    // Layer 0
    w.conv0_w = load_weights_from_bin(weight_dir + "convolution_W.bin", 16*3*3*3);
    w.bn0_s = load_weights_from_bin(weight_dir + "BatchNormalization_scale.bin", 16);
    w.bn0_b = load_weights_from_bin(weight_dir + "BatchNormalization_B.bin", 16);
    w.bn0_m = load_weights_from_bin(weight_dir + "BatchNormalization_mean.bin", 16);
    w.bn0_v = load_weights_from_bin(weight_dir + "BatchNormalization_variance.bin", 16);

    // Layer 1
    w.conv1_w = load_weights_from_bin(weight_dir + "convolution1_W.bin", 32*16*3*3);
    w.bn1_s = load_weights_from_bin(weight_dir + "BatchNormalization_scale1.bin", 32);
    w.bn1_b = load_weights_from_bin(weight_dir + "BatchNormalization_B1.bin", 32);
    w.bn1_m = load_weights_from_bin(weight_dir + "BatchNormalization_mean1.bin", 32);
    w.bn1_v = load_weights_from_bin(weight_dir + "BatchNormalization_variance1.bin", 32);

    // Layer 2
    w.conv2_w = load_weights_from_bin(weight_dir + "convolution2_W.bin", 64*32*3*3);
    w.bn2_s = load_weights_from_bin(weight_dir + "BatchNormalization_scale2.bin", 64);
    w.bn2_b = load_weights_from_bin(weight_dir + "BatchNormalization_B2.bin", 64);
    w.bn2_m = load_weights_from_bin(weight_dir + "BatchNormalization_mean2.bin", 64);
    w.bn2_v = load_weights_from_bin(weight_dir + "BatchNormalization_variance2.bin", 64);

    // Layer 3
    w.conv3_w = load_weights_from_bin(weight_dir + "convolution3_W.bin", 128*64*3*3);
    w.bn3_s = load_weights_from_bin(weight_dir + "BatchNormalization_scale3.bin", 128);
    w.bn3_b = load_weights_from_bin(weight_dir + "BatchNormalization_B3.bin", 128);
    w.bn3_m = load_weights_from_bin(weight_dir + "BatchNormalization_mean3.bin", 128);
    w.bn3_v = load_weights_from_bin(weight_dir + "BatchNormalization_variance3.bin", 128);

    // Layer 4
    w.conv4_w = load_weights_from_bin(weight_dir + "convolution4_W.bin", 256*128*3*3);
    w.bn4_s = load_weights_from_bin(weight_dir + "BatchNormalization_scale4.bin", 256);
    w.bn4_b = load_weights_from_bin(weight_dir + "BatchNormalization_B4.bin", 256);
    w.bn4_m = load_weights_from_bin(weight_dir + "BatchNormalization_mean4.bin", 256);
    w.bn4_v = load_weights_from_bin(weight_dir + "BatchNormalization_variance4.bin", 256);

    // Layer 5
    w.conv5_w = load_weights_from_bin(weight_dir + "convolution5_W.bin", 512*256*3*3);
    w.bn5_s = load_weights_from_bin(weight_dir + "BatchNormalization_scale5.bin", 512);
    w.bn5_b = load_weights_from_bin(weight_dir + "BatchNormalization_B5.bin", 512);
    w.bn5_m = load_weights_from_bin(weight_dir + "BatchNormalization_mean5.bin", 512);
    w.bn5_v = load_weights_from_bin(weight_dir + "BatchNormalization_variance5.bin", 512);

    // Layer 6
    w.conv6_w = load_weights_from_bin(weight_dir + "convolution6_W.bin", 1024*512*3*3);
    w.bn6_s = load_weights_from_bin(weight_dir + "BatchNormalization_scale6.bin", 1024);
    w.bn6_b = load_weights_from_bin(weight_dir + "BatchNormalization_B6.bin", 1024);
    w.bn6_m = load_weights_from_bin(weight_dir + "BatchNormalization_mean6.bin", 1024);
    w.bn6_v = load_weights_from_bin(weight_dir + "BatchNormalization_variance6.bin", 1024);

    // Layer 7
    w.conv7_w = load_weights_from_bin(weight_dir + "convolution7_W.bin", 1024*1024*3*3);
    w.bn7_s = load_weights_from_bin(weight_dir + "BatchNormalization_scale7.bin", 1024);
    w.bn7_b = load_weights_from_bin(weight_dir + "BatchNormalization_B7.bin", 1024);
    w.bn7_m = load_weights_from_bin(weight_dir + "BatchNormalization_mean7.bin", 1024);
    w.bn7_v = load_weights_from_bin(weight_dir + "BatchNormalization_variance7.bin", 1024);

    // Layer 8 (Final)
    w.conv8_w = load_weights_from_bin(weight_dir + "convolution8_W.bin", 125*1024*1*1);
    w.conv8_b = load_weights_from_bin(weight_dir + "convolution8_B.bin", 125);
    
    std::cout << "All weights loaded." << std::endl;
}