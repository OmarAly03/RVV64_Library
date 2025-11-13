// model.cpp
#include "yolo_model.hpp"
#include "kernels.hpp"
#include <opencv2/opencv.hpp> // For BoundingBox::get_rect() and IOU

// --- 1. Post-processing functions (copied from our previous example) ---
// We make them static so they are private to this file.

static inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

// MODIFIED: Operates on a raw pointer and size
static inline void softmax(float* x, size_t size) {
    if (size == 0) return;
    float max_val = *std::max_element(x, x + size);
    float sum = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        x[i] = std::exp(x[i] - max_val);
        sum += x[i];
    }
    for (size_t i = 0; i < size; ++i) {
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

// MODIFIED: Uses C-style array for class_scores
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

                // Use a stack-allocated C-style array
                float class_scores[NUM_CLASSES];
                int class_start_idx = anchor_offset + (5 * GRID_H * GRID_W) + cell_offset;
                for (int c = 0; c < NUM_CLASSES; ++c) {
                    int class_idx = class_start_idx + (c * GRID_H * GRID_W);
                    class_scores[c] = net_output[class_idx];
                }
                
                softmax(class_scores, NUM_CLASSES);
                
                // Find max element in C-style array
                int class_id = std::max_element(class_scores, class_scores + NUM_CLASSES) - class_scores;
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


// --- 2. Main Inference Function (HEAVILY MODIFIED) ---

std::vector<BoundingBox> yolo_model_inference(
    const ModelWeights& w,
    const std::vector<float>& input_image)
{
    // --- 1. Setup Buffers ---
    // We use two "ping-pong" buffers for activations
    // MODIFIED: Allocate buffers ONCE and re-use them.
    // This is the single most important change for embedded performance.
    
    // Max size for buf_a (ping) is 692,224 (from Layer 1: 16*208*208)
    // We get this by finding the max size of all `buf_a.resize` calls
    const size_t buf_a_max_size = 692224; 
    // Max size for buf_b (pong) is 2,777,088 (from Layer 0: 16*416*416)
    const size_t buf_b_max_size = 2777088; 

    // Use 'static' to allocate in static data segment, not stack/heap per call
    static std::vector<float> buf_a(buf_a_max_size);
    static std::vector<float> buf_b(buf_b_max_size);

    const float* in_ptr;
    float* out_ptr;

    // --- 2. Preprocessing ---
    // MODIFIED: Copy input image data into our static buffer
    if(input_image.size() > buf_a.size()) {
        std::cerr << "Error: Input image size is larger than buffer." << std::endl;
        return {};
    }
    std::copy(input_image.begin(), input_image.end(), buf_a.begin());
    
    preprocess_image(buf_a.data(), w.pp_scale.data(), w.pp_bias.data(), 3, 416, 416);
    in_ptr = buf_a.data(); // Input for next layer
    
    // --- 3. Model Body ---
    // ALL .resize() calls are removed.
    // ALL kernel calls now use .data() for weights.
    // ALL leaky_relu calls use explicit sizes.

    // Layer 0: Conv(16) -> BN -> Leaky
    out_ptr = buf_b.data();
    conv2d(in_ptr, out_ptr, w.conv0_w.data(), 3, 416, 416, 16, 416, 416, 3, 1, 1, 1);
    batch_normalization(out_ptr, w.bn0_s.data(), w.bn0_b.data(), w.bn0_m.data(), w.bn0_v.data(), 16, 416, 416);
    leaky_relu(out_ptr, 16 * 416 * 416, 0.1f);
    in_ptr = buf_b.data();

    // Layer 1: MaxPool(k=2, s=2)
    out_ptr = buf_a.data();
    max_pool_2d(in_ptr, out_ptr, 16, 416, 416, 208, 208, 2, 2, 0, 0);
    in_ptr = buf_a.data();

    // Layer 2: Conv(32) -> BN -> Leaky
    out_ptr = buf_b.data();
    conv2d(in_ptr, out_ptr, w.conv1_w.data(), 16, 208, 208, 32, 208, 208, 3, 1, 1, 1);
    batch_normalization(out_ptr, w.bn1_s.data(), w.bn1_b.data(), w.bn1_m.data(), w.bn1_v.data(), 32, 208, 208);
    leaky_relu(out_ptr, 32 * 208 * 208, 0.1f);
    in_ptr = buf_b.data();

    // Layer 3: MaxPool(k=2, s=2)
    out_ptr = buf_a.data();
    max_pool_2d(in_ptr, out_ptr, 32, 208, 208, 104, 104, 2, 2, 0, 0);
    in_ptr = buf_a.data();

    // Layer 4: Conv(64) -> BN -> Leaky
    out_ptr = buf_b.data();
    conv2d(in_ptr, out_ptr, w.conv2_w.data(), 32, 104, 104, 64, 104, 104, 3, 1, 1, 1);
    batch_normalization(out_ptr, w.bn2_s.data(), w.bn2_b.data(), w.bn2_m.data(), w.bn2_v.data(), 64, 104, 104);
    leaky_relu(out_ptr, 64 * 104 * 104, 0.1f);
    in_ptr = buf_b.data();

    // Layer 5: MaxPool(k=2, s=2)
    out_ptr = buf_a.data();
    max_pool_2d(in_ptr, out_ptr, 64, 104, 104, 52, 52, 2, 2, 0, 0);
    in_ptr = buf_a.data();

    // Layer 6: Conv(128) -> BN -> Leaky
    out_ptr = buf_b.data();
    conv2d(in_ptr, out_ptr, w.conv3_w.data(), 64, 52, 52, 128, 52, 52, 3, 1, 1, 1);
    batch_normalization(out_ptr, w.bn3_s.data(), w.bn3_b.data(), w.bn3_m.data(), w.bn3_v.data(), 128, 52, 52);
    leaky_relu(out_ptr, 128 * 52 * 52, 0.1f);
    in_ptr = buf_b.data();

    // Layer 7: MaxPool(k=2, s=2)
    out_ptr = buf_a.data();
    max_pool_2d(in_ptr, out_ptr, 128, 52, 52, 26, 26, 2, 2, 0, 0);
    in_ptr = buf_a.data();

    // Layer 8: Conv(256) -> BN -> Leaky
    out_ptr = buf_b.data();
    conv2d(in_ptr, out_ptr, w.conv4_w.data(), 128, 26, 26, 256, 26, 26, 3, 1, 1, 1);
    batch_normalization(out_ptr, w.bn4_s.data(), w.bn4_b.data(), w.bn4_m.data(), w.bn4_v.data(), 256, 26, 26);
    leaky_relu(out_ptr, 256 * 26 * 26, 0.1f);
    in_ptr = buf_b.data();

    // Layer 9: MaxPool(k=2, s=2)
    out_ptr = buf_a.data();
    max_pool_2d(in_ptr, out_ptr, 256, 26, 26, 13, 13, 2, 2, 0, 0);
    in_ptr = buf_a.data();

    // Layer 10: Conv(512) -> BN -> Leaky
    out_ptr = buf_b.data();
    conv2d(in_ptr, out_ptr, w.conv5_w.data(), 256, 13, 13, 512, 13, 13, 3, 1, 1, 1);
    batch_normalization(out_ptr, w.bn5_s.data(), w.bn5_b.data(), w.bn5_m.data(), w.bn5_v.data(), 512, 13, 13);
    leaky_relu(out_ptr, 512 * 13 * 13, 0.1f);
    in_ptr = buf_b.data();

    // Layer 11: MaxPool(k=2, s=1, p=1) -- NOTE THE PADDING
    out_ptr = buf_a.data();
    max_pool_2d(in_ptr, out_ptr, 512, 13, 13, 13, 13, 2, 1, 0, 0);
    in_ptr = buf_a.data();

    // Layer 12: Conv(1024) -> BN -> Leaky
    out_ptr = buf_b.data();
    conv2d(in_ptr, out_ptr, w.conv6_w.data(), 512, 13, 13, 1024, 13, 13, 3, 1, 1, 1);
    batch_normalization(out_ptr, w.bn6_s.data(), w.bn6_b.data(), w.bn6_m.data(), w.bn6_v.data(), 1024, 13, 13);
    leaky_relu(out_ptr, 1024 * 13 * 13, 0.1f);
    in_ptr = buf_b.data();

    // Layer 13: Conv(1024) -> BN -> Leaky
    out_ptr = buf_a.data();
    conv2d(in_ptr, out_ptr, w.conv7_w.data(), 1024, 13, 13, 1024, 13, 13, 3, 1, 1, 1);
    batch_normalization(out_ptr, w.bn7_s.data(), w.bn7_b.data(), w.bn7_m.data(), w.bn7_v.data(), 1024, 13, 13);
    leaky_relu(out_ptr, 1024 * 13 * 13, 0.1f); // Fixed size
    in_ptr = buf_a.data();

    // Layer 14: Final Conv(125) + Bias
    out_ptr = buf_b.data();
    conv2d(in_ptr, out_ptr, w.conv8_w.data(), 1024, 13, 13, 125, 13, 13, 1, 1, 0, 0);
    add_bias(out_ptr, w.conv8_b.data(), 125, 13, 13);
    
    // out_ptr (pointing to buf_b) now holds the final (125, 13, 13) grid

    // --- 4. Post-processing ---
    // NOTE: These functions STILL use std::vector.
    std::vector<BoundingBox> boxes = decode_output(out_ptr, ANCHORS);
    return non_max_suppression(boxes);
}


// --- 3. Weight Loading Function ---
// (No changes needed for this part, as loading happens once)

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

    // ... (rest of the loading function is identical) ...

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