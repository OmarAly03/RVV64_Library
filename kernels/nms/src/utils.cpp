#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <vector>
#include <riscv_vector.h>
#include "../include/defs.h"
#include "../../lib/rvv_defs.hpp"

using namespace std;

// Scalar IoU computation
float compute_iou(const float* box1, const float* box2, int center_point_box) {
    float converted_box1[4], converted_box2[4];
    
    // Convert to corner format if needed
    if (center_point_box == CENTER_FORMAT) {
        convert_box_format(box1, converted_box1, CENTER_FORMAT, CORNER_FORMAT);
        convert_box_format(box2, converted_box2, CENTER_FORMAT, CORNER_FORMAT);
        box1 = converted_box1;
        box2 = converted_box2;
    }
    
    // Extract coordinates: box format [y1, x1, y2, x2]
    float y1_1 = box1[0], x1_1 = box1[1], y2_1 = box1[2], x2_1 = box1[3];
    float y1_2 = box2[0], x1_2 = box2[1], y2_2 = box2[2], x2_2 = box2[3];
    
    // Compute intersection coordinates
    float inter_y1 = max(y1_1, y1_2);
    float inter_x1 = max(x1_1, x1_2);
    float inter_y2 = min(y2_1, y2_2);
    float inter_x2 = min(x2_1, x2_2);
    
    // Compute intersection area
    float inter_width = max(0.0f, inter_x2 - inter_x1);
    float inter_height = max(0.0f, inter_y2 - inter_y1);
    float inter_area = inter_width * inter_height;
    
    // Compute areas of both boxes
    float area1 = (x2_1 - x1_1) * (y2_1 - y1_1);
    float area2 = (x2_2 - x1_2) * (y2_2 - y1_2);
    
    // Compute union area
    float union_area = area1 + area2 - inter_area;
    
    // Compute IoU
    if (union_area <= 0.0f) return 0.0f;
    return inter_area / union_area;
}

// Convert box format between center and corner representations
void convert_box_format(const float* box, float* converted_box, int from_format, int to_format) {
    if (from_format == to_format) {
        // No conversion needed
        for (int i = 0; i < 4; i++) {
            converted_box[i] = box[i];
        }
        return;
    }
    
    if (from_format == CENTER_FORMAT && to_format == CORNER_FORMAT) {
        // Convert from [x_center, y_center, width, height] to [y1, x1, y2, x2]
        float x_center = box[0];
        float y_center = box[1];
        float width = box[2];
        float height = box[3];
        
        converted_box[0] = y_center - height / 2.0f;  // y1
        converted_box[1] = x_center - width / 2.0f;   // x1
        converted_box[2] = y_center + height / 2.0f;  // y2
        converted_box[3] = x_center + width / 2.0f;   // x2
    } else if (from_format == CORNER_FORMAT && to_format == CENTER_FORMAT) {
        // Convert from [y1, x1, y2, x2] to [x_center, y_center, width, height]
        float y1 = box[0], x1 = box[1], y2 = box[2], x2 = box[3];
        
        converted_box[0] = (x1 + x2) / 2.0f;  // x_center
        converted_box[1] = (y1 + y2) / 2.0f;  // y_center
        converted_box[2] = x2 - x1;           // width
        converted_box[3] = y2 - y1;           // height
    }
}

// RVV vectorized box format conversion
void convert_box_format_rvv(const float* box, float* converted_box, int from_format, int to_format) {
    if (from_format == to_format) {
        // No conversion needed - copy with vector load/store
        size_t vl = 4;
        auto v_box = VECTOR_LOAD<float, M1>(box, vl);
        VECTOR_STORE<float, M1>(converted_box, v_box, vl);
        return;
    }
    
    size_t vl = 4;
    auto v_box = VECTOR_LOAD<float, M1>(box, vl);
    
    if (from_format == CENTER_FORMAT && to_format == CORNER_FORMAT) {
        // Convert from [x_center, y_center, width, height] to [y1, x1, y2, x2]
        // Load: [x_center, y_center, width, height]
        auto x_center = VECTOR_EXTRACT_SCALAR<float, M1>(v_box);
        auto y_center = VECTOR_EXTRACT_SCALAR<float, M1>(VECTOR_SLIDEDOWN<float, M1>(v_box, 1, vl));
        auto width = VECTOR_EXTRACT_SCALAR<float, M1>(VECTOR_SLIDEDOWN<float, M1>(v_box, 2, vl));
        auto height = VECTOR_EXTRACT_SCALAR<float, M1>(VECTOR_SLIDEDOWN<float, M1>(v_box, 3, vl));
        
        // Create vector [width/2, height/2, width/2, height/2]
        float half_dims[4] = {width * 0.5f, height * 0.5f, width * 0.5f, height * 0.5f};
        auto v_half_dims = VECTOR_LOAD<float, M1>(half_dims, vl);
        
        // Create center vector [x_center, y_center, x_center, y_center]
        float centers[4] = {x_center, y_center, x_center, y_center};
        auto v_centers = VECTOR_LOAD<float, M1>(centers, vl);
        
        // Compute [x_center - width/2, y_center - height/2, x_center + width/2, y_center + height/2]
        float signs[4] = {-1.0f, -1.0f, 1.0f, 1.0f};
        auto v_signs = VECTOR_LOAD<float, M1>(signs, vl);
        auto v_result_temp = VECTOR_FMACC<float, M1>(v_centers, v_half_dims, v_signs, vl);
        
        // Swap to get [y1, x1, y2, x2] from [x1, y1, x2, y2]
        float result_temp[4];
        VECTOR_STORE<float, M1>(result_temp, v_result_temp, vl);
        converted_box[0] = result_temp[1];  // y1
        converted_box[1] = result_temp[0];  // x1
        converted_box[2] = result_temp[3];  // y2
        converted_box[3] = result_temp[2];  // x2
        
    } else if (from_format == CORNER_FORMAT && to_format == CENTER_FORMAT) {
        // Convert from [y1, x1, y2, x2] to [x_center, y_center, width, height]
        auto y1 = VECTOR_EXTRACT_SCALAR<float, M1>(v_box);
        auto x1 = VECTOR_EXTRACT_SCALAR<float, M1>(VECTOR_SLIDEDOWN<float, M1>(v_box, 1, vl));
        auto y2 = VECTOR_EXTRACT_SCALAR<float, M1>(VECTOR_SLIDEDOWN<float, M1>(v_box, 2, vl));
        auto x2 = VECTOR_EXTRACT_SCALAR<float, M1>(VECTOR_SLIDEDOWN<float, M1>(v_box, 3, vl));
        
        // Create vectors [x1, y1, x2, y2] and [x2, y2, x1, y1]
        float corners1[4] = {x1, y1, x2, y2};
        float corners2[4] = {x2, y2, x1, y1};
        auto v_corners1 = VECTOR_LOAD<float, M1>(corners1, vl);
        auto v_corners2 = VECTOR_LOAD<float, M1>(corners2, vl);
        
        // Sum: [x1+x2, y1+y2, x2+x1, y2+y1] (but we only need first two for center)
        auto v_sum = VECTOR_ADD<float, M1>(v_corners1, v_corners2, vl);
        
        // Difference: [x2-x1, y2-y1, x1-x2, y1-y2] (we only need first two for width/height)
        auto v_diff = VECTOR_SUB<float, M1>(v_corners2, v_corners1, vl);
        
        // Divide by 2 for centers
        auto v_half = VECTOR_MOVE<float, M1>(0.5f, vl);
        auto v_center = VECTOR_MUL<float, M1>(v_sum, v_half, vl);
        
        // Result: [x_center, y_center, width, height]
        VECTOR_STORE<float, M1>(converted_box, v_center, 2);      // Store centers
        VECTOR_STORE<float, M1>(converted_box + 2, v_diff, 2);    // Store dimensions
    }
}

// Scalar NMS implementation for reference
vector<SelectedIndex> nms_scalar(
    const float* boxes, const float* scores,
    size_t num_batches, size_t num_classes, size_t spatial_dimension,
    int64_t max_output_boxes_per_class, float iou_threshold, float score_threshold,
    int center_point_box
) {
    vector<SelectedIndex> selected_indices;
    
    if (max_output_boxes_per_class == 0) {
        return selected_indices;
    }
    
    for (size_t batch = 0; batch < num_batches; batch++) {
        for (size_t cls = 0; cls < num_classes; cls++) {
            // Create vector of (score, index) pairs for current batch and class
            vector<pair<float, size_t>> score_index_pairs;
            
            // Filter scores above threshold
            for (size_t i = 0; i < spatial_dimension; i++) {
                size_t score_idx = batch * num_classes * spatial_dimension + cls * spatial_dimension + i;
                float score = scores[score_idx];
                
                if (score >= score_threshold) {
                    score_index_pairs.push_back({score, i});
                }
            }
            
            // Sort by score in descending order
            sort(score_index_pairs.begin(), score_index_pairs.end(),
                      [](const pair<float, size_t>& a, const pair<float, size_t>& b) {
                          return a.first > b.first;
                      });
            
            vector<bool> suppressed(score_index_pairs.size(), false);
            int64_t selected_count = 0;
            
            // Apply NMS
            for (size_t i = 0; i < score_index_pairs.size() && selected_count < max_output_boxes_per_class; i++) {
                if (suppressed[i]) continue;
                
                size_t box_idx = score_index_pairs[i].second;
                selected_indices.push_back({static_cast<int64_t>(batch), static_cast<int64_t>(cls), static_cast<int64_t>(box_idx)});
                selected_count++;
                
                // Suppress overlapping boxes
                const float* current_box = &boxes[batch * spatial_dimension * 4 + box_idx * 4];
                
                for (size_t j = i + 1; j < score_index_pairs.size(); j++) {
                    if (suppressed[j]) continue;
                    
                    size_t other_box_idx = score_index_pairs[j].second;
                    const float* other_box = &boxes[batch * spatial_dimension * 4 + other_box_idx * 4];
                    
                    float iou = compute_iou(current_box, other_box, center_point_box);
                    if (iou > iou_threshold) {
                        suppressed[j] = true;
                    }
                }
            }
        }
    }
    
    return selected_indices;
}

// Write NMS results to text file
void write_nms_results_to_file(const char* filename, const vector<SelectedIndex>& results) {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Cannot open file " << filename << " for writing." << endl;
        return;
    }
    
    file << "Selected indices (batch_index, class_index, box_index):" << endl;
    for (const auto& result : results) {
        file << result.batch_index << ", " << result.class_index << ", " << result.box_index << endl;
    }
    
    file.close();
    cout << "NMS results written to " << filename << endl;
}

// Write NMS results to binary file
void write_nms_results_binary(const char* filename, const vector<SelectedIndex>& results) {
    ofstream file(filename, ios::binary);
    if (!file.is_open()) {
        cerr << "Error: Cannot open file " << filename << " for writing." << endl;
        return;
    }
    
    size_t count = results.size();
    file.write(reinterpret_cast<const char*>(&count), sizeof(count));
    
    for (const auto& result : results) {
        file.write(reinterpret_cast<const char*>(&result.batch_index), sizeof(result.batch_index));
        file.write(reinterpret_cast<const char*>(&result.class_index), sizeof(result.class_index));
        file.write(reinterpret_cast<const char*>(&result.box_index), sizeof(result.box_index));
    }
    
    file.close();
}