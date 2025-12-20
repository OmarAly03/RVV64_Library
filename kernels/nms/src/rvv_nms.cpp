#include <riscv_vector.h>
#include <cstddef>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstring>
#include "../include/defs.h"
#include "rvv_defs.hpp"

using namespace std;

// Template version of compute_iou_rvv that accepts LMUL parameter
template<int LMUL>
float compute_iou_rvv_template(const float* box1, const float* box2, int center_point_box) {
    float converted_box1[4], converted_box2[4];

    if (center_point_box == CENTER_FORMAT) {
        convert_box_format_rvv(box1, converted_box1, CENTER_FORMAT, CORNER_FORMAT);
        convert_box_format_rvv(box2, converted_box2, CENTER_FORMAT, CORNER_FORMAT);
        box1 = converted_box1;
        box2 = converted_box2;
    }

    size_t vl = 2;
    auto b1_xy1 = VECTOR_LOAD<float, LMUL>(box1, vl);
    auto b1_xy2 = VECTOR_LOAD<float, LMUL>(box1 + 2, vl);

    auto b2_xy1 = VECTOR_LOAD<float, LMUL>(box2, vl);
    auto b2_xy2 = VECTOR_LOAD<float, LMUL>(box2 + 2, vl);

    auto inter_xy1 = VECTOR_MAX<float, LMUL>(b1_xy1, b2_xy1, vl);
    auto inter_xy2 = VECTOR_MIN<float, LMUL>(b1_xy2, b2_xy2, vl);

    auto inter_wh = VECTOR_SUB<float, LMUL>(inter_xy2, inter_xy1, vl);
    inter_wh = VECTOR_MAX<float, LMUL>(inter_wh, 0.0f, vl);

    auto inter_w = VECTOR_EXTRACT_SCALAR<float, LMUL>(VECTOR_SLIDEDOWN<float, LMUL>(inter_wh, 1, vl));
    auto inter_h = VECTOR_EXTRACT_SCALAR<float, LMUL>(inter_wh);
    float inter_area = inter_w * inter_h;

    auto b1_wh = VECTOR_SUB<float, LMUL>(b1_xy2, b1_xy1, vl);
    auto area1 = VECTOR_EXTRACT_SCALAR<float, LMUL>(b1_wh) * VECTOR_EXTRACT_SCALAR<float, LMUL>(VECTOR_SLIDEDOWN<float, LMUL>(b1_wh, 1, vl));

    auto b2_wh = VECTOR_SUB<float, LMUL>(b2_xy2, b2_xy1, vl);
    auto area2 = VECTOR_EXTRACT_SCALAR<float, LMUL>(b2_wh) * VECTOR_EXTRACT_SCALAR<float, LMUL>(VECTOR_SLIDEDOWN<float, LMUL>(b2_wh, 1, vl));

    float union_area = area1 + area2 - inter_area;

    return union_area > 0 ? inter_area / union_area : 0;
}

// Backward compatibility wrapper (defaults to M1)
float compute_iou_rvv(const float* box1, const float* box2, int center_point_box) {
    return compute_iou_rvv_template<M1>(box1, box2, center_point_box);
}

// Module: Filter scores above threshold using vectorized operations
template<int LMUL>
void filter_scores_by_threshold(
    const float* scores,
    size_t batch, size_t cls, size_t spatial_dimension,
    size_t num_classes,
    float score_threshold,
    vector<pair<float, size_t>>& score_index_pairs
) {
    for (size_t i = 0; i < spatial_dimension; i += SET_VECTOR_LENGTH_MAX<float, LMUL>()) {
        size_t vl = SET_VECTOR_LENGTH<float, LMUL>(spatial_dimension - i);
        size_t score_idx = batch * num_classes * spatial_dimension + cls * spatial_dimension + i;

        auto vscores = VECTOR_LOAD<float, LMUL>(&scores[score_idx], vl);
        auto vthreshold = VECTOR_MOVE<float, LMUL>(score_threshold, vl);
        auto mask = VECTOR_GE<float, LMUL>(vscores, vthreshold, vl);

        size_t count = VECTOR_COUNT_POP(mask, vl);
        if (count > 0) {
            auto all_indices = VECTOR_VID<uint32_t, LMUL>(vl);
            auto selected_indices_vec = VECTOR_COMPRESS<uint32_t, LMUL>(all_indices, mask, vl);
            uint32_t* indices_arr = new uint32_t[count];
            VECTOR_STORE<uint32_t, LMUL>(indices_arr, selected_indices_vec, count);
            for (size_t k = 0; k < count; k++) {
                size_t j = indices_arr[k];
                score_index_pairs.push_back({scores[score_idx + j], i + j});
            }
            delete[] indices_arr;
        }
    }
}

// Module: Sort score-index pairs by score in descending order
void sort_by_score_descending(vector<pair<float, size_t>>& score_index_pairs) {
    sort(score_index_pairs.begin(), score_index_pairs.end(),
         [](const pair<float, size_t>& a, const pair<float, size_t>& b) {
             return a.first > b.first;  // Descending order by score
         });
}

// Helper: Fast overlap check using separating axis test
// Returns true if boxes overlap at all, false if completely separated
inline bool can_boxes_overlap(const float* box1, const float* box2) {
    // box format: [y1, x1, y2, x2] in corner format
    // Quick rejection: if boxes don't overlap at all, IoU = 0
    return !(box1[2] < box2[0] || box2[2] < box1[0] ||  // no y overlap
             box1[3] < box2[1] || box2[3] < box1[1]);   // no x overlap
}

// Module: Apply greedy NMS suppression with fast overlap rejection
template<int LMUL>
void apply_greedy_suppression(
    const float* boxes,
    const vector<pair<float, size_t>>& score_index_pairs,
    size_t batch, size_t cls, size_t spatial_dimension,
    int64_t max_output_boxes_per_class,
    float iou_threshold,
    int center_point_box,
    vector<SelectedIndex>& selected_indices
) {
    const size_t n = score_index_pairs.size();
    if (n == 0) return;
    
    vector<bool> suppressed(n, false);
    int64_t selected_count = 0;
    
    // Pre-convert all boxes to corner format to avoid repeated conversions
    vector<float> box_corners(n * 4);
    for (size_t i = 0; i < n; i++) {
        size_t box_idx = score_index_pairs[i].second;
        const float* box = &boxes[batch * spatial_dimension * 4 + box_idx * 4];
        
        if (center_point_box == CENTER_FORMAT) {
            convert_box_format_rvv(box, &box_corners[i * 4], CENTER_FORMAT, CORNER_FORMAT);
        } else {
            memcpy(&box_corners[i * 4], box, 4 * sizeof(float));
        }
    }

    for (size_t i = 0; i < n && selected_count < max_output_boxes_per_class; i++) {
        if (suppressed[i]) continue;

        size_t box_idx = score_index_pairs[i].second;
        selected_indices.push_back({static_cast<int64_t>(batch), static_cast<int64_t>(cls), static_cast<int64_t>(box_idx)});
        selected_count++;

        if (selected_count >= max_output_boxes_per_class) break;

        const float* current_box = &box_corners[i * 4];
        
        // Process remaining boxes in batches for better vectorization
        size_t j = i + 1;
        
        // Vectorized batch processing: check multiple boxes at once
        while (j < n) {
            size_t batch_end = min(j + 8, n);  // Process 8 boxes at a time
            
            for (size_t k = j; k < batch_end; k++) {
                if (suppressed[k]) continue;
                
                const float* other_box = &box_corners[k * 4];
                
                // Fast rejection filter (separating axis test)
                if (!can_boxes_overlap(current_box, other_box)) {
                    continue;  // Skip expensive IoU computation
                }
                
                // Full IoU computation only for candidates
                float iou = compute_iou_rvv_template<LMUL>(current_box, other_box, CORNER_FORMAT);
                
                if (iou > iou_threshold) {
                    suppressed[k] = true;
                }
            }
            
            j = batch_end;
        }
    }
}

// Template NMS implementation for all LMUL variants
template<int LMUL>
vector<SelectedIndex> nms_template(
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
            // Step 1: Filter scores above threshold
            vector<pair<float, size_t>> score_index_pairs;
            filter_scores_by_threshold<LMUL>(scores, batch, cls, spatial_dimension, 
                                             num_classes, score_threshold, score_index_pairs);

            // Step 2: Sort by score (highest first)
            sort_by_score_descending(score_index_pairs);

            // Step 3: Apply greedy NMS suppression
            apply_greedy_suppression<LMUL>(boxes, score_index_pairs, batch, cls, spatial_dimension,
                                          max_output_boxes_per_class, iou_threshold, 
                                          center_point_box, selected_indices);
        }
    }

    return selected_indices;
}

// NMS implementation with RVV m1
vector<SelectedIndex> nms_e32m1(
    const float* boxes, const float* scores,
    size_t num_batches, size_t num_classes, size_t spatial_dimension,
    int64_t max_output_boxes_per_class, float iou_threshold, float score_threshold,
    int center_point_box
) {
    return nms_template<M1>(boxes, scores, num_batches, num_classes, spatial_dimension,
                            max_output_boxes_per_class, iou_threshold, score_threshold, center_point_box);
}

// NMS implementation with RVV m2
vector<SelectedIndex> nms_e32m2(
    const float* boxes, const float* scores,
    size_t num_batches, size_t num_classes, size_t spatial_dimension,
    int64_t max_output_boxes_per_class, float iou_threshold, float score_threshold,
    int center_point_box
) {
    return nms_template<M2>(boxes, scores, num_batches, num_classes, spatial_dimension,
                            max_output_boxes_per_class, iou_threshold, score_threshold, center_point_box);
}

// NMS implementation with RVV m4
vector<SelectedIndex> nms_e32m4(
    const float* boxes, const float* scores,
    size_t num_batches, size_t num_classes, size_t spatial_dimension,
    int64_t max_output_boxes_per_class, float iou_threshold, float score_threshold,
    int center_point_box
) {
    return nms_template<M4>(boxes, scores, num_batches, num_classes, spatial_dimension,
                            max_output_boxes_per_class, iou_threshold, score_threshold, center_point_box);
}

// NMS implementation with RVV m8
vector<SelectedIndex> nms_e32m8(
    const float* boxes, const float* scores,
    size_t num_batches, size_t num_classes, size_t spatial_dimension,
    int64_t max_output_boxes_per_class, float iou_threshold, float score_threshold,
    int center_point_box
) {
    return nms_template<M8>(boxes, scores, num_batches, num_classes, spatial_dimension,
                            max_output_boxes_per_class, iou_threshold, score_threshold, center_point_box);
}