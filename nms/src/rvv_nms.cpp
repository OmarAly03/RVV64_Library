#include <riscv_vector.h>
#include <cstddef>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstring>
#include "../include/defs.h"
#include "rvv_defs.hpp"

using namespace std;

float compute_iou_rvv(const float* box1, const float* box2, int center_point_box) {
    float converted_box1[4], converted_box2[4];

    if (center_point_box == CENTER_FORMAT) {
        convert_box_format(box1, converted_box1, CENTER_FORMAT, CORNER_FORMAT);
        convert_box_format(box2, converted_box2, CENTER_FORMAT, CORNER_FORMAT);
        box1 = converted_box1;
        box2 = converted_box2;
    }

    size_t vl = 2;
    auto b1_xy1 = VECTOR_LOAD<float, M1>(box1, vl);
    auto b1_xy2 = VECTOR_LOAD<float, M1>(box1 + 2, vl);

    auto b2_xy1 = VECTOR_LOAD<float, M1>(box2, vl);
    auto b2_xy2 = VECTOR_LOAD<float, M1>(box2 + 2, vl);

    auto inter_xy1 = VECTOR_MAX<float, M1>(b1_xy1, b2_xy1, vl);
    auto inter_xy2 = VECTOR_MIN<float, M1>(b1_xy2, b2_xy2, vl);

    auto inter_wh = VECTOR_SUB<float, M1>(inter_xy2, inter_xy1, vl);
    inter_wh = VECTOR_MAX<float, M1>(inter_wh, 0.0f, vl);

    auto inter_w = VECTOR_EXTRACT_SCALAR<float, M1>(VECTOR_SLIDEDOWN<float, M1>(inter_wh, 1, vl));
    auto inter_h = VECTOR_EXTRACT_SCALAR<float, M1>(inter_wh);
    float inter_area = inter_w * inter_h;

    auto b1_wh = VECTOR_SUB<float, M1>(b1_xy2, b1_xy1, vl);
    auto area1 = VECTOR_EXTRACT_SCALAR<float, M1>(b1_wh) * VECTOR_EXTRACT_SCALAR<float, M1>(VECTOR_SLIDEDOWN<float, M1>(b1_wh, 1, vl));

    auto b2_wh = VECTOR_SUB<float, M1>(b2_xy2, b2_xy1, vl);
    auto area2 = VECTOR_EXTRACT_SCALAR<float, M1>(b2_wh) * VECTOR_EXTRACT_SCALAR<float, M1>(VECTOR_SLIDEDOWN<float, M1>(b2_wh, 1, vl));

    float union_area = area1 + area2 - inter_area;

    return union_area > 0 ? inter_area / union_area : 0;
}

// NMS implementation with RVV m1
vector<SelectedIndex> nms_e32m1(
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
            vector<pair<float, size_t>> score_index_pairs;

			for (size_t i = 0; i < spatial_dimension; i += SET_VECTOR_LENGTH_MAX<float, M1>()) {
				size_t vl = SET_VECTOR_LENGTH<float, M1>(spatial_dimension - i);
                size_t score_idx = batch * num_classes * spatial_dimension + cls * spatial_dimension + i;

                auto vscores = VECTOR_LOAD<float, M1>(&scores[score_idx], vl);
                auto vthreshold = VECTOR_MOVE<float, M1>(score_threshold, vl);
                auto mask = VECTOR_GE<float, M1>(vscores, vthreshold, vl);

                size_t count = VECTOR_COUNT_POP(mask, vl);
                if (count > 0) {
                    auto all_indices = VECTOR_VID<uint32_t, M1>(vl);
                    auto selected_indices_vec = VECTOR_COMPRESS<uint32_t, M1>(all_indices, mask, vl);
                    uint32_t* indices_arr = new uint32_t[count];
                    VECTOR_STORE<uint32_t, M1>(indices_arr, selected_indices_vec, count);
                    for (size_t k = 0; k < count; k++) {
                        size_t j = indices_arr[k];
                        score_index_pairs.push_back({scores[score_idx + j], i + j});
                    }
                    delete[] indices_arr;
                }
            }

            sort(score_index_pairs.begin(), score_index_pairs.end(),
                      [](const pair<float, size_t>& a, const pair<float, size_t>& b) {
                          return a.first > b.first;
                      });

            vector<bool> suppressed(score_index_pairs.size(), false);
            int64_t selected_count = 0;

            for (size_t i = 0; i < score_index_pairs.size() && selected_count < max_output_boxes_per_class; i++) {
                if (suppressed[i]) continue;

                size_t box_idx = score_index_pairs[i].second;
                selected_indices.push_back({static_cast<int64_t>(batch), static_cast<int64_t>(cls), static_cast<int64_t>(box_idx)});
                selected_count++;

                const float* current_box = &boxes[batch * spatial_dimension * 4 + box_idx * 4];

                for (size_t j = i + 1; j < score_index_pairs.size(); j++) {
                    if (suppressed[j]) continue;

                    size_t other_box_idx = score_index_pairs[j].second;
                    const float* other_box = &boxes[batch * spatial_dimension * 4 + other_box_idx * 4];

                    float iou = compute_iou_rvv(current_box, other_box, center_point_box);
                    if (iou > iou_threshold) {
                        suppressed[j] = true;
                    }
                }
            }
        }
    }

    return selected_indices;
}

// NMS implementation with RVV m2
vector<SelectedIndex> nms_e32m2(
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
            vector<pair<float, size_t>> score_index_pairs;

			for (size_t i = 0; i < spatial_dimension; i += SET_VECTOR_LENGTH_MAX<float, M2>()) {
				size_t vl = SET_VECTOR_LENGTH<float, M2>(spatial_dimension - i);
                size_t score_idx = batch * num_classes * spatial_dimension + cls * spatial_dimension + i;

                auto vscores = VECTOR_LOAD<float, M2>(&scores[score_idx], vl);
                auto vthreshold = VECTOR_MOVE<float, M2>(score_threshold, vl);
                auto mask = VECTOR_GE<float, M2>(vscores, vthreshold, vl);

                size_t count = VECTOR_COUNT_POP(mask, vl);
                if (count > 0) {
                    auto all_indices = VECTOR_VID<uint32_t, M2>(vl);
                    auto selected_indices_vec = VECTOR_COMPRESS<uint32_t, M2>(all_indices, mask, vl);
                    uint32_t* indices_arr = new uint32_t[count];
                    VECTOR_STORE<uint32_t, M2>(indices_arr, selected_indices_vec, count);
                    for (size_t k = 0; k < count; k++) {
                        size_t j = indices_arr[k];
                        score_index_pairs.push_back({scores[score_idx + j], i + j});
                    }
                    delete[] indices_arr;
                }
            }

            sort(score_index_pairs.begin(), score_index_pairs.end(),
                      [](const pair<float, size_t>& a, const pair<float, size_t>& b) {
                          return a.first > b.first;
                      });

            vector<bool> suppressed(score_index_pairs.size(), false);
            int64_t selected_count = 0;

            for (size_t i = 0; i < score_index_pairs.size() && selected_count < max_output_boxes_per_class; i++) {
                if (suppressed[i]) continue;

                size_t box_idx = score_index_pairs[i].second;
                selected_indices.push_back({static_cast<int64_t>(batch), static_cast<int64_t>(cls), static_cast<int64_t>(box_idx)});
                selected_count++;

                const float* current_box = &boxes[batch * spatial_dimension * 4 + box_idx * 4];

                for (size_t j = i + 1; j < score_index_pairs.size(); j++) {
                    if (suppressed[j]) continue;

                    size_t other_box_idx = score_index_pairs[j].second;
                    const float* other_box = &boxes[batch * spatial_dimension * 4 + other_box_idx * 4];

                    float iou = compute_iou_rvv(current_box, other_box, center_point_box);
                    if (iou > iou_threshold) {
                        suppressed[j] = true;
                    }
                }
            }
        }
    }

    return selected_indices;
}

// NMS implementation with RVV m4
vector<SelectedIndex> nms_e32m4(
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
            vector<pair<float, size_t>> score_index_pairs;

			for (size_t i = 0; i < spatial_dimension; i += SET_VECTOR_LENGTH_MAX<float, M4>()) {
				size_t vl = SET_VECTOR_LENGTH<float, M4>(spatial_dimension - i);
                size_t score_idx = batch * num_classes * spatial_dimension + cls * spatial_dimension + i;

                auto vscores = VECTOR_LOAD<float, M4>(&scores[score_idx], vl);
                auto vthreshold = VECTOR_MOVE<float, M4>(score_threshold, vl);
                auto mask = VECTOR_GE<float, M4>(vscores, vthreshold, vl);

                size_t count = VECTOR_COUNT_POP(mask, vl);
                if (count > 0) {
                    auto all_indices = VECTOR_VID<uint32_t, M4>(vl);
                    auto selected_indices_vec = VECTOR_COMPRESS<uint32_t, M4>(all_indices, mask, vl);
                    uint32_t* indices_arr = new uint32_t[count];
                    VECTOR_STORE<uint32_t, M4>(indices_arr, selected_indices_vec, count);
                    for (size_t k = 0; k < count; k++) {
                        size_t j = indices_arr[k];
                        score_index_pairs.push_back({scores[score_idx + j], i + j});
                    }
                    delete[] indices_arr;
                }
            }

            sort(score_index_pairs.begin(), score_index_pairs.end(),
                      [](const pair<float, size_t>& a, const pair<float, size_t>& b) {
                          return a.first > b.first;
                      });

            vector<bool> suppressed(score_index_pairs.size(), false);
            int64_t selected_count = 0;

            for (size_t i = 0; i < score_index_pairs.size() && selected_count < max_output_boxes_per_class; i++) {
                if (suppressed[i]) continue;

                size_t box_idx = score_index_pairs[i].second;
                selected_indices.push_back({static_cast<int64_t>(batch), static_cast<int64_t>(cls), static_cast<int64_t>(box_idx)});
                selected_count++;

                const float* current_box = &boxes[batch * spatial_dimension * 4 + box_idx * 4];

                for (size_t j = i + 1; j < score_index_pairs.size(); j++) {
                    if (suppressed[j]) continue;

                    size_t other_box_idx = score_index_pairs[j].second;
                    const float* other_box = &boxes[batch * spatial_dimension * 4 + other_box_idx * 4];

                    float iou = compute_iou_rvv(current_box, other_box, center_point_box);
                    if (iou > iou_threshold) {
                        suppressed[j] = true;
                    }
                }
            }
        }
    }

    return selected_indices;
}

// NMS implementation with RVV m8
vector<SelectedIndex> nms_e32m8(
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
            vector<pair<float, size_t>> score_index_pairs;

			for (size_t i = 0; i < spatial_dimension; i += SET_VECTOR_LENGTH_MAX<float, M8>()) {
				size_t vl = SET_VECTOR_LENGTH<float, M8>(spatial_dimension - i);
                size_t score_idx = batch * num_classes * spatial_dimension + cls * spatial_dimension + i;

                auto vscores = VECTOR_LOAD<float, M8>(&scores[score_idx], vl);
                auto vthreshold = VECTOR_MOVE<float, M8>(score_threshold, vl);
                auto mask = VECTOR_GE<float, M8>(vscores, vthreshold, vl);

                size_t count = VECTOR_COUNT_POP(mask, vl);
                if (count > 0) {
                    auto all_indices = VECTOR_VID<uint32_t, M8>(vl);
                    auto selected_indices_vec = VECTOR_COMPRESS<uint32_t, M8>(all_indices, mask, vl);
                    uint32_t* indices_arr = new uint32_t[count];
                    VECTOR_STORE<uint32_t, M8>(indices_arr, selected_indices_vec, count);
                    for (size_t k = 0; k < count; k++) {
                        size_t j = indices_arr[k];
                        score_index_pairs.push_back({scores[score_idx + j], i + j});
                    }
                    delete[] indices_arr;
                }
            }

            sort(score_index_pairs.begin(), score_index_pairs.end(),
                      [](const pair<float, size_t>& a, const pair<float, size_t>& b) {
                          return a.first > b.first;
                      });

            vector<bool> suppressed(score_index_pairs.size(), false);
            int64_t selected_count = 0;

            for (size_t i = 0; i < score_index_pairs.size() && selected_count < max_output_boxes_per_class; i++) {
                if (suppressed[i]) continue;

                size_t box_idx = score_index_pairs[i].second;
                selected_indices.push_back({static_cast<int64_t>(batch), static_cast<int64_t>(cls), static_cast<int64_t>(box_idx)});
                selected_count++;

                const float* current_box = &boxes[batch * spatial_dimension * 4 + box_idx * 4];

                for (size_t j = i + 1; j < score_index_pairs.size(); j++) {
                    if (suppressed[j]) continue;

                    size_t other_box_idx = score_index_pairs[j].second;
                    const float* other_box = &boxes[batch * spatial_dimension * 4 + other_box_idx * 4];

                    float iou = compute_iou_rvv(current_box, other_box, center_point_box);
                    if (iou > iou_threshold) {
                        suppressed[j] = true;
                    }
                }
            }
        }
    }

    return selected_indices;
}