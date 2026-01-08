#include <riscv_vector.h>
#include <stddef.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "../include/defs.h"
#include "../../../lib/rvv_defs.hpp"

/* ============================================================================
 * M1 IMPLEMENTATIONS
 * ============================================================================ */

static float compute_iou_rvv_m1(const float* box1, const float* box2, int center_point_box) {
    float converted_box1[4], converted_box2[4];

    if (center_point_box == CENTER_FORMAT) {
        convert_box_format_rvv(box1, converted_box1, CENTER_FORMAT, CORNER_FORMAT);
        convert_box_format_rvv(box2, converted_box2, CENTER_FORMAT, CORNER_FORMAT);
        box1 = converted_box1;
        box2 = converted_box2;
    }

    size_t vl = 2;
    vfloat32m1_t b1_xy1 = VECTOR_LOAD<float, M1>(box1, vl);
    vfloat32m1_t b1_xy2 = VECTOR_LOAD<float, M1>(box1 + 2, vl);

    vfloat32m1_t b2_xy1 = VECTOR_LOAD<float, M1>(box2, vl);
    vfloat32m1_t b2_xy2 = VECTOR_LOAD<float, M1>(box2 + 2, vl);

    vfloat32m1_t inter_xy1 = VECTOR_MAX<float, M1>(b1_xy1, b2_xy1, vl);
    vfloat32m1_t inter_xy2 = VECTOR_MIN<float, M1>(b1_xy2, b2_xy2, vl);

    vfloat32m1_t inter_wh = VECTOR_SUB<float, M1>(inter_xy2, inter_xy1, vl);
    inter_wh = VECTOR_MAX<float, M1>(inter_wh, 0.0f, vl);

    float inter_w = VECTOR_EXTRACT_SCALAR<float, M1>(VECTOR_SLIDEDOWN<float, M1>(inter_wh, 1, vl));
    float inter_h = VECTOR_EXTRACT_SCALAR<float, M1>(inter_wh);
    float inter_area = inter_w * inter_h;

    vfloat32m1_t b1_wh = VECTOR_SUB<float, M1>(b1_xy2, b1_xy1, vl);
    float area1 = VECTOR_EXTRACT_SCALAR<float, M1>(b1_wh) * VECTOR_EXTRACT_SCALAR<float, M1>(VECTOR_SLIDEDOWN<float, M1>(b1_wh, 1, vl));

    vfloat32m1_t b2_wh = VECTOR_SUB<float, M1>(b2_xy2, b2_xy1, vl);
    float area2 = VECTOR_EXTRACT_SCALAR<float, M1>(b2_wh) * VECTOR_EXTRACT_SCALAR<float, M1>(VECTOR_SLIDEDOWN<float, M1>(b2_wh, 1, vl));

    float union_area = area1 + area2 - inter_area;

    return union_area > 0 ? inter_area / union_area : 0;
}

float compute_iou_rvv(const float* box1, const float* box2, int center_point_box) {
    return compute_iou_rvv_m1(box1, box2, center_point_box);
}

static void filter_scores_by_threshold_m1(
    const float* scores,
    size_t batch, size_t cls, size_t spatial_dimension,
    size_t num_classes,
    float score_threshold,
    ScoreIndexVector* score_index_pairs
) {
    for (size_t i = 0; i < spatial_dimension; i += SET_VECTOR_LENGTH_MAX<float, M1>()) {
        size_t vl = SET_VECTOR_LENGTH<float, M1>(spatial_dimension - i);
        size_t score_idx = batch * num_classes * spatial_dimension + cls * spatial_dimension + i;

        vfloat32m1_t vscores = VECTOR_LOAD<float, M1>(&scores[score_idx], vl);
        vfloat32m1_t vthreshold = VECTOR_MOVE<float, M1>(score_threshold, vl);
        vbool32_t mask = VECTOR_GE<float, M1>(vscores, vthreshold, vl);

        size_t count = VECTOR_COUNT_POP(mask, vl);
        if (count > 0) {
            vuint32m1_t all_indices = VECTOR_VID<uint32_t, M1>(vl);
            vuint32m1_t selected_indices_vec = VECTOR_COMPRESS<uint32_t, M1>(all_indices, mask, vl);
            uint32_t* indices_arr = (uint32_t*)malloc(count * sizeof(uint32_t));
            VECTOR_STORE<uint32_t, M1>(indices_arr, selected_indices_vec, count);
            for (size_t k = 0; k < count; k++) {
                size_t j = indices_arr[k];
                ScoreIndexPair pair = {scores[score_idx + j], i + j};
                push_score_index(score_index_pairs, pair);
            }
            free(indices_arr);
        }
    }
}

static void sort_by_score_descending(ScoreIndexVector* score_index_pairs) {
    qsort(score_index_pairs->data, score_index_pairs->size, sizeof(ScoreIndexPair), compare_scores_desc);
}

static int can_boxes_overlap(const float* box1, const float* box2) {
    return !(box1[2] < box2[0] || box2[2] < box1[0] ||
             box1[3] < box2[1] || box2[3] < box1[1]);
}

static void apply_greedy_suppression_m1(
    const float* boxes,
    const ScoreIndexVector* score_index_pairs,
    size_t batch, size_t cls, size_t spatial_dimension,
    int64_t max_output_boxes_per_class,
    float iou_threshold,
    int center_point_box,
    SelectedIndexVector* selected_indices
) {
    const size_t n = score_index_pairs->size;
    if (n == 0) return;
    
    char* suppressed = (char*)calloc(n, sizeof(char));
    int64_t selected_count = 0;
    
    float* box_corners = (float*)malloc(n * 4 * sizeof(float));
    for (size_t i = 0; i < n; i++) {
        size_t box_idx = score_index_pairs->data[i].index;
        const float* box = &boxes[batch * spatial_dimension * 4 + box_idx * 4];
        
        if (center_point_box == CENTER_FORMAT) {
            convert_box_format_rvv(box, &box_corners[i * 4], CENTER_FORMAT, CORNER_FORMAT);
        } else {
            memcpy(&box_corners[i * 4], box, 4 * sizeof(float));
        }
    }

    for (size_t i = 0; i < n && selected_count < max_output_boxes_per_class; i++) {
        if (suppressed[i]) continue;

        size_t box_idx = score_index_pairs->data[i].index;
        SelectedIndex sel = {(int64_t)batch, (int64_t)cls, (int64_t)box_idx};
        push_selected_index(selected_indices, sel);
        selected_count++;

        if (selected_count >= max_output_boxes_per_class) break;

        const float* current_box = &box_corners[i * 4];
        
        size_t j = i + 1;
        
        while (j < n) {
            size_t batch_end = (j + 8 < n) ? (j + 8) : n;
            
            for (size_t k = j; k < batch_end; k++) {
                if (suppressed[k]) continue;
                
                const float* other_box = &box_corners[k * 4];
                
                if (!can_boxes_overlap(current_box, other_box)) {
                    continue;
                }
                
                float iou = compute_iou_rvv_m1(current_box, other_box, CORNER_FORMAT);
                
                if (iou > iou_threshold) {
                    suppressed[k] = 1;
                }
            }
            
            j = batch_end;
        }
    }
    
    free(box_corners);
    free(suppressed);
}

SelectedIndexVector nms_e32m1(
    const float* boxes, const float* scores,
    size_t num_batches, size_t num_classes, size_t spatial_dimension,
    int64_t max_output_boxes_per_class, float iou_threshold, float score_threshold,
    int center_point_box
) {
    SelectedIndexVector selected_indices;
    init_selected_vector(&selected_indices);

    if (max_output_boxes_per_class == 0) {
        return selected_indices;
    }

    for (size_t batch = 0; batch < num_batches; batch++) {
        for (size_t cls = 0; cls < num_classes; cls++) {
            ScoreIndexVector score_index_pairs;
            init_score_vector(&score_index_pairs);
            filter_scores_by_threshold_m1(scores, batch, cls, spatial_dimension, 
                                         num_classes, score_threshold, &score_index_pairs);

            sort_by_score_descending(&score_index_pairs);

            apply_greedy_suppression_m1(boxes, &score_index_pairs, batch, cls, spatial_dimension,
                                       max_output_boxes_per_class, iou_threshold, 
                                       center_point_box, &selected_indices);
            
            free_score_vector(&score_index_pairs);
        }
    }

    return selected_indices;
}

/* ============================================================================
 * M2 IMPLEMENTATIONS (Duplicated with M2 types)
 * ============================================================================ */

static float compute_iou_rvv_m2(const float* box1, const float* box2, int center_point_box) {
    float converted_box1[4], converted_box2[4];

    if (center_point_box == CENTER_FORMAT) {
        convert_box_format_rvv(box1, converted_box1, CENTER_FORMAT, CORNER_FORMAT);
        convert_box_format_rvv(box2, converted_box2, CENTER_FORMAT, CORNER_FORMAT);
        box1 = converted_box1;
        box2 = converted_box2;
    }

    size_t vl = 2;
    vfloat32m2_t b1_xy1 = VECTOR_LOAD<float, M2>(box1, vl);
    vfloat32m2_t b1_xy2 = VECTOR_LOAD<float, M2>(box1 + 2, vl);

    vfloat32m2_t b2_xy1 = VECTOR_LOAD<float, M2>(box2, vl);
    vfloat32m2_t b2_xy2 = VECTOR_LOAD<float, M2>(box2 + 2, vl);

    vfloat32m2_t inter_xy1 = VECTOR_MAX<float, M2>(b1_xy1, b2_xy1, vl);
    vfloat32m2_t inter_xy2 = VECTOR_MIN<float, M2>(b1_xy2, b2_xy2, vl);

    vfloat32m2_t inter_wh = VECTOR_SUB<float, M2>(inter_xy2, inter_xy1, vl);
    inter_wh = VECTOR_MAX<float, M2>(inter_wh, 0.0f, vl);

    float inter_w = VECTOR_EXTRACT_SCALAR<float, M2>(VECTOR_SLIDEDOWN<float, M2>(inter_wh, 1, vl));
    float inter_h = VECTOR_EXTRACT_SCALAR<float, M2>(inter_wh);
    float inter_area = inter_w * inter_h;

    vfloat32m2_t b1_wh = VECTOR_SUB<float, M2>(b1_xy2, b1_xy1, vl);
    float area1 = VECTOR_EXTRACT_SCALAR<float, M2>(b1_wh) * VECTOR_EXTRACT_SCALAR<float, M2>(VECTOR_SLIDEDOWN<float, M2>(b1_wh, 1, vl));

    vfloat32m2_t b2_wh = VECTOR_SUB<float, M2>(b2_xy2, b2_xy1, vl);
    float area2 = VECTOR_EXTRACT_SCALAR<float, M2>(b2_wh) * VECTOR_EXTRACT_SCALAR<float, M2>(VECTOR_SLIDEDOWN<float, M2>(b2_wh, 1, vl));

    float union_area = area1 + area2 - inter_area;

    return union_area > 0 ? inter_area / union_area : 0;
}

static void filter_scores_by_threshold_m2(
    const float* scores,
    size_t batch, size_t cls, size_t spatial_dimension,
    size_t num_classes,
    float score_threshold,
    ScoreIndexVector* score_index_pairs
) {
    for (size_t i = 0; i < spatial_dimension; i += SET_VECTOR_LENGTH_MAX<float, M2>()) {
        size_t vl = SET_VECTOR_LENGTH<float, M2>(spatial_dimension - i);
        size_t score_idx = batch * num_classes * spatial_dimension + cls * spatial_dimension + i;

        vfloat32m2_t vscores = VECTOR_LOAD<float, M2>(&scores[score_idx], vl);
        vfloat32m2_t vthreshold = VECTOR_MOVE<float, M2>(score_threshold, vl);
        vbool16_t mask = VECTOR_GE<float, M2>(vscores, vthreshold, vl);

        size_t count = VECTOR_COUNT_POP(mask, vl);
        if (count > 0) {
            vuint32m2_t all_indices = VECTOR_VID<uint32_t, M2>(vl);
            vuint32m2_t selected_indices_vec = VECTOR_COMPRESS<uint32_t, M2>(all_indices, mask, vl);
            uint32_t* indices_arr = (uint32_t*)malloc(count * sizeof(uint32_t));
            VECTOR_STORE<uint32_t, M2>(indices_arr, selected_indices_vec, count);
            for (size_t k = 0; k < count; k++) {
                size_t j = indices_arr[k];
                ScoreIndexPair pair = {scores[score_idx + j], i + j};
                push_score_index(score_index_pairs, pair);
            }
            free(indices_arr);
        }
    }
}

static void apply_greedy_suppression_m2(
    const float* boxes,
    const ScoreIndexVector* score_index_pairs,
    size_t batch, size_t cls, size_t spatial_dimension,
    int64_t max_output_boxes_per_class,
    float iou_threshold,
    int center_point_box,
    SelectedIndexVector* selected_indices
) {
    const size_t n = score_index_pairs->size;
    if (n == 0) return;
    
    char* suppressed = (char*)calloc(n, sizeof(char));
    int64_t selected_count = 0;
    
    float* box_corners = (float*)malloc(n * 4 * sizeof(float));
    for (size_t i = 0; i < n; i++) {
        size_t box_idx = score_index_pairs->data[i].index;
        const float* box = &boxes[batch * spatial_dimension * 4 + box_idx * 4];
        
        if (center_point_box == CENTER_FORMAT) {
            convert_box_format_rvv(box, &box_corners[i * 4], CENTER_FORMAT, CORNER_FORMAT);
        } else {
            memcpy(&box_corners[i * 4], box, 4 * sizeof(float));
        }
    }

    for (size_t i = 0; i < n && selected_count < max_output_boxes_per_class; i++) {
        if (suppressed[i]) continue;

        size_t box_idx = score_index_pairs->data[i].index;
        SelectedIndex sel = {(int64_t)batch, (int64_t)cls, (int64_t)box_idx};
        push_selected_index(selected_indices, sel);
        selected_count++;

        if (selected_count >= max_output_boxes_per_class) break;

        const float* current_box = &box_corners[i * 4];
        
        size_t j = i + 1;
        
        while (j < n) {
            size_t batch_end = (j + 8 < n) ? (j + 8) : n;
            
            for (size_t k = j; k < batch_end; k++) {
                if (suppressed[k]) continue;
                
                const float* other_box = &box_corners[k * 4];
                
                if (!can_boxes_overlap(current_box, other_box)) {
                    continue;
                }
                
                float iou = compute_iou_rvv_m2(current_box, other_box, CORNER_FORMAT);
                
                if (iou > iou_threshold) {
                    suppressed[k] = 1;
                }
            }
            
            j = batch_end;
        }
    }
    
    free(box_corners);
    free(suppressed);
}

SelectedIndexVector nms_e32m2(
    const float* boxes, const float* scores,
    size_t num_batches, size_t num_classes, size_t spatial_dimension,
    int64_t max_output_boxes_per_class, float iou_threshold, float score_threshold,
    int center_point_box
) {
    SelectedIndexVector selected_indices;
    init_selected_vector(&selected_indices);

    if (max_output_boxes_per_class == 0) {
        return selected_indices;
    }

    for (size_t batch = 0; batch < num_batches; batch++) {
        for (size_t cls = 0; cls < num_classes; cls++) {
            ScoreIndexVector score_index_pairs;
            init_score_vector(&score_index_pairs);
            filter_scores_by_threshold_m2(scores, batch, cls, spatial_dimension, 
                                         num_classes, score_threshold, &score_index_pairs);

            sort_by_score_descending(&score_index_pairs);

            apply_greedy_suppression_m2(boxes, &score_index_pairs, batch, cls, spatial_dimension,
                                       max_output_boxes_per_class, iou_threshold, 
                                       center_point_box, &selected_indices);
            
            free_score_vector(&score_index_pairs);
        }
    }

    return selected_indices;
}

/* ============================================================================
 * M4 IMPLEMENTATIONS (Duplicated with M4 types)
 * ============================================================================ */

static float compute_iou_rvv_m4(const float* box1, const float* box2, int center_point_box) {
    float converted_box1[4], converted_box2[4];

    if (center_point_box == CENTER_FORMAT) {
        convert_box_format_rvv(box1, converted_box1, CENTER_FORMAT, CORNER_FORMAT);
        convert_box_format_rvv(box2, converted_box2, CENTER_FORMAT, CORNER_FORMAT);
        box1 = converted_box1;
        box2 = converted_box2;
    }

    size_t vl = 2;
    vfloat32m4_t b1_xy1 = VECTOR_LOAD<float, M4>(box1, vl);
    vfloat32m4_t b1_xy2 = VECTOR_LOAD<float, M4>(box1 + 2, vl);

    vfloat32m4_t b2_xy1 = VECTOR_LOAD<float, M4>(box2, vl);
    vfloat32m4_t b2_xy2 = VECTOR_LOAD<float, M4>(box2 + 2, vl);

    vfloat32m4_t inter_xy1 = VECTOR_MAX<float, M4>(b1_xy1, b2_xy1, vl);
    vfloat32m4_t inter_xy2 = VECTOR_MIN<float, M4>(b1_xy2, b2_xy2, vl);

    vfloat32m4_t inter_wh = VECTOR_SUB<float, M4>(inter_xy2, inter_xy1, vl);
    inter_wh = VECTOR_MAX<float, M4>(inter_wh, 0.0f, vl);

    float inter_w = VECTOR_EXTRACT_SCALAR<float, M4>(VECTOR_SLIDEDOWN<float, M4>(inter_wh, 1, vl));
    float inter_h = VECTOR_EXTRACT_SCALAR<float, M4>(inter_wh);
    float inter_area = inter_w * inter_h;

    vfloat32m4_t b1_wh = VECTOR_SUB<float, M4>(b1_xy2, b1_xy1, vl);
    float area1 = VECTOR_EXTRACT_SCALAR<float, M4>(b1_wh) * VECTOR_EXTRACT_SCALAR<float, M4>(VECTOR_SLIDEDOWN<float, M4>(b1_wh, 1, vl));

    vfloat32m4_t b2_wh = VECTOR_SUB<float, M4>(b2_xy2, b2_xy1, vl);
    float area2 = VECTOR_EXTRACT_SCALAR<float, M4>(b2_wh) * VECTOR_EXTRACT_SCALAR<float, M4>(VECTOR_SLIDEDOWN<float, M4>(b2_wh, 1, vl));

    float union_area = area1 + area2 - inter_area;

    return union_area > 0 ? inter_area / union_area : 0;
}

static void filter_scores_by_threshold_m4(
    const float* scores,
    size_t batch, size_t cls, size_t spatial_dimension,
    size_t num_classes,
    float score_threshold,
    ScoreIndexVector* score_index_pairs
) {
    for (size_t i = 0; i < spatial_dimension; i += SET_VECTOR_LENGTH_MAX<float, M4>()) {
        size_t vl = SET_VECTOR_LENGTH<float, M4>(spatial_dimension - i);
        size_t score_idx = batch * num_classes * spatial_dimension + cls * spatial_dimension + i;

        vfloat32m4_t vscores = VECTOR_LOAD<float, M4>(&scores[score_idx], vl);
        vfloat32m4_t vthreshold = VECTOR_MOVE<float, M4>(score_threshold, vl);
        vbool8_t mask = VECTOR_GE<float, M4>(vscores, vthreshold, vl);

        size_t count = VECTOR_COUNT_POP(mask, vl);
        if (count > 0) {
            vuint32m4_t all_indices = VECTOR_VID<uint32_t, M4>(vl);
            vuint32m4_t selected_indices_vec = VECTOR_COMPRESS<uint32_t, M4>(all_indices, mask, vl);
            uint32_t* indices_arr = (uint32_t*)malloc(count * sizeof(uint32_t));
            VECTOR_STORE<uint32_t, M4>(indices_arr, selected_indices_vec, count);
            for (size_t k = 0; k < count; k++) {
                size_t j = indices_arr[k];
                ScoreIndexPair pair = {scores[score_idx + j], i + j};
                push_score_index(score_index_pairs, pair);
            }
            free(indices_arr);
        }
    }
}

static void apply_greedy_suppression_m4(
    const float* boxes,
    const ScoreIndexVector* score_index_pairs,
    size_t batch, size_t cls, size_t spatial_dimension,
    int64_t max_output_boxes_per_class,
    float iou_threshold,
    int center_point_box,
    SelectedIndexVector* selected_indices
) {
    const size_t n = score_index_pairs->size;
    if (n == 0) return;
    
    char* suppressed = (char*)calloc(n, sizeof(char));
    int64_t selected_count = 0;
    
    float* box_corners = (float*)malloc(n * 4 * sizeof(float));
    for (size_t i = 0; i < n; i++) {
        size_t box_idx = score_index_pairs->data[i].index;
        const float* box = &boxes[batch * spatial_dimension * 4 + box_idx * 4];
        
        if (center_point_box == CENTER_FORMAT) {
            convert_box_format_rvv(box, &box_corners[i * 4], CENTER_FORMAT, CORNER_FORMAT);
        } else {
            memcpy(&box_corners[i * 4], box, 4 * sizeof(float));
        }
    }

    for (size_t i = 0; i < n && selected_count < max_output_boxes_per_class; i++) {
        if (suppressed[i]) continue;

        size_t box_idx = score_index_pairs->data[i].index;
        SelectedIndex sel = {(int64_t)batch, (int64_t)cls, (int64_t)box_idx};
        push_selected_index(selected_indices, sel);
        selected_count++;

        if (selected_count >= max_output_boxes_per_class) break;

        const float* current_box = &box_corners[i * 4];
        
        size_t j = i + 1;
        
        while (j < n) {
            size_t batch_end = (j + 8 < n) ? (j + 8) : n;
            
            for (size_t k = j; k < batch_end; k++) {
                if (suppressed[k]) continue;
                
                const float* other_box = &box_corners[k * 4];
                
                if (!can_boxes_overlap(current_box, other_box)) {
                    continue;
                }
                
                float iou = compute_iou_rvv_m4(current_box, other_box, CORNER_FORMAT);
                
                if (iou > iou_threshold) {
                    suppressed[k] = 1;
                }
            }
            
            j = batch_end;
        }
    }
    
    free(box_corners);
    free(suppressed);
}

SelectedIndexVector nms_e32m4(
    const float* boxes, const float* scores,
    size_t num_batches, size_t num_classes, size_t spatial_dimension,
    int64_t max_output_boxes_per_class, float iou_threshold, float score_threshold,
    int center_point_box
) {
    SelectedIndexVector selected_indices;
    init_selected_vector(&selected_indices);

    if (max_output_boxes_per_class == 0) {
        return selected_indices;
    }

    for (size_t batch = 0; batch < num_batches; batch++) {
        for (size_t cls = 0; cls < num_classes; cls++) {
            ScoreIndexVector score_index_pairs;
            init_score_vector(&score_index_pairs);
            filter_scores_by_threshold_m4(scores, batch, cls, spatial_dimension, 
                                         num_classes, score_threshold, &score_index_pairs);

            sort_by_score_descending(&score_index_pairs);

            apply_greedy_suppression_m4(boxes, &score_index_pairs, batch, cls, spatial_dimension,
                                       max_output_boxes_per_class, iou_threshold, 
                                       center_point_box, &selected_indices);
            
            free_score_vector(&score_index_pairs);
        }
    }

    return selected_indices;
}

/* ============================================================================
 * M8 IMPLEMENTATIONS (Duplicated with M8 types)
 * ============================================================================ */

static float compute_iou_rvv_m8(const float* box1, const float* box2, int center_point_box) {
    float converted_box1[4], converted_box2[4];

    if (center_point_box == CENTER_FORMAT) {
        convert_box_format_rvv(box1, converted_box1, CENTER_FORMAT, CORNER_FORMAT);
        convert_box_format_rvv(box2, converted_box2, CENTER_FORMAT, CORNER_FORMAT);
        box1 = converted_box1;
        box2 = converted_box2;
    }

    size_t vl = 2;
    vfloat32m8_t b1_xy1 = VECTOR_LOAD<float, M8>(box1, vl);
    vfloat32m8_t b1_xy2 = VECTOR_LOAD<float, M8>(box1 + 2, vl);

    vfloat32m8_t b2_xy1 = VECTOR_LOAD<float, M8>(box2, vl);
    vfloat32m8_t b2_xy2 = VECTOR_LOAD<float, M8>(box2 + 2, vl);

    vfloat32m8_t inter_xy1 = VECTOR_MAX<float, M8>(b1_xy1, b2_xy1, vl);
    vfloat32m8_t inter_xy2 = VECTOR_MIN<float, M8>(b1_xy2, b2_xy2, vl);

    vfloat32m8_t inter_wh = VECTOR_SUB<float, M8>(inter_xy2, inter_xy1, vl);
    inter_wh = VECTOR_MAX<float, M8>(inter_wh, 0.0f, vl);

    float inter_w = VECTOR_EXTRACT_SCALAR<float, M8>(VECTOR_SLIDEDOWN<float, M8>(inter_wh, 1, vl));
    float inter_h = VECTOR_EXTRACT_SCALAR<float, M8>(inter_wh);
    float inter_area = inter_w * inter_h;

    vfloat32m8_t b1_wh = VECTOR_SUB<float, M8>(b1_xy2, b1_xy1, vl);
    float area1 = VECTOR_EXTRACT_SCALAR<float, M8>(b1_wh) * VECTOR_EXTRACT_SCALAR<float, M8>(VECTOR_SLIDEDOWN<float, M8>(b1_wh, 1, vl));

    vfloat32m8_t b2_wh = VECTOR_SUB<float, M8>(b2_xy2, b2_xy1, vl);
    float area2 = VECTOR_EXTRACT_SCALAR<float, M8>(b2_wh) * VECTOR_EXTRACT_SCALAR<float, M8>(VECTOR_SLIDEDOWN<float, M8>(b2_wh, 1, vl));

    float union_area = area1 + area2 - inter_area;

    return union_area > 0 ? inter_area / union_area : 0;
}

static void filter_scores_by_threshold_m8(
    const float* scores,
    size_t batch, size_t cls, size_t spatial_dimension,
    size_t num_classes,
    float score_threshold,
    ScoreIndexVector* score_index_pairs
) {
    for (size_t i = 0; i < spatial_dimension; i += SET_VECTOR_LENGTH_MAX<float, M8>()) {
        size_t vl = SET_VECTOR_LENGTH<float, M8>(spatial_dimension - i);
        size_t score_idx = batch * num_classes * spatial_dimension + cls * spatial_dimension + i;

        vfloat32m8_t vscores = VECTOR_LOAD<float, M8>(&scores[score_idx], vl);
        vfloat32m8_t vthreshold = VECTOR_MOVE<float, M8>(score_threshold, vl);
        vbool4_t mask = VECTOR_GE<float, M8>(vscores, vthreshold, vl);

        size_t count = VECTOR_COUNT_POP(mask, vl);
        if (count > 0) {
            vuint32m8_t all_indices = VECTOR_VID<uint32_t, M8>(vl);
            vuint32m8_t selected_indices_vec = VECTOR_COMPRESS<uint32_t, M8>(all_indices, mask, vl);
            uint32_t* indices_arr = (uint32_t*)malloc(count * sizeof(uint32_t));
            VECTOR_STORE<uint32_t, M8>(indices_arr, selected_indices_vec, count);
            for (size_t k = 0; k < count; k++) {
                size_t j = indices_arr[k];
                ScoreIndexPair pair = {scores[score_idx + j], i + j};
                push_score_index(score_index_pairs, pair);
            }
            free(indices_arr);
        }
    }
}

static void apply_greedy_suppression_m8(
    const float* boxes,
    const ScoreIndexVector* score_index_pairs,
    size_t batch, size_t cls, size_t spatial_dimension,
    int64_t max_output_boxes_per_class,
    float iou_threshold,
    int center_point_box,
    SelectedIndexVector* selected_indices
) {
    const size_t n = score_index_pairs->size;
    if (n == 0) return;
    
    char* suppressed = (char*)calloc(n, sizeof(char));
    int64_t selected_count = 0;
    
    float* box_corners = (float*)malloc(n * 4 * sizeof(float));
    for (size_t i = 0; i < n; i++) {
        size_t box_idx = score_index_pairs->data[i].index;
        const float* box = &boxes[batch * spatial_dimension * 4 + box_idx * 4];
        
        if (center_point_box == CENTER_FORMAT) {
            convert_box_format_rvv(box, &box_corners[i * 4], CENTER_FORMAT, CORNER_FORMAT);
        } else {
            memcpy(&box_corners[i * 4], box, 4 * sizeof(float));
        }
    }

    for (size_t i = 0; i < n && selected_count < max_output_boxes_per_class; i++) {
        if (suppressed[i]) continue;

        size_t box_idx = score_index_pairs->data[i].index;
        SelectedIndex sel = {(int64_t)batch, (int64_t)cls, (int64_t)box_idx};
        push_selected_index(selected_indices, sel);
        selected_count++;

        if (selected_count >= max_output_boxes_per_class) break;

        const float* current_box = &box_corners[i * 4];
        
        size_t j = i + 1;
        
        while (j < n) {
            size_t batch_end = (j + 8 < n) ? (j + 8) : n;
            
            for (size_t k = j; k < batch_end; k++) {
                if (suppressed[k]) continue;
                
                const float* other_box = &box_corners[k * 4];
                
                if (!can_boxes_overlap(current_box, other_box)) {
                    continue;
                }
                
                float iou = compute_iou_rvv_m8(current_box, other_box, CORNER_FORMAT);
                
                if (iou > iou_threshold) {
                    suppressed[k] = 1;
                }
            }
            
            j = batch_end;
        }
    }
    
    free(box_corners);
    free(suppressed);
}

SelectedIndexVector nms_e32m8(
    const float* boxes, const float* scores,
    size_t num_batches, size_t num_classes, size_t spatial_dimension,
    int64_t max_output_boxes_per_class, float iou_threshold, float score_threshold,
    int center_point_box
) {
    SelectedIndexVector selected_indices;
    init_selected_vector(&selected_indices);

    if (max_output_boxes_per_class == 0) {
        return selected_indices;
    }

    for (size_t batch = 0; batch < num_batches; batch++) {
        for (size_t cls = 0; cls < num_classes; cls++) {
            ScoreIndexVector score_index_pairs;
            init_score_vector(&score_index_pairs);
            filter_scores_by_threshold_m8(scores, batch, cls, spatial_dimension, 
                                         num_classes, score_threshold, &score_index_pairs);

            sort_by_score_descending(&score_index_pairs);

            apply_greedy_suppression_m8(boxes, &score_index_pairs, batch, cls, spatial_dimension,
                                       max_output_boxes_per_class, iou_threshold, 
                                       center_point_box, &selected_indices);
            
            free_score_vector(&score_index_pairs);
        }
    }

    return selected_indices;
}
