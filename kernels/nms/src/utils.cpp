#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <riscv_vector.h>
#include "../include/defs.h"
#include "../../../lib/rvv_defs.hpp"

// Helper functions for vector management
void init_selected_vector(SelectedIndexVector* vec) {
    vec->data = NULL;
    vec->size = 0;
    vec->capacity = 0;
}

void push_selected_index(SelectedIndexVector* vec, SelectedIndex item) {
    if (vec->size >= vec->capacity) {
        size_t new_capacity = vec->capacity == 0 ? 8 : vec->capacity * 2;
        SelectedIndex* new_data = (SelectedIndex*)realloc(vec->data, new_capacity * sizeof(SelectedIndex));
        if (!new_data) return;
        vec->data = new_data;
        vec->capacity = new_capacity;
    }
    vec->data[vec->size++] = item;
}

void free_selected_vector(SelectedIndexVector* vec) {
    if (vec->data) free(vec->data);
    vec->data = NULL;
    vec->size = 0;
    vec->capacity = 0;
}

void init_score_vector(ScoreIndexVector* vec) {
    vec->data = NULL;
    vec->size = 0;
    vec->capacity = 0;
}

void push_score_index(ScoreIndexVector* vec, ScoreIndexPair item) {
    if (vec->size >= vec->capacity) {
        size_t new_capacity = vec->capacity == 0 ? 8 : vec->capacity * 2;
        ScoreIndexPair* new_data = (ScoreIndexPair*)realloc(vec->data, new_capacity * sizeof(ScoreIndexPair));
        if (!new_data) return;
        vec->data = new_data;
        vec->capacity = new_capacity;
    }
    vec->data[vec->size++] = item;
}

void free_score_vector(ScoreIndexVector* vec) {
    if (vec->data) free(vec->data);
    vec->data = NULL;
    vec->size = 0;
    vec->capacity = 0;
}

// Comparison function for qsort
int compare_scores_desc(const void* a, const void* b) {
    const ScoreIndexPair* pa = (const ScoreIndexPair*)a;
    const ScoreIndexPair* pb = (const ScoreIndexPair*)b;
    if (pa->score > pb->score) return -1;
    if (pa->score < pb->score) return 1;
    return 0;
}

// Helper functions for max/min
static float fmaxf_inline(float a, float b) { return a > b ? a : b; }
static float fminf_inline(float a, float b) { return a < b ? a : b; }

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
    float inter_y1 = fmaxf_inline(y1_1, y1_2);
    float inter_x1 = fmaxf_inline(x1_1, x1_2);
    float inter_y2 = fminf_inline(y2_1, y2_2);
    float inter_x2 = fminf_inline(x2_1, x2_2);
    
    // Compute intersection area
    float inter_width = fmaxf_inline(0.0f, inter_x2 - inter_x1);
    float inter_height = fmaxf_inline(0.0f, inter_y2 - inter_y1);
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
        vfloat32m1_t v_box = VECTOR_LOAD<float, M1>(box, vl);
        VECTOR_STORE<float, M1>(converted_box, v_box, vl);
        return;
    }
    
    size_t vl = 4;
    vfloat32m1_t v_box = VECTOR_LOAD<float, M1>(box, vl);
    
    if (from_format == CENTER_FORMAT && to_format == CORNER_FORMAT) {
        // Convert from [x_center, y_center, width, height] to [y1, x1, y2, x2]
        // Load: [x_center, y_center, width, height]
        float x_center = VECTOR_EXTRACT_SCALAR<float, M1>(v_box);
        float y_center = VECTOR_EXTRACT_SCALAR<float, M1>(VECTOR_SLIDEDOWN<float, M1>(v_box, 1, vl));
        float width = VECTOR_EXTRACT_SCALAR<float, M1>(VECTOR_SLIDEDOWN<float, M1>(v_box, 2, vl));
        float height = VECTOR_EXTRACT_SCALAR<float, M1>(VECTOR_SLIDEDOWN<float, M1>(v_box, 3, vl));
        
        // Create vector [width/2, height/2, width/2, height/2]
        float half_dims[4] = {width * 0.5f, height * 0.5f, width * 0.5f, height * 0.5f};
        vfloat32m1_t v_half_dims = VECTOR_LOAD<float, M1>(half_dims, vl);
        
        // Create center vector [x_center, y_center, x_center, y_center]
        float centers[4] = {x_center, y_center, x_center, y_center};
        vfloat32m1_t v_centers = VECTOR_LOAD<float, M1>(centers, vl);
        
        // Compute [x_center - width/2, y_center - height/2, x_center + width/2, y_center + height/2]
        float signs[4] = {-1.0f, -1.0f, 1.0f, 1.0f};
        vfloat32m1_t v_signs = VECTOR_LOAD<float, M1>(signs, vl);
        vfloat32m1_t v_result_temp = VECTOR_FMACC<float, M1>(v_centers, v_half_dims, v_signs, vl);
        
        // Swap to get [y1, x1, y2, x2] from [x1, y1, x2, y2]
        float result_temp[4];
        VECTOR_STORE<float, M1>(result_temp, v_result_temp, vl);
        converted_box[0] = result_temp[1];  // y1
        converted_box[1] = result_temp[0];  // x1
        converted_box[2] = result_temp[3];  // y2
        converted_box[3] = result_temp[2];  // x2
        
    } else if (from_format == CORNER_FORMAT && to_format == CENTER_FORMAT) {
        // Convert from [y1, x1, y2, x2] to [x_center, y_center, width, height]
        float y1 = VECTOR_EXTRACT_SCALAR<float, M1>(v_box);
        float x1 = VECTOR_EXTRACT_SCALAR<float, M1>(VECTOR_SLIDEDOWN<float, M1>(v_box, 1, vl));
        float y2 = VECTOR_EXTRACT_SCALAR<float, M1>(VECTOR_SLIDEDOWN<float, M1>(v_box, 2, vl));
        float x2 = VECTOR_EXTRACT_SCALAR<float, M1>(VECTOR_SLIDEDOWN<float, M1>(v_box, 3, vl));
        
        // Create vectors [x1, y1, x2, y2] and [x2, y2, x1, y1]
        float corners1[4] = {x1, y1, x2, y2};
        float corners2[4] = {x2, y2, x1, y1};
        vfloat32m1_t v_corners1 = VECTOR_LOAD<float, M1>(corners1, vl);
        vfloat32m1_t v_corners2 = VECTOR_LOAD<float, M1>(corners2, vl);
        
        // Sum: [x1+x2, y1+y2, x2+x1, y2+y1] (but we only need first two for center)
        vfloat32m1_t v_sum = VECTOR_ADD<float, M1>(v_corners1, v_corners2, vl);
        
        // Difference: [x2-x1, y2-y1, x1-x2, y1-y2] (we only need first two for width/height)
        vfloat32m1_t v_diff = VECTOR_SUB<float, M1>(v_corners2, v_corners1, vl);
        
        // Divide by 2 for centers
        vfloat32m1_t v_half = VECTOR_MOVE<float, M1>(0.5f, vl);
        vfloat32m1_t v_center = VECTOR_MUL<float, M1>(v_sum, v_half, vl);
        
        // Result: [x_center, y_center, width, height]
        VECTOR_STORE<float, M1>(converted_box, v_center, 2);      // Store centers
        VECTOR_STORE<float, M1>(converted_box + 2, v_diff, 2);    // Store dimensions
    }
}

// Scalar NMS implementation for reference
SelectedIndexVector nms_scalar(
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
            // Create vector of (score, index) pairs for current batch and class
            ScoreIndexVector score_index_pairs;
            init_score_vector(&score_index_pairs);
            
            // Filter scores above threshold
            for (size_t i = 0; i < spatial_dimension; i++) {
                size_t score_idx = batch * num_classes * spatial_dimension + cls * spatial_dimension + i;
                float score = scores[score_idx];
                
                if (score >= score_threshold) {
                    ScoreIndexPair pair = {score, i};
                    push_score_index(&score_index_pairs, pair);
                }
            }
            
            // Sort by score in descending order
            qsort(score_index_pairs.data, score_index_pairs.size, sizeof(ScoreIndexPair), compare_scores_desc);
            
            char* suppressed = (char*)calloc(score_index_pairs.size, sizeof(char));
            int64_t selected_count = 0;
            
            // Apply NMS
            for (size_t i = 0; i < score_index_pairs.size && selected_count < max_output_boxes_per_class; i++) {
                if (suppressed[i]) continue;
                
                size_t box_idx = score_index_pairs.data[i].index;
                SelectedIndex sel = {(int64_t)batch, (int64_t)cls, (int64_t)box_idx};
                push_selected_index(&selected_indices, sel);
                selected_count++;
                
                // Suppress overlapping boxes
                const float* current_box = &boxes[batch * spatial_dimension * 4 + box_idx * 4];
                
                for (size_t j = i + 1; j < score_index_pairs.size; j++) {
                    if (suppressed[j]) continue;
                    
                    size_t other_box_idx = score_index_pairs.data[j].index;
                    const float* other_box = &boxes[batch * spatial_dimension * 4 + other_box_idx * 4];
                    
                    float iou = compute_iou(current_box, other_box, center_point_box);
                    if (iou > iou_threshold) {
                        suppressed[j] = 1;
                    }
                }
            }
            
            free(suppressed);
            free_score_vector(&score_index_pairs);
        }
    }
    
    return selected_indices;
}

// Write NMS results to text file
void write_nms_results_to_file(const char* filename, const SelectedIndexVector* results) {
    FILE* file = fopen(filename, "w");
    if (!file) {
        fprintf(stderr, "Error: Cannot open file %s for writing.\n", filename);
        return;
    }
    
    fprintf(file, "Selected indices (batch_index, class_index, box_index):\n");
    for (size_t i = 0; i < results->size; i++) {
        fprintf(file, "%lld, %lld, %lld\n", 
                (long long)results->data[i].batch_index,
                (long long)results->data[i].class_index,
                (long long)results->data[i].box_index);
    }
    
    fclose(file);
    printf("NMS results written to %s\n", filename);
}

// Write NMS results to binary file
void write_nms_results_binary(const char* filename, const SelectedIndexVector* results) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        fprintf(stderr, "Error: Cannot open file %s for writing.\n", filename);
        return;
    }
    
    size_t count = results->size;
    fwrite(&count, sizeof(count), 1, file);
    
    for (size_t i = 0; i < results->size; i++) {
        fwrite(&results->data[i].batch_index, sizeof(results->data[i].batch_index), 1, file);
        fwrite(&results->data[i].class_index, sizeof(results->data[i].class_index), 1, file);
        fwrite(&results->data[i].box_index, sizeof(results->data[i].box_index), 1, file);
    }
    
    fclose(file);
}