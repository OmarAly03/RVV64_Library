#ifndef DEFS_H
#define DEFS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Box format constants
#define CORNER_FORMAT 0  // [y1, x1, y2, x2]
#define CENTER_FORMAT 1  // [x_center, y_center, width, height]

// Structure to hold selected indices
typedef struct {
    int64_t batch_index;
    int64_t class_index;
    int64_t box_index;
} SelectedIndex;

// C-compatible vector structure
typedef struct {
    SelectedIndex* data;
    size_t size;
    size_t capacity;
} SelectedIndexVector;

// Helper structures for internal use
typedef struct {
    float score;
    size_t index;
} ScoreIndexPair;

typedef struct {
    ScoreIndexPair* data;
    size_t size;
    size_t capacity;
} ScoreIndexVector;

// Vector management functions
void init_selected_vector(SelectedIndexVector* vec);
void push_selected_index(SelectedIndexVector* vec, SelectedIndex item);
void free_selected_vector(SelectedIndexVector* vec);

// Internal helper functions for NMS implementation
void init_score_vector(ScoreIndexVector* vec);
void push_score_index(ScoreIndexVector* vec, ScoreIndexPair item);
void free_score_vector(ScoreIndexVector* vec);
int compare_scores_desc(const void* a, const void* b);

// NMS functions with different RVV configurations
SelectedIndexVector nms_e32m1(
    const float* boxes, const float* scores,
    size_t num_batches, size_t num_classes, size_t spatial_dimension,
    int64_t max_output_boxes_per_class, float iou_threshold, float score_threshold,
    int center_point_box
);

SelectedIndexVector nms_e32m2(
    const float* boxes, const float* scores,
    size_t num_batches, size_t num_classes, size_t spatial_dimension,
    int64_t max_output_boxes_per_class, float iou_threshold, float score_threshold,
    int center_point_box
);

SelectedIndexVector nms_e32m4(
    const float* boxes, const float* scores,
    size_t num_batches, size_t num_classes, size_t spatial_dimension,
    int64_t max_output_boxes_per_class, float iou_threshold, float score_threshold,
    int center_point_box
);

SelectedIndexVector nms_e32m8(
    const float* boxes, const float* scores,
    size_t num_batches, size_t num_classes, size_t spatial_dimension,
    int64_t max_output_boxes_per_class, float iou_threshold, float score_threshold,
    int center_point_box
);

// Scalar reference implementation
SelectedIndexVector nms_scalar(
    const float* boxes, const float* scores,
    size_t num_batches, size_t num_classes, size_t spatial_dimension,
    int64_t max_output_boxes_per_class, float iou_threshold, float score_threshold,
    int center_point_box
);

// Utility functions
void write_nms_results_to_file(const char* filename, const SelectedIndexVector* results);
void write_nms_results_binary(const char* filename, const SelectedIndexVector* results);
float compute_iou(const float* box1, const float* box2, int center_point_box);
void convert_box_format(const float* box, float* converted_box, int from_format, int to_format);
void convert_box_format_rvv(const float* box, float* converted_box, int from_format, int to_format);

#ifdef __cplusplus
}
#endif

#endif