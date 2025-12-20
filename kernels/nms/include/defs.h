#ifndef DEFS_H
#define DEFS_H

#include <cstddef>
#include <vector>

// Box format constants
#define CORNER_FORMAT 0  // [y1, x1, y2, x2]
#define CENTER_FORMAT 1  // [x_center, y_center, width, height]

// Structure to hold selected indices
struct SelectedIndex {
    int64_t batch_index;
    int64_t class_index;
    int64_t box_index;
};

// NMS functions with different RVV configurations
std::vector<SelectedIndex> nms_e32m1(
    const float* boxes, const float* scores,
    std::size_t num_batches, std::size_t num_classes, std::size_t spatial_dimension,
    int64_t max_output_boxes_per_class, float iou_threshold, float score_threshold,
    int center_point_box
);

std::vector<SelectedIndex> nms_e32m2(
    const float* boxes, const float* scores,
    std::size_t num_batches, std::size_t num_classes, std::size_t spatial_dimension,
    int64_t max_output_boxes_per_class, float iou_threshold, float score_threshold,
    int center_point_box
);

std::vector<SelectedIndex> nms_e32m4(
    const float* boxes, const float* scores,
    std::size_t num_batches, std::size_t num_classes, std::size_t spatial_dimension,
    int64_t max_output_boxes_per_class, float iou_threshold, float score_threshold,
    int center_point_box
);

std::vector<SelectedIndex> nms_e32m8(
    const float* boxes, const float* scores,
    std::size_t num_batches, std::size_t num_classes, std::size_t spatial_dimension,
    int64_t max_output_boxes_per_class, float iou_threshold, float score_threshold,
    int center_point_box
);

// Scalar reference implementation
std::vector<SelectedIndex> nms_scalar(
    const float* boxes, const float* scores,
    std::size_t num_batches, std::size_t num_classes, std::size_t spatial_dimension,
    int64_t max_output_boxes_per_class, float iou_threshold, float score_threshold,
    int center_point_box
);

// Utility functions
void write_nms_results_to_file(const char* filename, const std::vector<SelectedIndex>& results);
void write_nms_results_binary(const char* filename, const std::vector<SelectedIndex>& results);
float compute_iou(const float* box1, const float* box2, int center_point_box);
void convert_box_format(const float* box, float* converted_box, int from_format, int to_format);
void convert_box_format_rvv(const float* box, float* converted_box, int from_format, int to_format);

#endif