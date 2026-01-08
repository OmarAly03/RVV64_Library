#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/stat.h>
#include <sys/types.h>
#include "./include/defs.h"

// Simple random number generator replacement
static unsigned int g_seed = 42;

static void srand_custom(unsigned int seed) {
    g_seed = seed;
}

static float rand_float_range(float min, float max) {
    g_seed = g_seed * 1103515245 + 12345;
    return min + ((float)((g_seed / 65536) % 32768)) / 32767.0f * (max - min);
}

static long long get_time_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

int main(int argc, char* argv[]) {
    // --- HANDLE ARGUMENTS ---
    size_t num_batches = 1, num_classes = 2, spatial_dimension = 100;
    int64_t max_output_boxes_per_class = 50;
    float iou_threshold = 0.5f;
    float score_threshold = 0.1f;
    int center_point_box = CORNER_FORMAT;
    
    if (argc >= 4) {
        int batches = atoi(argv[1]);
        int classes = atoi(argv[2]);
        int spatial = atoi(argv[3]);
        
        if (batches > 0 && classes > 0 && spatial > 0) {
            num_batches = (size_t)batches;
            num_classes = (size_t)classes;
            spatial_dimension = (size_t)spatial;
        }
    }
    
    if (argc >= 5) {
        max_output_boxes_per_class = (int64_t)atoi(argv[4]);
    }
    
    if (argc >= 6) {
        iou_threshold = atof(argv[5]);
    }
    
    if (argc >= 7) {
        score_threshold = atof(argv[6]);
    }
    
    // --- MEMORY ALLOCATION ---
    size_t boxes_size = num_batches * spatial_dimension * 4;
    size_t scores_size = num_batches * num_classes * spatial_dimension;
    
    float* boxes = (float*)malloc(boxes_size * sizeof(float));
    float* scores = (float*)malloc(scores_size * sizeof(float));
    
    if (!boxes || !scores) {
        fprintf(stderr, "Memory allocation failed!\n");
        return 1;
    }
    
    // --- INITIALIZE DATA ---
    srand_custom(42); // Fixed seed for reproducibility
    
    for (size_t i = 0; i < boxes_size; i += 4) {
        float y1 = rand_float_range(0.0f, 100.0f);
        float x1 = rand_float_range(0.0f, 100.0f);
        float height = rand_float_range(1.0f, 20.0f);
        float width = rand_float_range(1.0f, 20.0f);
        
        boxes[i] = y1;
        boxes[i + 1] = x1;
        boxes[i + 2] = y1 + height;
        boxes[i + 3] = x1 + width;
    }
    
    for (size_t i = 0; i < scores_size; i++) {
        scores[i] = rand_float_range(0.0f, 1.0f);
    }
    
    system("mkdir -p ./output_files");
    
    FILE* boxes_file = fopen("./output_files/boxes.bin", "wb");
    fwrite(boxes, sizeof(float), boxes_size, boxes_file);
    fclose(boxes_file);
    
    FILE* scores_file = fopen("./output_files/scores.bin", "wb");
    fwrite(scores, sizeof(float), scores_size, scores_file);
    fclose(scores_file);
    
    SelectedIndexVector result_scalar = nms_scalar(boxes, scores, num_batches, num_classes, spatial_dimension,
                                   max_output_boxes_per_class, iou_threshold, score_threshold, center_point_box);
    write_nms_results_binary("./output_files/nms_scalar.bin", &result_scalar);
    free_selected_vector(&result_scalar);
    
    SelectedIndexVector result_m1 = nms_e32m1(boxes, scores, num_batches, num_classes, spatial_dimension,
                               max_output_boxes_per_class, iou_threshold, score_threshold, center_point_box);
    write_nms_results_binary("./output_files/nms_e32m1.bin", &result_m1);
    free_selected_vector(&result_m1);
    
    SelectedIndexVector result_m2 = nms_e32m2(boxes, scores, num_batches, num_classes, spatial_dimension,
                               max_output_boxes_per_class, iou_threshold, score_threshold, center_point_box);
    write_nms_results_binary("./output_files/nms_e32m2.bin", &result_m2);
    free_selected_vector(&result_m2);
    
    SelectedIndexVector result_m4 = nms_e32m4(boxes, scores, num_batches, num_classes, spatial_dimension,
                               max_output_boxes_per_class, iou_threshold, score_threshold, center_point_box);
    write_nms_results_binary("./output_files/nms_e32m4.bin", &result_m4);
    free_selected_vector(&result_m4);
    
    SelectedIndexVector result_m8 = nms_e32m8(boxes, scores, num_batches, num_classes, spatial_dimension,
                               max_output_boxes_per_class, iou_threshold, score_threshold, center_point_box);
    write_nms_results_binary("./output_files/nms_e32m8.bin", &result_m8);
    free_selected_vector(&result_m8);
    
    free(boxes);
    free(scores);
    
    return 0;
}
