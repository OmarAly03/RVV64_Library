#include <iostream>
#include <cstdlib>
#include <chrono>
#include <random>
#include <fstream>
#include "./include/defs.h"

using namespace std;
using namespace std::chrono;

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
            num_batches = static_cast<size_t>(batches);
            num_classes = static_cast<size_t>(classes);
            spatial_dimension = static_cast<size_t>(spatial);
        }
    }
    
    if (argc >= 5) {
        max_output_boxes_per_class = static_cast<int64_t>(atoi(argv[4]));
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
    
    float* boxes = new float[boxes_size];
    float* scores = new float[scores_size];
    
    if (!boxes || !scores) {
        cerr << "Memory allocation failed!" << endl;
        return 1;
    }
    
    // --- INITIALIZE DATA ---
    random_device rd;
    mt19937 gen(42); // Fixed seed for reproducibility
    uniform_real_distribution<float> coord_dist(0.0f, 100.0f);
    uniform_real_distribution<float> score_dist(0.0f, 1.0f);
    uniform_real_distribution<float> size_dist(1.0f, 20.0f);
    
    for (size_t i = 0; i < boxes_size; i += 4) {
        float y1 = coord_dist(gen);
        float x1 = coord_dist(gen);
        float height = size_dist(gen);
        float width = size_dist(gen);
        
        boxes[i] = y1;
        boxes[i + 1] = x1;
        boxes[i + 2] = y1 + height;
        boxes[i + 3] = x1 + width;
    }
    
    for (size_t i = 0; i < scores_size; i++) {
        scores[i] = score_dist(gen);
    }
    
    system("mkdir -p ./output_files");
    
    ofstream boxes_file("./output_files/boxes.bin", ios::binary);
    boxes_file.write(reinterpret_cast<const char*>(boxes), boxes_size * sizeof(float));
    boxes_file.close();
    
    ofstream scores_file("./output_files/scores.bin", ios::binary);
    scores_file.write(reinterpret_cast<const char*>(scores), scores_size * sizeof(float));
    scores_file.close();
    
    auto result_scalar = nms_scalar(boxes, scores, num_batches, num_classes, spatial_dimension,
                                   max_output_boxes_per_class, iou_threshold, score_threshold, center_point_box);
    write_nms_results_binary("./output_files/nms_scalar.bin", result_scalar);
    
    auto result_m1 = nms_e32m1(boxes, scores, num_batches, num_classes, spatial_dimension,
                               max_output_boxes_per_class, iou_threshold, score_threshold, center_point_box);
    write_nms_results_binary("./output_files/nms_e32m1.bin", result_m1);
    
    auto result_m2 = nms_e32m2(boxes, scores, num_batches, num_classes, spatial_dimension,
                               max_output_boxes_per_class, iou_threshold, score_threshold, center_point_box);
    write_nms_results_binary("./output_files/nms_e32m2.bin", result_m2);
    
    auto result_m4 = nms_e32m4(boxes, scores, num_batches, num_classes, spatial_dimension,
                               max_output_boxes_per_class, iou_threshold, score_threshold, center_point_box);
    write_nms_results_binary("./output_files/nms_e32m4.bin", result_m4);
    
    auto result_m8 = nms_e32m8(boxes, scores, num_batches, num_classes, spatial_dimension,
                               max_output_boxes_per_class, iou_threshold, score_threshold, center_point_box);
    write_nms_results_binary("./output_files/nms_e32m8.bin", result_m8);
    
    delete[] boxes;
    delete[] scores;
    
    return 0;
}
