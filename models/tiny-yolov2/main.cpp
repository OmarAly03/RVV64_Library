// main.cpp
#include "yolo_model.hpp"
#include <opencv2/opencv.hpp>
#include <chrono>

// Helper to get a cv::Rect from a BoundingBox
static cv::Rect get_rect(const BoundingBox& box) {
    float x_min = box.x - box.w / 2.0f;
    float y_min = box.y - box.h / 2.0f;
    float x_max = box.x + box.w / 2.0f;
    float y_max = box.y + box.h / 2.0f;
    return cv::Rect(cv::Point((int)x_min, (int)y_min), cv::Point((int)x_max, (int)y_max));
}

// Helper function to draw boxes (copied from our previous example)
void scale_and_draw_boxes(cv::Mat& image, const std::vector<BoundingBox>& boxes) {
    float scale_x = (float)image.cols / GRID_W;
    float scale_y = (float)image.rows / GRID_H;

    for (const auto& box : boxes) {
        float x = box.x * scale_x;
        float y = box.y * scale_y;
        float w = box.w * scale_x;
        float h = box.h * scale_y;
        cv::Rect rect = get_rect(BoundingBox{x, y, w, h, 0.0f, box.class_id});
        
        rect.x = std::max(0, rect.x);
        rect.y = std::max(0, rect.y);
        rect.width = std::min(image.cols - rect.x - 1, rect.width);
        rect.height = std::min(image.rows - rect.y - 1, rect.height);

        cv::rectangle(image, rect, cv::Scalar(0, 255, 0), 2); // Green
        std::string label = LABELS[box.class_id];
        std::string text = cv::format("%s: %.2f", label.c_str(), box.score);
        
        int baseline = 0;
        cv::Size text_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 2, &baseline);
        cv::Point text_org(rect.x, rect.y - baseline - 2);

        cv::rectangle(image, text_org + cv::Point(0, baseline + 2), 
                      text_org + cv::Point(text_size.width, -text_size.height), 
                      cv::Scalar(0, 255, 0), cv::FILLED);
        cv::putText(image, text, text_org, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1); // Black text
    }
}


int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <original_image.jpg> <weights_directory>" << std::endl;
        return -1;
    }

    std::string original_image_path = argv[1];
    std::string weights_dir = argv[2];
    std::string input_bin_path = "image_binaries/mark1.bin";
    std::string output_image_path = "output_images/output_mark1.jpg";

    // 1. Load the pre-processed input tensor
    std::cout << "Loading input tensor from " << input_bin_path << "..." << std::endl;
    std::vector<float> input_tensor = load_input_image(input_bin_path);
    if (input_tensor.size() != 1 * 3 * NET_H * NET_W) {
        std::cerr << "Error: Input tensor has wrong size." << std::endl;
        return -1;
    }

    // 2. Load the original image for drawing
    cv::Mat original_image = cv::imread(original_image_path);
    if (original_image.empty()) {
        std::cerr << "Error: Could not load original image from " << original_image_path << std::endl;
        return -1;
    }

    // 3. Load all model weights
    ModelWeights weights;
    load_all_weights(weights, weights_dir);

    // 4. Run inference
    std::cout << "Running inference..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    
    std::vector<BoundingBox> final_boxes = yolo_model_inference(weights, input_tensor);
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "Inference (incl. post-processing) took: " << duration.count() << " ms" << std::endl;

    // 5. Draw boxes and save
    std::cout << "Found " << final_boxes.size() << " final boxes!" << std::endl;
    scale_and_draw_boxes(original_image, final_boxes);
    cv::imwrite(output_image_path, original_image);
    
    std::cout << "Saved final image to: " << output_image_path << std::endl;
    
    return 0;
}