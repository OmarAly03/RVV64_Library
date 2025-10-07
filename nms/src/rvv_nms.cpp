#include <riscv_vector.h>
#include <cstddef>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstring>
#include "../include/defs.h"

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
    vfloat32m1_t b1_xy1 = __riscv_vle32_v_f32m1(box1, vl);
    vfloat32m1_t b1_xy2 = __riscv_vle32_v_f32m1(box1 + 2, vl);

    vfloat32m1_t b2_xy1 = __riscv_vle32_v_f32m1(box2, vl);
    vfloat32m1_t b2_xy2 = __riscv_vle32_v_f32m1(box2 + 2, vl);

    vfloat32m1_t inter_xy1 = __riscv_vfmax_vv_f32m1(b1_xy1, b2_xy1, vl);
    vfloat32m1_t inter_xy2 = __riscv_vfmin_vv_f32m1(b1_xy2, b2_xy2, vl);

    vfloat32m1_t inter_wh = __riscv_vfsub_vv_f32m1(inter_xy2, inter_xy1, vl);
    inter_wh = __riscv_vfmax_vf_f32m1(inter_wh, 0.0f, vl);

    float inter_w = __riscv_vfmv_f_s_f32m1_f32(__riscv_vslidedown_vx_f32m1(inter_wh, 1, vl));
    float inter_h = __riscv_vfmv_f_s_f32m1_f32(inter_wh);
    float inter_area = inter_w * inter_h;

    vfloat32m1_t b1_wh = __riscv_vfsub_vv_f32m1(b1_xy2, b1_xy1, vl);
    float area1 = __riscv_vfmv_f_s_f32m1_f32(b1_wh) * __riscv_vfmv_f_s_f32m1_f32(__riscv_vslidedown_vx_f32m1(b1_wh, 1, vl));

    vfloat32m1_t b2_wh = __riscv_vfsub_vv_f32m1(b2_xy2, b2_xy1, vl);
    float area2 = __riscv_vfmv_f_s_f32m1_f32(b2_wh) * __riscv_vfmv_f_s_f32m1_f32(__riscv_vslidedown_vx_f32m1(b2_wh, 1, vl));

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

            for (size_t i = 0; i < spatial_dimension; i += __riscv_vsetvlmax_e32m1()) {
                size_t vl = __riscv_vsetvl_e32m1(spatial_dimension - i);
                size_t score_idx = batch * num_classes * spatial_dimension + cls * spatial_dimension + i;

                vfloat32m1_t vscores = __riscv_vle32_v_f32m1(&scores[score_idx], vl);
                vfloat32m1_t vthreshold = __riscv_vfmv_v_f_f32m1(score_threshold, vl);
                vbool32_t mask = __riscv_vmfge_vv_f32m1_b32(vscores, vthreshold, vl);

                size_t count = __riscv_vcpop_m_b32(mask, vl);
                if (count > 0) {
                    vuint32m1_t all_indices = __riscv_vid_v_u32m1(vl);
                    vuint32m1_t selected_indices_vec = __riscv_vcompress_vm_u32m1(all_indices, mask, vl);
                    uint32_t* indices_arr = new uint32_t[count];
                    __riscv_vse32_v_u32m1(indices_arr, selected_indices_vec, count);
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

            for (size_t i = 0; i < spatial_dimension; i += __riscv_vsetvlmax_e32m2()) {
                size_t vl = __riscv_vsetvl_e32m2(spatial_dimension - i);
                size_t score_idx = batch * num_classes * spatial_dimension + cls * spatial_dimension + i;

                vfloat32m2_t vscores = __riscv_vle32_v_f32m2(&scores[score_idx], vl);
                vfloat32m2_t vthreshold = __riscv_vfmv_v_f_f32m2(score_threshold, vl);
                vbool16_t mask = __riscv_vmfge_vv_f32m2_b16(vscores, vthreshold, vl);

                size_t count = __riscv_vcpop_m_b16(mask, vl);
                if (count > 0) {
                    vuint32m2_t all_indices = __riscv_vid_v_u32m2(vl);
                    vuint32m2_t selected_indices_vec = __riscv_vcompress_vm_u32m2(all_indices, mask, vl);
                    uint32_t* indices_arr = new uint32_t[count];
                    __riscv_vse32_v_u32m2(indices_arr, selected_indices_vec, count);
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

            for (size_t i = 0; i < spatial_dimension; i += __riscv_vsetvlmax_e32m4()) {
                size_t vl = __riscv_vsetvl_e32m4(spatial_dimension - i);
                size_t score_idx = batch * num_classes * spatial_dimension + cls * spatial_dimension + i;

                vfloat32m4_t vscores = __riscv_vle32_v_f32m4(&scores[score_idx], vl);
                vfloat32m4_t vthreshold = __riscv_vfmv_v_f_f32m4(score_threshold, vl);
                vbool8_t mask = __riscv_vmfge_vv_f32m4_b8(vscores, vthreshold, vl);

                size_t count = __riscv_vcpop_m_b8(mask, vl);
                if (count > 0) {
                    vuint32m4_t all_indices = __riscv_vid_v_u32m4(vl);
                    vuint32m4_t selected_indices_vec = __riscv_vcompress_vm_u32m4(all_indices, mask, vl);
                    uint32_t* indices_arr = new uint32_t[count];
                    __riscv_vse32_v_u32m4(indices_arr, selected_indices_vec, count);
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

            for (size_t i = 0; i < spatial_dimension; i += __riscv_vsetvlmax_e32m8()) {
                size_t vl = __riscv_vsetvl_e32m8(spatial_dimension - i);
                size_t score_idx = batch * num_classes * spatial_dimension + cls * spatial_dimension + i;

                vfloat32m8_t vscores = __riscv_vle32_v_f32m8(&scores[score_idx], vl);
                vfloat32m8_t vthreshold = __riscv_vfmv_v_f_f32m8(score_threshold, vl);
                vbool4_t mask = __riscv_vmfge_vv_f32m8_b4(vscores, vthreshold, vl);

                size_t count = __riscv_vcpop_m_b4(mask, vl);
                if (count > 0) {
                    vuint32m8_t all_indices = __riscv_vid_v_u32m8(vl);
                    vuint32m8_t selected_indices_vec = __riscv_vcompress_vm_u32m8(all_indices, mask, vl);
                    uint32_t* indices_arr = new uint32_t[count];
                    __riscv_vse32_v_u32m8(indices_arr, selected_indices_vec, count);
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