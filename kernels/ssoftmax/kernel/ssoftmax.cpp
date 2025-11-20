#include <riscv_vector.h>
#include <stdint.h>
#include <math.h>
#include <float.h>

extern "C" {

// --- MATH LIB INLINE START ---
inline vfloat32m1_t __exp_2xf32(vfloat32m1_t x, size_t gvl) {
    vfloat32m1_t exp_hi = __riscv_vfmv_v_f_f32m1(88.3762626647949, gvl);
    vfloat32m1_t exp_lo = __riscv_vfmv_v_f_f32m1(-88.3762626647949, gvl);
    vfloat32m1_t log2ef  = __riscv_vfmv_v_f_f32m1(1.44269504088896341, gvl);
    vfloat32m1_t inv_ln2 = __riscv_vfmv_v_f_f32m1(0.693359375, gvl);
    vfloat32m1_t inv_ln2_err = __riscv_vfmv_v_f_f32m1(-2.12194440e-4, gvl);
    vfloat32m1_t p0 = __riscv_vfmv_v_f_f32m1(1.9875691500E-4, gvl);
    vfloat32m1_t p1 = __riscv_vfmv_v_f_f32m1(1.3981999507E-3, gvl);
    vfloat32m1_t p2 = __riscv_vfmv_v_f_f32m1(8.3334519073E-3, gvl);
    vfloat32m1_t p3 = __riscv_vfmv_v_f_f32m1(4.1665795894E-2, gvl);
    vfloat32m1_t p4 = __riscv_vfmv_v_f_f32m1(1.6666665459E-1, gvl);
    vfloat32m1_t p5 = __riscv_vfmv_v_f_f32m1(5.0000001201E-1, gvl);
    vfloat32m1_t one = __riscv_vfmv_v_f_f32m1(1.0, gvl);
    vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0.0, gvl);
    vfloat32m1_t half = __riscv_vfmv_v_f_f32m1(0.5, gvl);

    x = __riscv_vfmin_vv_f32m1(x, exp_hi, gvl);
    x = __riscv_vfmax_vv_f32m1(x, exp_lo, gvl);

    vfloat32m1_t fx = __riscv_vfmacc_vv_f32m1(half, x, log2ef, gvl);
    vint32m1_t mm = __riscv_vfcvt_x_f_v_i32m1(fx, gvl);
    vfloat32m1_t tmp = __riscv_vfcvt_f_x_v_f32m1(mm, gvl);

    vbool32_t mask = __riscv_vmflt_vv_f32m1_b32(fx, tmp, gvl);
    vfloat32m1_t mask_add = __riscv_vmerge_vvm_f32m1(zero, one, mask, gvl);
    fx = __riscv_vfsub_vv_f32m1(tmp, mask_add, gvl);

    vfloat32m1_t z = __riscv_vfmul_vv_f32m1(fx, inv_ln2, gvl);
    x = __riscv_vfsub_vv_f32m1(x, z, gvl);
    z = __riscv_vfmul_vv_f32m1(fx, inv_ln2_err, gvl);
    x = __riscv_vfsub_vv_f32m1(x, z, gvl);
    z = __riscv_vfmul_vv_f32m1(x, x, gvl);

    vfloat32m1_t y = p0;
    y = __riscv_vfmadd_vv_f32m1(y, x, p1, gvl);
    y = __riscv_vfmadd_vv_f32m1(y, x, p2, gvl);
    y = __riscv_vfmadd_vv_f32m1(y, x, p3, gvl);
    y = __riscv_vfmadd_vv_f32m1(y, x, p4, gvl);
    y = __riscv_vfmadd_vv_f32m1(y, x, p5, gvl);
    y = __riscv_vfmadd_vv_f32m1(y, z, x, gvl);
    y = __riscv_vfadd_vv_f32m1(y, one, gvl);

    mm = __riscv_vfcvt_x_f_v_i32m1(fx, gvl);
    mm = __riscv_vadd_vv_i32m1(mm, __riscv_vmv_v_x_i32m1(0x7f, gvl), gvl);
    mm = __riscv_vsll_vv_i32m1(mm, __riscv_vmv_v_x_u32m1(23, gvl), gvl);
    
    vfloat32m1_t pow2n = __riscv_vreinterpret_v_i32m1_f32m1(mm);
    y = __riscv_vfmul_vv_f32m1(y, pow2n, gvl);
    return y;
}
// --- MATH LIB INLINE END ---

void softmax_vec(const float *i, float *o, uint64_t channels, uint64_t innerSize) {
  size_t avl = innerSize;
  size_t vl;
  float *_i = (float *)i;
  float *_o = (float *)o;
  float *__i = (float *)i;
  float *__o = (float *)o;

  for (vl = __riscv_vsetvl_e32m1(avl); avl > 0; avl -= vl) {
    vl = __riscv_vsetvl_e32m1(avl);
    vfloat32m1_t max_chunk_v = __riscv_vle32_v_f32m1(__i, vl);
    __i += innerSize;
    for (uint64_t ch = 1; ch < channels; ++ch) {
      vfloat32m1_t buf_chunk_v = __riscv_vle32_v_f32m1(__i, vl);
      __i += innerSize;
      max_chunk_v = __riscv_vfmax_vv_f32m1(max_chunk_v, buf_chunk_v, vl);
    }
    __i = _i;

    vfloat32m1_t den_chunk_v = __riscv_vfmv_v_f_f32m1(0, vl);
    for (uint64_t ch = 0; ch < channels; ++ch) {
      vfloat32m1_t buf_chunk_v = __riscv_vle32_v_f32m1(__i, vl);
      buf_chunk_v = __riscv_vfsub_vv_f32m1(buf_chunk_v, max_chunk_v, vl);
      buf_chunk_v = __exp_2xf32(buf_chunk_v, vl);
      __riscv_vse32_v_f32m1(__o, buf_chunk_v, vl);
      den_chunk_v = __riscv_vfadd_vv_f32m1(den_chunk_v, buf_chunk_v, vl);
      __i += innerSize; __o += innerSize;
    }
    __i = _i; __o = _o;

    for (uint64_t ch = 0; ch < channels; ++ch) {
      vfloat32m1_t num_chunk_v = __riscv_vle32_v_f32m1(__o, vl);
      vfloat32m1_t res_chunk_v = __riscv_vfdiv_vv_f32m1(num_chunk_v, den_chunk_v, vl);
      __riscv_vse32_v_f32m1(__o, res_chunk_v, vl);
      __o += innerSize;
    }
    _i += vl; _o += vl;
    __i = _i; __o = _o;
  }
}

void softmax_scalar(float* input, float* output, size_t size) {
    float max_val = -FLT_MAX;
    for (size_t i = 0; i < size; i++) if (input[i] > max_val) max_val = input[i];
    float sum = 0.0f;
    for (size_t i = 0; i < size; i++) {
        output[i] = expf(input[i] - max_val);
        sum += output[i];
    }
    for (size_t i = 0; i < size; i++) output[i] /= sum;
}

}