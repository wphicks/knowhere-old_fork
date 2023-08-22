#include "../src/matrix/specializations/detail/select_k_float_int64_t.cu"
#include "../src/matrix/specializations/detail/select_k_float_uint32_t.cu"
#include "../src/matrix/specializations/detail/select_k_half_int64_t.cu"
#include "../src/matrix/specializations/detail/select_k_half_uint32_t.cu"
#include "../src/neighbors/specializations/detail/compute_similarity_float_float_fast.cu"
#include "../src/neighbors/specializations/detail/compute_similarity_float_float_no_basediff.cu"
#include "../src/neighbors/specializations/detail/compute_similarity_float_float_no_smem_lut.cu"
#include "../src/neighbors/specializations/detail/compute_similarity_float_fp8s_fast.cu"
#include "../src/neighbors/specializations/detail/compute_similarity_float_fp8s_no_basediff.cu"
#include "../src/neighbors/specializations/detail/compute_similarity_float_fp8s_no_smem_lut.cu"
#include "../src/neighbors/specializations/detail/compute_similarity_float_fp8u_fast.cu"
#include "../src/neighbors/specializations/detail/compute_similarity_float_fp8u_no_basediff.cu"
#include "../src/neighbors/specializations/detail/compute_similarity_float_fp8u_no_smem_lut.cu"
#include "../src/neighbors/specializations/detail/compute_similarity_float_half_fast.cu"
#include "../src/neighbors/specializations/detail/compute_similarity_float_half_no_basediff.cu"
#include "../src/neighbors/specializations/detail/compute_similarity_float_half_no_smem_lut.cu"
#include "../src/neighbors/specializations/detail/compute_similarity_half_fp8s_fast.cu"
#include "../src/neighbors/specializations/detail/compute_similarity_half_fp8s_no_basediff.cu"
#include "../src/neighbors/specializations/detail/compute_similarity_half_fp8s_no_smem_lut.cu"
#include "../src/neighbors/specializations/detail/compute_similarity_half_fp8u_fast.cu"
#include "../src/neighbors/specializations/detail/compute_similarity_half_fp8u_no_basediff.cu"
#include "../src/neighbors/specializations/detail/compute_similarity_half_fp8u_no_smem_lut.cu"
#include "../src/neighbors/specializations/detail/compute_similarity_half_half_fast.cu"
#include "../src/neighbors/specializations/detail/compute_similarity_half_half_no_basediff.cu"
#include "../src/neighbors/specializations/detail/compute_similarity_half_half_no_smem_lut.cu"
#include "../src/neighbors/specializations/fused_l2_knn_int_float_false.cu"
#include "../src/neighbors/specializations/fused_l2_knn_int_float_true.cu"
#include "../src/neighbors/specializations/fused_l2_knn_long_float_false.cu"
#include "../src/neighbors/specializations/fused_l2_knn_long_float_true.cu"
#include "../src/neighbors/specializations/ivfflat_build_float_int64_t.cu"
#include "../src/neighbors/specializations/ivfflat_extend_float_int64_t.cu"
#include "../src/neighbors/specializations/ivfflat_search_float_int64_t.cu"
#include "../src/neighbors/specializations/ivfpq_build_float_int64_t.cu"
#include "../src/neighbors/specializations/ivfpq_extend_float_int64_t.cu"
#include "../src/neighbors/specializations/ivfpq_search_float_int64_t.cu"
