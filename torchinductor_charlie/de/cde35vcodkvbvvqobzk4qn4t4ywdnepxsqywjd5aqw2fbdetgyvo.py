
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[8388608], 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*i64', 'in_ptr2': '*fp16', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=1, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_clone_mul_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'C766DF4C74330B401EEA30AF57196D45F0DB0787EA9002C9D886631511308A07', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_clone_mul_2(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x5 = xindex
    x0 = xindex % 128
    x1 = (xindex // 128) % 8
    x6 = (xindex // 1024)
    x2 = (xindex // 1024) % 1024
    x3 = (xindex // 1048576)
    tmp0 = x5 % 2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((2*(x0 // 2)) + (128*x1) + (1280*x6)), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tl.load(in_ptr1 + (x2), tmp4, eviction_policy='evict_last', other=0.0)
    tmp8 = tl.full([XBLOCK], 131072, tl.int32)
    tmp9 = tmp7 + tmp8
    tmp10 = tmp7 < 0
    tmp11 = tl.where(tmp10, tmp9, tmp7)
    tl.device_assert(((0 <= tl.broadcast_to(tmp11, [XBLOCK])) & (tl.broadcast_to(tmp11, [XBLOCK]) < 131072)) | ~(tmp4), "index out of bounds: 0 <= tl.broadcast_to(tmp11, [XBLOCK]) < 131072")
    tmp13 = tl.load(in_ptr2 + ((2*(x0 // 2)) + (128*tmp11)), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp6 * tmp14
    tmp16 = tl.load(in_ptr0 + (1 + (2*(x0 // 2)) + (128*x1) + (1280*x6)), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tl.load(in_ptr2 + (1 + (2*(x0 // 2)) + (128*tmp11)), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tmp17 * tmp19
    tmp21 = tmp15 - tmp20
    tmp22 = tl.full(tmp21.shape, 0.0, tmp21.dtype)
    tmp23 = tl.where(tmp4, tmp21, tmp22)
    tmp24 = tmp0 >= tmp3
    tmp25 = tl.full([1], 2, tl.int64)
    tmp26 = tmp0 < tmp25
    tmp27 = tl.load(in_ptr0 + (1 + (2*(x0 // 2)) + (128*x1) + (1280*x6)), tmp24, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp28 = tmp27.to(tl.float32)
    tmp29 = tl.load(in_ptr1 + (x2), tmp24, eviction_policy='evict_last', other=0.0)
    tmp30 = tmp29 + tmp8
    tmp31 = tmp29 < 0
    tmp32 = tl.where(tmp31, tmp30, tmp29)
    tl.device_assert(((0 <= tl.broadcast_to(tmp32, [XBLOCK])) & (tl.broadcast_to(tmp32, [XBLOCK]) < 131072)) | ~(tmp24), "index out of bounds: 0 <= tl.broadcast_to(tmp32, [XBLOCK]) < 131072")
    tmp34 = tl.load(in_ptr2 + ((2*(x0 // 2)) + (128*tmp32)), tmp24, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp35 = tmp34.to(tl.float32)
    tmp36 = tmp28 * tmp35
    tmp37 = tl.load(in_ptr0 + ((2*(x0 // 2)) + (128*x1) + (1280*x6)), tmp24, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp38 = tmp37.to(tl.float32)
    tmp39 = tl.load(in_ptr2 + (1 + (2*(x0 // 2)) + (128*tmp32)), tmp24, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp40 = tmp39.to(tl.float32)
    tmp41 = tmp38 * tmp40
    tmp42 = tmp36 + tmp41
    tmp43 = tl.full(tmp42.shape, 0.0, tmp42.dtype)
    tmp44 = tl.where(tmp24, tmp42, tmp43)
    tmp45 = tl.where(tmp4, tmp23, tmp44)
    tmp46 = tmp45.to(tl.float32)
    tmp47 = tmp46.to(tl.float32)
    tmp48 = 0.29730177875068026
    tmp49 = tmp47 * tmp48
    tl.store(out_ptr0 + (x0 + (128*x2) + (131072*x1) + (1048576*x3)), tmp49, None)
