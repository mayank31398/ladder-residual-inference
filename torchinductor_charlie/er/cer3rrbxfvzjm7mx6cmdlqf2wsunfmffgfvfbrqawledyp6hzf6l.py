
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'out_ptr0': '*fp16', 'out_ptr1': '*fp16', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_put_1', 'mutated_arg_names': ['out_ptr0', 'out_ptr1'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'C766DF4C74330B401EEA30AF57196D45F0DB0787EA9002C9D886631511308A07', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_index_put_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = (xindex // 128) % 1024
    x4 = xindex
    x0 = xindex % 128
    x3 = (xindex // 128)
    x2 = (xindex // 131072)
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp53 = tl.load(in_ptr1 + (1152 + x0 + (1280*x3)), None).to(tl.float32)
    tmp1 = tl.full([XBLOCK], 1536, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert((0 <= tmp4) & (tmp4 < 1536), "index out of bounds: 0 <= tmp4 < 1536")
    tmp6 = x4 % 2
    tmp7 = tl.full([1], 0, tl.int64)
    tmp8 = tmp6 >= tmp7
    tmp9 = tl.full([1], 1, tl.int64)
    tmp10 = tmp6 < tmp9
    tmp11 = tl.load(in_ptr1 + (1024 + (2*(x0 // 2)) + (1280*x3)), tmp10, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tl.load(in_ptr0 + (x1), tmp10, eviction_policy='evict_last', other=0.0)
    tmp14 = tl.full([XBLOCK], 131072, tl.int32)
    tmp15 = tmp13 + tmp14
    tmp16 = tmp13 < 0
    tmp17 = tl.where(tmp16, tmp15, tmp13)
    tl.device_assert(((0 <= tl.broadcast_to(tmp17, [XBLOCK])) & (tl.broadcast_to(tmp17, [XBLOCK]) < 131072)) | ~(tmp10), "index out of bounds: 0 <= tl.broadcast_to(tmp17, [XBLOCK]) < 131072")
    tmp19 = tl.load(in_ptr2 + ((2*(x0 // 2)) + (128*tmp17)), tmp10, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tmp12 * tmp20
    tmp22 = tl.load(in_ptr1 + (1025 + (2*(x0 // 2)) + (1280*x3)), tmp10, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tl.load(in_ptr2 + (1 + (2*(x0 // 2)) + (128*tmp17)), tmp10, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp23 * tmp25
    tmp27 = tmp21 - tmp26
    tmp28 = tl.full(tmp27.shape, 0.0, tmp27.dtype)
    tmp29 = tl.where(tmp10, tmp27, tmp28)
    tmp30 = tmp6 >= tmp9
    tmp31 = tl.full([1], 2, tl.int64)
    tmp32 = tmp6 < tmp31
    tmp33 = tl.load(in_ptr1 + (1025 + (2*(x0 // 2)) + (1280*x3)), tmp30, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp34 = tmp33.to(tl.float32)
    tmp35 = tl.load(in_ptr0 + (x1), tmp30, eviction_policy='evict_last', other=0.0)
    tmp36 = tmp35 + tmp14
    tmp37 = tmp35 < 0
    tmp38 = tl.where(tmp37, tmp36, tmp35)
    tl.device_assert(((0 <= tl.broadcast_to(tmp38, [XBLOCK])) & (tl.broadcast_to(tmp38, [XBLOCK]) < 131072)) | ~(tmp30), "index out of bounds: 0 <= tl.broadcast_to(tmp38, [XBLOCK]) < 131072")
    tmp40 = tl.load(in_ptr2 + ((2*(x0 // 2)) + (128*tmp38)), tmp30, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp41 = tmp40.to(tl.float32)
    tmp42 = tmp34 * tmp41
    tmp43 = tl.load(in_ptr1 + (1024 + (2*(x0 // 2)) + (1280*x3)), tmp30, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp44 = tmp43.to(tl.float32)
    tmp45 = tl.load(in_ptr2 + (1 + (2*(x0 // 2)) + (128*tmp38)), tmp30, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp46 = tmp45.to(tl.float32)
    tmp47 = tmp44 * tmp46
    tmp48 = tmp42 + tmp47
    tmp49 = tl.full(tmp48.shape, 0.0, tmp48.dtype)
    tmp50 = tl.where(tmp30, tmp48, tmp49)
    tmp51 = tl.where(tmp10, tmp29, tmp50)
    tmp52 = tmp51.to(tl.float32)
    tl.store(out_ptr0 + (x0 + (128*tmp4) + (196608*x2)), tmp52, None)
    tl.store(out_ptr1 + (x0 + (128*tmp4) + (196608*x2)), tmp53, None)
