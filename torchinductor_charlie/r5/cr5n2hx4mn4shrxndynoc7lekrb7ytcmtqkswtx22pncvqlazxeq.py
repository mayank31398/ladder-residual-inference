
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints=[65536, 2048],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*i1', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=7, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__safe_softmax_add_index_scalar_tensor_where_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 3, 'backend_hash': 'C766DF4C74330B401EEA30AF57196D45F0DB0787EA9002C9D886631511308A07', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False}
)
@triton.jit
def triton_red_fused__safe_softmax_add_index_scalar_tensor_where_4(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 65536
    rnumel = 1536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x0 = xindex % 1024
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    _tmp18 = tl.full([XBLOCK, RBLOCK], 0, tl.int1)
    _tmp21 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_out_ptr0 + (r2 + (1536*x3)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.full([XBLOCK, RBLOCK], 1536, tl.int32)
        tmp3 = tmp1 + tmp2
        tmp4 = tmp1 < 0
        tmp5 = tl.where(tmp4, tmp3, tmp1)
        tl.device_assert((0 <= tmp5) & (tmp5 < 1536), "index out of bounds: 0 <= tmp5 < 1536")
        tmp7 = tl.load(in_ptr1 + (r2 + (1536*tmp5)), rmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
        tmp8 = 0.0
        tmp9 = float("-inf")
        tmp10 = tl.where(tmp7, tmp8, tmp9)
        tmp11 = tmp10.to(tl.float32)
        tmp12 = tmp0 + tmp11
        tmp13 = tmp12 == tmp9
        tmp14 = tmp13 == 0
        tmp15 = tmp14.to(tl.int64)
        tmp16 = (tmp15 != 0)
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
        tmp19 = _tmp18 | tmp17
        _tmp18 = tl.where(rmask, tmp19, _tmp18)
        tmp20 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp22 = triton_helpers.maximum(_tmp21, tmp20)
        _tmp21 = tl.where(rmask, tmp22, _tmp21)
    tmp18 = triton_helpers.any(_tmp18.to(tl.int8), 1)[:, None].to(tl.int1)
    tmp21 = triton_helpers.max2(_tmp21, 1)[:, None]
    _tmp38 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp23 = tl.load(in_out_ptr0 + (r2 + (1536*x3)), rmask, eviction_policy='evict_last', other=0.0)
        tmp24 = tl.full([XBLOCK, RBLOCK], 1536, tl.int32)
        tmp25 = tmp1 + tmp24
        tmp26 = tmp1 < 0
        tmp27 = tl.where(tmp26, tmp25, tmp1)
        tl.device_assert((0 <= tmp27) & (tmp27 < 1536), "index out of bounds: 0 <= tmp27 < 1536")
        tmp29 = tl.load(in_ptr1 + (r2 + (1536*tmp27)), rmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
        tmp30 = 0.0
        tmp31 = float("-inf")
        tmp32 = tl.where(tmp29, tmp30, tmp31)
        tmp33 = tmp32.to(tl.float32)
        tmp34 = tmp23 + tmp33
        tmp35 = tmp34 - tmp21
        tmp36 = tl_math.exp(tmp35)
        tmp37 = tl.broadcast_to(tmp36, [XBLOCK, RBLOCK])
        tmp39 = _tmp38 + tmp37
        _tmp38 = tl.where(rmask, tmp39, _tmp38)
    tmp38 = tl.sum(_tmp38, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp41 = tl.load(in_out_ptr0 + (r2 + (1536*x3)), rmask, eviction_policy='evict_first', other=0.0)
        tmp40 = tmp18 == 0
        tmp42 = tl.full([XBLOCK, RBLOCK], 1536, tl.int32)
        tmp43 = tmp1 + tmp42
        tmp44 = tmp1 < 0
        tmp45 = tl.where(tmp44, tmp43, tmp1)
        tl.device_assert((0 <= tmp45) & (tmp45 < 1536), "index out of bounds: 0 <= tmp45 < 1536")
        tmp47 = tl.load(in_ptr1 + (r2 + (1536*tmp45)), rmask, eviction_policy='evict_first', other=0.0).to(tl.int1)
        tmp48 = 0.0
        tmp49 = float("-inf")
        tmp50 = tl.where(tmp47, tmp48, tmp49)
        tmp51 = tmp50.to(tl.float32)
        tmp52 = tmp41 + tmp51
        tmp53 = tmp52 - tmp21
        tmp54 = tl_math.exp(tmp53)
        tmp55 = tmp54 / tmp38
        tmp56 = tl.where(tmp40, tmp48, tmp55)
        tl.store(in_out_ptr0 + (r2 + (1536*x3)), tmp56, rmask)
