
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints=[4, 131072],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr2': '*i32', 'load_seed_offset': 'i32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=2, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax__to_copy_argmax_div_exponential_lt_scalar_tensor_where_21', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 1, 'backend_hash': 'C766DF4C74330B401EEA30AF57196D45F0DB0787EA9002C9D886631511308A07', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False}
)
@triton.jit
def triton_red_fused__softmax__to_copy_argmax_div_exponential_lt_scalar_tensor_where_21(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, load_seed_offset, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4
    rnumel = 128256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp4 = tl.load(in_ptr2 + (199 + (200*x0)), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp9 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    _tmp25 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    _tmp25_index = tl.full([XBLOCK, RBLOCK], 9223372036854775807, tl.int64)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp3 = tl.load(in_ptr1 + (r1 + (128256*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp0 = tl.load(in_ptr0 + load_seed_offset)
        tmp1 = r1 + (128256*x0)
        tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
        tmp5 = tmp3 < tmp4
        tmp6 = float("-inf")
        tmp7 = tl.where(tmp5, tmp6, tmp3)
        tmp8 = tmp7.to(tl.float32)
        tmp10 = tmp8 - tmp9
        tmp11 = tl_math.exp(tmp10)
        tmp13 = tmp11 / tmp12
        tmp14 = tmp13.to(tl.float32)
        tmp15 = 0.9999999403953552
        tmp16 = tmp2 >= tmp15
        tmp17 = tl_math.log(tmp2)
        tmp18 = -5.960464477539063e-08
        tmp19 = tl.where(tmp16, tmp18, tmp17)
        tmp20 = -1.0
        tmp21 = tmp19 * tmp20
        tmp22 = tmp21.to(tl.float32)
        tmp23 = tmp14 / tmp22
        tmp24 = tl.broadcast_to(tmp23, [XBLOCK, RBLOCK])
        _tmp25_next, _tmp25_index_next = triton_helpers.maximum_with_index(
            _tmp25, _tmp25_index, tmp24, rindex
        )
        _tmp25 = tl.where(rmask & xmask, _tmp25_next, _tmp25)
        _tmp25_index = tl.where(rmask & xmask, _tmp25_index_next, _tmp25_index)
    tmp25_val, tmp25_idx = triton_helpers.max_with_index(_tmp25, _tmp25_index, 1)
    tmp25 = tmp25_idx[:, None]
    tmp26 = tmp25.to(tl.int32)
    tl.store(out_ptr2 + (x0), tmp26, xmask)
