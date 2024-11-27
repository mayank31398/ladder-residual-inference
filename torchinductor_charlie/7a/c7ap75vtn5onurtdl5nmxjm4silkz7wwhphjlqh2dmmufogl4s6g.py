
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints=[1, 131072],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr2': '*i32', 'load_seed_offset': 'i32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=2, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 8), equal_to_1=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax__to_copy_argmax_div_exponential_lt_scalar_tensor_where_21', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 1, 'backend_hash': 'C766DF4C74330B401EEA30AF57196D45F0DB0787EA9002C9D886631511308A07', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False}
)
@triton.jit
def triton_red_fused__softmax__to_copy_argmax_div_exponential_lt_scalar_tensor_where_21(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, load_seed_offset, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 128256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    tmp4 = tl.load(in_ptr2 + (199)).to(tl.float32)
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp10 = tl.load(in_ptr3 + (0))
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
    tmp14 = tl.load(in_ptr4 + (0))
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    _tmp28 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    _tmp28_index = tl.full([XBLOCK, RBLOCK], 9223372036854775807, tl.int64)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp3 = tl.load(in_ptr1 + (r0), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp0 = tl.load(in_ptr0 + load_seed_offset)
        tmp1 = r0
        tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
        tmp6 = tmp3 < tmp5
        tmp7 = float("-inf")
        tmp8 = tl.where(tmp6, tmp7, tmp3)
        tmp9 = tmp8.to(tl.float32)
        tmp12 = tmp9 - tmp11
        tmp13 = tl_math.exp(tmp12)
        tmp16 = tmp13 / tmp15
        tmp17 = tmp16.to(tl.float32)
        tmp18 = 0.9999999403953552
        tmp19 = tmp2 >= tmp18
        tmp20 = tl_math.log(tmp2)
        tmp21 = -5.960464477539063e-08
        tmp22 = tl.where(tmp19, tmp21, tmp20)
        tmp23 = -1.0
        tmp24 = tmp22 * tmp23
        tmp25 = tmp24.to(tl.float32)
        tmp26 = tmp17 / tmp25
        tmp27 = tl.broadcast_to(tmp26, [XBLOCK, RBLOCK])
        _tmp28_next, _tmp28_index_next = triton_helpers.maximum_with_index(
            _tmp28, _tmp28_index, tmp27, rindex
        )
        _tmp28 = tl.where(rmask, _tmp28_next, _tmp28)
        _tmp28_index = tl.where(rmask, _tmp28_index_next, _tmp28_index)
    tmp28_val, tmp28_idx = triton_helpers.max_with_index(_tmp28, _tmp28_index, 1)
    tmp28 = tmp28_idx[:, None]
    tmp29 = tmp28.to(tl.int32)
    tl.store(out_ptr2 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp29, None)
