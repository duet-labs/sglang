"""Which requests the scheduler holds, and whether any are left.

Each query reads its holders only through the accessors below, so where a kind
of request lives is stated once. Functions take the live ``Scheduler`` and only
read it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Deque, List, Optional, Set

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.runtime_context import get_memory, get_parallel

if TYPE_CHECKING:
    from sglang.srt.disaggregation.decode import DecodeRequest
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.managers.scheduler import Scheduler


# ----------------------------------------------------------------- holders


def get_waiting_queue(scheduler: Scheduler) -> List[Req]:
    return scheduler.waiting_queue


def get_running_batch(scheduler: Scheduler) -> ScheduleBatch:
    return scheduler.running_batch


def get_last_batch(scheduler: Scheduler) -> Optional[ScheduleBatch]:
    return scheduler.last_batch


def get_chunked_req(scheduler: Scheduler) -> Optional[Req]:
    return scheduler.chunked_req


def get_result_queue(scheduler: Scheduler) -> Deque:
    return scheduler.result_queue


def get_pp_running_batches(scheduler: Scheduler) -> List[ScheduleBatch]:
    # Set up by the pipeline-parallel event loops only.
    return scheduler.running_mbs


def get_pp_batches(scheduler: Scheduler) -> List[Optional[ScheduleBatch]]:
    return scheduler.mbs


def get_grammar_queue(scheduler: Scheduler) -> List[Req]:
    return scheduler.grammar_manager.grammar_queue


def get_prefill_bootstrap_queue(scheduler: Scheduler) -> List[Req]:
    return scheduler.disagg_prefill_bootstrap_queue.queue


def get_prefill_inflight_queue(scheduler: Scheduler) -> List[Req]:
    return scheduler.disagg_prefill_inflight_queue


def get_decode_prealloc_queue(scheduler: Scheduler) -> List[DecodeRequest]:
    return scheduler.disagg_decode_prealloc_queue.queue


def get_decode_pending_dp_rank_queue(scheduler: Scheduler) -> List[DecodeRequest]:
    # A subset of the prealloc queue: the same DecodeRequest objects.
    return scheduler.disagg_decode_prealloc_queue.pending_reqs


def get_decode_retracted_queue(scheduler: Scheduler) -> List[Req]:
    return scheduler.disagg_decode_prealloc_queue.retracted_queue


def get_decode_held_rebootstrap_reqs(scheduler: Scheduler) -> List[Req]:
    return scheduler.disagg_decode_prealloc_queue.held_rebootstrap_reqs


def get_decode_transfer_queue(scheduler: Scheduler) -> List[DecodeRequest]:
    return scheduler.disagg_decode_transfer_queue.queue


def get_dllm_waiting_queue(scheduler: Scheduler) -> List[Req]:
    return scheduler.dllm_manager.waiting_queue


def get_hisparse_staging_reqs(scheduler: Scheduler) -> List[Req]:
    if scheduler.hisparse_coordinator is None:
        return []
    return [act.req for act in scheduler.hisparse_coordinator.ack_staging_queue]


# ----------------------------------------------------------------- idle


def is_idle(scheduler: Scheduler) -> bool:
    return _is_idle(scheduler, for_health_check=False, ignore_waiting=False)


def is_idle_for_health_check(scheduler: Scheduler) -> bool:
    return _is_idle(scheduler, for_health_check=True, ignore_waiting=False)


def is_idle_ignoring_waiting(scheduler: Scheduler) -> bool:
    return _is_idle(scheduler, for_health_check=False, ignore_waiting=True)


def is_idle_for_flush(scheduler: Scheduler) -> bool:
    # While paused, waiting requests that hold no KV do not block a flush.
    if scheduler._engine_paused:
        return is_idle_ignoring_waiting(scheduler)
    return is_idle(scheduler)


def _is_idle(
    scheduler: Scheduler, for_health_check: bool, ignore_waiting: bool
) -> bool:
    # Health check piggybacks on running requests in process_output.
    # Only running_batch + waiting_queue guarantee active GPU processing;
    # disagg queues (bootstrap/prealloc/transfer) may have items without
    # any request actually running on GPU -- e.g. stuck handshake, full
    # KV cache, or stalled transfer -- so they can't carry health info.
    # Batch running status
    idle = (
        get_running_batch(scheduler).is_empty()
        and get_chunked_req(scheduler) is None
        and not scheduler.dllm_manager.any_staging_reqs()
        and (get_last_batch(scheduler) is None or get_last_batch(scheduler).is_empty())
        and (not scheduler.enable_overlap or len(get_result_queue(scheduler)) == 0)
        and _pp_microbatches_drained(scheduler)
    )

    # Waiting queues: waiting + bootstrapping + preallocation + kv transfer (decode)
    if not ignore_waiting:
        idle &= len(get_waiting_queue(scheduler)) == 0
    else:
        # A PD-decode request waiting with its transferred KV still owns
        # its request row; a flush would free it.
        idle &= not any(r.kv.holds_kv for r in get_waiting_queue(scheduler))

    if (
        for_health_check
        and not scheduler._engine_paused
        and scheduler.disaggregation_mode == DisaggregationMode.DECODE
        and scheduler.disagg_decode_prealloc_queue is not None
    ):
        idle &= len(get_decode_retracted_queue(scheduler)) == 0

    if not for_health_check:
        # Grammar queue and prefill inflight queue may not produce batch
        # results instantly, but they still indicate the server is not idle.
        idle &= len(get_grammar_queue(scheduler)) == 0
        if scheduler.disaggregation_mode == DisaggregationMode.PREFILL:
            idle &= len(get_prefill_inflight_queue(scheduler)) == 0
            idle &= len(get_prefill_bootstrap_queue(scheduler)) == 0

        if scheduler.disaggregation_mode == DisaggregationMode.DECODE:
            idle &= len(get_decode_prealloc_queue(scheduler)) == 0
            idle &= len(get_decode_retracted_queue(scheduler)) == 0
            idle &= len(get_decode_transfer_queue(scheduler)) == 0
            if scheduler.decode_offload_manager is not None:
                idle &= len(scheduler.decode_offload_manager.ongoing_offload) == 0

        # HiSparse: staging requests transitioning prefill -> decode
        if scheduler.enable_hisparse:
            idle &= not scheduler.hisparse_coordinator.has_ongoing_staging()

        # HiCache: in-flight async ops (GPU<->Host<->L3) must drain before
        # destructive operations like attach/detach/flush_cache.
        if scheduler.enable_hierarchical_cache:
            tc = scheduler.tree_cache
            idle &= len(tc.ongoing_write_through) == 0
            idle &= len(tc.ongoing_load_back) == 0
            if tc.enable_storage:
                idle &= len(tc.ongoing_prefetch) == 0
                idle &= len(tc.ongoing_backup) == 0
                if get_memory().hicache_host_memory_mode == "buffer_only":
                    # Queued writes, staged prefetches, and in-flight
                    # storage writes still hold host staging
                    # (buffer-mode unified tree only).
                    idle &= tc.buffer_pipeline.is_idle()
        elif scheduler.enable_lmcache:
            idle &= not scheduler.tree_cache.has_pending_cache_operations()

    return idle


def _pp_microbatches_drained(scheduler: Scheduler) -> bool:
    if get_parallel().pp_size == 1:
        return True
    return all(x.is_empty() for x in get_pp_running_batches(scheduler)) and all(
        mb is None or mb.is_empty() for mb in get_pp_batches(scheduler)
    )


# ----------------------------------------------------------------- request sets


def inflight_batches(scheduler: Scheduler) -> list[ScheduleBatch | None]:
    if get_parallel().pp_size == 1:
        batches = [get_running_batch(scheduler), get_last_batch(scheduler)]
    else:
        batches = [*get_pp_running_batches(scheduler), *get_pp_batches(scheduler)]
    if scheduler.enable_continuous_input_polling and get_result_queue(scheduler):
        # Polling can run intake before the newest queued batch becomes last_batch.
        batches.extend(batch for batch, _ in get_result_queue(scheduler))
    return batches


def inflight_reqs(scheduler: Scheduler) -> Set[Req]:
    return {
        req
        for batch in inflight_batches(scheduler)
        if batch is not None
        for req in batch.reqs
    }


def weight_version_reqs(scheduler: Scheduler) -> Set[Req]:
    chunked_req = get_chunked_req(scheduler)
    live_reqs = {
        *inflight_reqs(scheduler),
        *get_waiting_queue(scheduler),
        *([chunked_req] if chunked_req is not None else []),
    }
    live_reqs.update(get_hisparse_staging_reqs(scheduler))
    return live_reqs


def retract_on_pause_reqs(scheduler: Scheduler) -> List[Req]:
    """The requests a retract-mode pause retracts, in retraction order. The
    caller drains pending results first and removes these from their holders."""
    running_batch = get_running_batch(scheduler)
    last_batch = get_last_batch(scheduler)
    chunked_req = get_chunked_req(scheduler)
    retract_reqs = [r for r in running_batch.reqs if not r.finished()]
    if (
        last_batch is not None
        and last_batch.forward_mode.is_extend()
        # A running batch that DP attention converted to extend is
        # last_batch itself; its requests are already listed above.
        and last_batch is not running_batch
        # Skip merge for disagg prefill: completed prefill requests are
        # already in disagg_prefill_inflight_queue. Merging them into
        # running_batch leaks them, since the prefill event loop never
        # calls update_running_batch to clean them up.
        and scheduler.disaggregation_mode != DisaggregationMode.PREFILL
    ):
        retract_reqs += [r for r in last_batch.reqs if not r.finished()]

    if (
        chunked_req is not None
        and not chunked_req.finished()
        and chunked_req not in retract_reqs
        and scheduler.disaggregation_mode != DisaggregationMode.PREFILL
    ):
        retract_reqs.append(chunked_req)

    if scheduler.disaggregation_mode == DisaggregationMode.DECODE:
        # Transferred requests wait for a slot holding their row and KV;
        # retract them too, so their KV is rebuilt under the new weights.
        retract_reqs += [r for r in get_waiting_queue(scheduler) if r.kv.holds_kv]
    return retract_reqs


def owning_row_reqs(scheduler: Scheduler) -> List[Req]:
    """The requests that own a request-pool row and its KV."""
    last_batch = get_last_batch(scheduler)
    running_batch = get_running_batch(scheduler)
    reqs = [] if last_batch is None else list(last_batch.reqs)
    # After decode running_batch IS last_batch. Compare by identity:
    # ScheduleBatch's dataclass __eq__ compares tensors and raises.
    if running_batch is not None and running_batch is not last_batch:
        reqs += running_batch.reqs
    # The chunked-prefill request parked between chunks is in neither batch;
    # it still owns its KV row.
    chunked_req = get_chunked_req(scheduler)
    if chunked_req is not None:
        reqs.append(chunked_req)

    # A request running on a session's record does not own that row.
    session_kvs = {id(kv) for kv in scheduler.tree_cache.session_records().values()}
    owners, seen = [], set()
    for req in reqs:
        if id(req) in seen:
            continue
        seen.add(id(req))
        if req.kv.holds_kv and id(req.kv) not in session_kvs:
            owners.append(req)
    return owners


def sampling_mask_reqs(scheduler: Scheduler) -> List[Req]:
    """The queued or running requests, including PP micro-batches and PD decode
    queues, that a sampling-mask setting change must account for."""
    batches = [get_running_batch(scheduler), get_last_batch(scheduler)]
    # Read without the PP accessors: a non-PP scheduler has no micro-batches.
    batches.extend(getattr(scheduler, "running_mbs", ()))
    batches.extend(getattr(scheduler, "mbs", ()))
    batches.extend(getattr(scheduler, "last_mbs", ()))
    queued = [
        *(req for batch in batches if batch is not None for req in batch.reqs),
        *get_waiting_queue(scheduler),
        *get_grammar_queue(scheduler),
    ]
    chunked_req = get_chunked_req(scheduler)
    if chunked_req is not None:
        queued.append(chunked_req)
    if scheduler.disaggregation_mode == DisaggregationMode.DECODE:
        queued.extend(
            decode_req.req
            for decode_req in (
                *get_decode_prealloc_queue(scheduler),
                *get_decode_pending_dp_rank_queue(scheduler),
                *get_decode_transfer_queue(scheduler),
            )
        )
        queued.extend(get_decode_retracted_queue(scheduler))
        queued.extend(get_decode_held_rebootstrap_reqs(scheduler))
    return queued


# ----------------------------------------------------------------- membership


def assert_single_membership(scheduler: Scheduler) -> None:
    """No request is in two of the containers that hold it between iterations.

    A request may be in none of them: a just-prefilled one lives only in
    last_batch until the next merge. last_batch, result_queue and the PP
    micro-batch lists index requests already counted here, and the PD
    pending-DP-rank and dLLM staging queues are subsets of the prealloc and
    dLLM waiting queues, so none of those is a container.
    """
    chunked_req = get_chunked_req(scheduler)
    containers = {
        "waiting_queue": get_waiting_queue(scheduler),
        "running_batch": get_running_batch(scheduler).reqs,
        "chunked_req": [chunked_req] if chunked_req is not None else [],
        "grammar_queue": get_grammar_queue(scheduler),
        "hisparse_staging": get_hisparse_staging_reqs(scheduler),
    }
    if scheduler.disaggregation_mode == DisaggregationMode.PREFILL:
        containers["prefill_bootstrap"] = get_prefill_bootstrap_queue(scheduler)
        containers["prefill_inflight"] = get_prefill_inflight_queue(scheduler)
    if (
        scheduler.disaggregation_mode == DisaggregationMode.DECODE
        and scheduler.disagg_decode_prealloc_queue is not None
    ):
        containers["decode_prealloc"] = [
            decode_req.req for decode_req in get_decode_prealloc_queue(scheduler)
        ]
        containers["decode_retracted"] = get_decode_retracted_queue(scheduler)
        containers["decode_held_rebootstrap"] = get_decode_held_rebootstrap_reqs(
            scheduler
        )
        containers["decode_transfer"] = [
            decode_req.req for decode_req in get_decode_transfer_queue(scheduler)
        ]
    if scheduler.dllm_config is not None:
        containers["dllm_waiting"] = get_dllm_waiting_queue(scheduler)

    owner = {}
    for name, reqs in containers.items():
        for req in reqs:
            first = owner.setdefault(id(req), name)
            assert first == name, f"Request {req.rid} is in both {first} and {name}"
