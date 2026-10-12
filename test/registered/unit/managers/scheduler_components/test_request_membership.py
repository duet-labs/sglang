"""Unit tests for request_queries.assert_single_membership."""

import unittest
from types import SimpleNamespace as NS

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.scheduler_components import request_queries
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _scheduler(**overrides):
    fields = dict(
        waiting_queue=[],
        running_batch=NS(reqs=[]),
        last_batch=None,
        chunked_req=None,
        grammar_manager=NS(grammar_queue=[]),
        hisparse_coordinator=None,
        disaggregation_mode=DisaggregationMode.NULL,
        dllm_config=None,
        dllm_manager=NS(waiting_queue=[]),
        disagg_prefill_bootstrap_queue=NS(queue=[]),
        disagg_prefill_inflight_queue=[],
        disagg_decode_prealloc_queue=NS(
            queue=[], pending_reqs=[], retracted_queue=[], held_rebootstrap_reqs=[]
        ),
        disagg_decode_transfer_queue=NS(queue=[]),
    )
    fields.update(overrides)
    return NS(**fields)


def _prealloc(queue=(), pending=()):
    return NS(
        queue=list(queue),
        pending_reqs=list(pending),
        retracted_queue=[],
        held_rebootstrap_reqs=[],
    )


class TestAssertSingleMembership(CustomTestCase):
    def setUp(self):
        self.a = NS(rid="a")
        self.b = NS(rid="b")

    def test_legitimate_layouts_pass(self):
        a, b = self.a, self.b
        running = NS(reqs=[a])
        layouts = {
            "running batch is last batch": _scheduler(
                running_batch=running, last_batch=running
            ),
            "chunked request also in last batch": _scheduler(
                chunked_req=b, last_batch=NS(reqs=[b])
            ),
            "pending-DP-rank is a subset of prealloc": _scheduler(
                disaggregation_mode=DisaggregationMode.DECODE,
                disagg_decode_prealloc_queue=_prealloc(
                    queue=[NS(req=b)], pending=[NS(req=b)]
                ),
            ),
            "prefill queues unset after a failed role switch": _scheduler(
                disaggregation_mode=DisaggregationMode.PREFILL,
                disagg_prefill_bootstrap_queue=None,
                disagg_prefill_inflight_queue=None,
                waiting_queue=[b],
            ),
        }
        for name, scheduler in layouts.items():
            with self.subTest(name):
                request_queries.assert_single_membership(scheduler)

    def test_cross_container_duplicates_raise(self):
        a, b = self.a, self.b
        running = NS(reqs=[a])
        layouts = {
            "waiting and running": _scheduler(waiting_queue=[a], running_batch=running),
            "chunked and running": _scheduler(chunked_req=a, running_batch=running),
            "hisparse staging and running": _scheduler(
                running_batch=running,
                hisparse_coordinator=NS(ack_staging_queue=[NS(req=a)]),
            ),
            "prealloc and transfer": _scheduler(
                disaggregation_mode=DisaggregationMode.DECODE,
                disagg_decode_prealloc_queue=_prealloc(queue=[NS(req=b)]),
                disagg_decode_transfer_queue=NS(queue=[NS(req=b)]),
            ),
            "prefill bootstrap and waiting": _scheduler(
                disaggregation_mode=DisaggregationMode.PREFILL,
                disagg_prefill_bootstrap_queue=NS(queue=[b]),
                waiting_queue=[b],
            ),
            "dllm waiting and waiting": _scheduler(
                dllm_config=object(),
                dllm_manager=NS(waiting_queue=[b]),
                waiting_queue=[b],
            ),
        }
        for name, scheduler in layouts.items():
            with self.subTest(name), self.assertRaisesRegex(AssertionError, "both"):
                request_queries.assert_single_membership(scheduler)


if __name__ == "__main__":
    unittest.main()
