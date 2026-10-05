"""Regress the C1 terminal lookahead without importing GPU scheduler modules.

The real loop/helper AST runs against a budget-blind planner and asynchronous
result boundary. Removing the drain submits an unused forward; moving it after
planning or popping twice corrupts the result lifecycle. This AST-only fixture
has a different dependency boundary from the existing Scheduler-import tests.
GPU stream safety and performance remain integration-test responsibilities.
"""

import __future__

import ast
import importlib.util
import unittest
from collections import Counter, deque
from enum import Enum, auto
from pathlib import Path
from types import MethodType, SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_ROOT = Path(__file__).resolve().parents[4]
_SCHEDULER_SOURCE = _ROOT / "python/sglang/srt/managers/scheduler.py"
_ENVIRON_SOURCE = _ROOT / "python/sglang/srt/environ.py"


class _Mode(Enum):
    EXTEND = auto()
    DECODE = auto()

    def is_extend(self):
        return self == self.EXTEND

    def is_decode(self):
        return self == self.DECODE


class _Disaggregation(Enum):
    NULL = auto()
    DECODE = auto()


class _Spec:
    def __init__(self, enabled=False):
        self.enabled = enabled

    def is_none(self):
        return not self.enabled


def _load_environ(path):
    spec = importlib.util.spec_from_file_location("_terminal_drain_env", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.envs


def _load_methods(path, envs, parallel, cuda=True, logs=None):
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    scheduler = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Scheduler"
    )
    names = {
        "event_loop_overlap",
        "_should_drain_c1_terminal_decode",
        "is_disable_overlap_for_batch",
    }
    methods = [
        node
        for node in scheduler.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert {node.name for node in methods} == names
    logs = [] if logs is None else logs
    namespace = dict(
        deque=deque,
        envs=envs,
        get_parallel=lambda: parallel,
        is_cuda=lambda: cuda,
        DisaggregationMode=_Disaggregation,
        ForwardMode=_Mode,
        DynamicGradMode=lambda: lambda fn: fn,
        logger=SimpleNamespace(
            info=lambda message: logs.append(("info", message)),
            warning=lambda message: logs.append(("warning", message)),
        ),
    )
    # Keep method bodies and decorators intact; only external context is supplied.
    exec(
        compile(
            ast.Module(body=methods, type_ignores=[]),
            str(path),
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        namespace,
    )
    return {name: namespace[name] for name in names}


class _Req:
    def __init__(self, rid, cap, prefill_chunks=1):
        self.rid = rid
        self.sampling_params = SimpleNamespace(max_new_tokens=cap)
        self.output_ids = []
        self.beam_group = self.grammar = self.to_finish = None
        self.is_retracted = self._finished = False
        self.inflight_middle_chunks = 0
        self.prefills_left = prefill_chunks
        self.tokens_issued = 0

    def finished(self):
        return self._finished


class _Batch:
    def __init__(self, reqs=(), mode=_Mode.DECODE, spec=None, iteration=None):
        self.reqs = list(reqs)
        self.forward_mode = mode
        self.spec_algorithm = spec or _Spec()
        self.forward_iter = iteration
        self.is_extend_in_batch = mode.is_extend()

    def copy(self):
        return _Batch(
            self.reqs, self.forward_mode, self.spec_algorithm, self.forward_iter
        )

    def grammar_needs_sync(self):
        return False


class _Runtime:
    """External planning/device/result collaborators, with a resource ledger."""

    def __init__(
        self, methods, caps=(2,), *, waiting_successor=False, prefill_chunks=1
    ):
        for name, method in methods.items():
            setattr(self, name, MethodType(method, self))
        self.is_generation = self.enable_overlap = True
        self.max_running_requests = 1
        self.require_mlp_sync = self.enable_hisparse = False
        self.enable_unified_memory = False
        self.spec_algorithm = _Spec()
        self.disaggregation_mode = _Disaggregation.NULL
        self.dllm_config = None
        self.gracefully_exit = self._engine_paused = False
        self.running_batch, self.last_batch = _Batch(), None
        self.requests = [
            _Req(str(i), cap, prefill_chunks) for i, cap in enumerate(caps)
        ]
        self.incoming = deque(self.requests)
        self.waiting = deque()
        self.waiting_successor = waiting_successor
        self.trace, self.launches, self.processed = [], [], set()
        self.releases = Counter()
        self.slot_owner = None
        self.iterations = self.serial = self.idle_count = 0
        self.token_to_kv_pool_allocator = SimpleNamespace(
            flush_opportunistic=lambda: None
        )

    def ingest_requests(self):
        self.iterations += 1
        assert self.iterations < 100, "loop lost a result or failed to reach idle"
        if self.waiting_successor:
            self.waiting.extend(self.incoming)
            self.incoming.clear()
        elif self.incoming and (self.slot_owner is None):
            self.waiting.append(self.incoming.popleft())

    def get_next_batch_to_run(self, running_batch, last_batch):
        # Deliberately no max-token or in-flight-output prediction here.
        if last_batch and last_batch.forward_mode.is_extend():
            req = last_batch.reqs[0] if last_batch.reqs else None
            if req and req.prefills_left:
                batch = _Batch([req], _Mode.EXTEND, self.spec_algorithm)
                return SimpleNamespace(batch_to_run=batch, running_batch=running_batch)
            running_batch = last_batch
        running_batch.reqs = [r for r in running_batch.reqs if not r.finished()]
        if running_batch.reqs:
            running_batch.forward_mode = _Mode.DECODE
            running_batch.is_extend_in_batch = False
            batch = running_batch
        elif self.waiting:
            req = self.waiting.popleft()
            assert self.slot_owner is None, "request slot reused before release"
            self.slot_owner = req
            batch = _Batch([req], _Mode.EXTEND, self.spec_algorithm)
        else:
            batch = None
        return SimpleNamespace(batch_to_run=batch, running_batch=running_batch)

    def run_batch(self, batch):
        req = batch.reqs[0]
        assert not req.finished(), "planned a forward after terminal result commit"
        self.serial += 1
        batch.forward_iter = self.serial
        if batch.forward_mode.is_extend():
            req.prefills_left -= 1
        produces_token = req.prefills_left == 0
        req.tokens_issued += int(produces_token)
        self.trace.append(("launch", req.rid, self.serial))
        self.launches.append((req.rid, batch.forward_mode))
        return SimpleNamespace(
            serial=self.serial,
            token=req.tokens_issued,
            produces_token=produces_token,
            sampled=not self.is_generation,
        )

    def _apply_war_barrier(self):
        self.trace.append(("barrier", self.serial))

    def launch_batch_sample_if_needed(self, result, batch):
        if result is not None:
            result.sampled = True
            self.trace.append(("sample", batch.reqs[0].rid, result.serial))

    def process_batch_result(self, batch, result):
        assert result.sampled, "popped a newly queued result before its sample"
        assert result.serial not in self.processed, "result processed twice"
        self.processed.add(result.serial)
        req = batch.reqs[0]
        self.trace.append(("process", req.rid, result.serial))
        if req.finished() or not result.produces_token:
            return
        req.output_ids.append(result.token)
        if len(req.output_ids) >= req.sampling_params.max_new_tokens:
            req._finished = True
            assert self.slot_owner is req, "released another request's slot"
            self.slot_owner = None
            self.releases[req.rid] += 1

    def on_idle(self):
        assert not self.result_queue and not self.running_batch.reqs
        self.idle_count += 1
        if not self.incoming and not self.waiting:
            self.gracefully_exit = True


class TestSchedulerTerminalDecodeDrain(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.envs = _load_environ(_ENVIRON_SOURCE)

    def _runtime(
        self,
        caps=(2,),
        *,
        enabled=True,
        consecutive=False,
        overrides=None,
        parallel=None,
        cuda=True,
        **kwargs,
    ):
        topology = SimpleNamespace(tp_size=1, pp_size=1, dp_size=1)
        for key, value in (parallel or {}).items():
            setattr(topology, key, value)
        logs = []
        methods = _load_methods(_SCHEDULER_SOURCE, self.envs, topology, cuda, logs)
        runtime = _Runtime(methods, caps, **kwargs)
        runtime.logs = logs
        for key, value in (overrides or {}).items():
            setattr(runtime, key, value)
        with (
            self.envs.SGLANG_ENABLE_C1_TERMINAL_DECODE_DRAIN.override(enabled),
            self.envs.SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP.override(consecutive),
            self.envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.override(0),
        ):
            runtime.event_loop_overlap()
        self.assertEqual(len(runtime.processed), len(runtime.launches))
        self.assertEqual(
            runtime.releases, Counter({r.rid: 1 for r in runtime.requests})
        )
        self.assertFalse(runtime.result_queue)
        self.assertIsNone(runtime.last_batch)
        self.assertIsNone(runtime.slot_owner)
        self.assertGreater(runtime.idle_count, 0)
        for req in runtime.requests:
            self.assertEqual(
                req.output_ids, list(range(1, req.sampling_params.max_new_tokens + 1))
            )
        return runtime

    def test_terminal_budget_keeps_required_overlap(self):
        for cap in (2, 4):
            for enabled in (False, True):
                with self.subTest(cap=cap, enabled=enabled):
                    runtime = self._runtime((cap,), enabled=enabled)
                    self.assertEqual(len(runtime.launches), cap + int(not enabled))
                    self.assertEqual(
                        runtime.logs,
                        [("info", "C1 terminal decode drain enabled.")]
                        if enabled
                        else [],
                    )
                    self.assertLess(
                        runtime.trace.index(("launch", "0", 2)),
                        runtime.trace.index(("process", "0", 1)),
                    )

    def test_one_token_request_keeps_prefill_fallback(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                runtime = self._runtime((1,), enabled=enabled)
                self.assertEqual(
                    runtime.launches, [("0", _Mode.EXTEND), ("0", _Mode.DECODE)]
                )

    def test_successor_result_is_not_popped_in_the_terminal_iteration(self):
        for waiting_successor in (False, True):
            with self.subTest(waiting_successor=waiting_successor):
                runtime = self._runtime((2, 4), waiting_successor=waiting_successor)
                self.assertEqual(
                    Counter(rid for rid, _ in runtime.launches), {"0": 2, "1": 4}
                )
                self.assertLess(
                    runtime.trace.index(("process", "0", 2)),
                    runtime.trace.index(("launch", "1", 3)),
                )
                self.assertLess(
                    runtime.trace.index(("launch", "1", 4)),
                    runtime.trace.index(("process", "1", 3)),
                )

    def test_unsupported_static_paths_preserve_existing_loop(self):
        cases = [
            {"cuda": False},
            *({"parallel": {key: 2}} for key in ("tp_size", "pp_size", "dp_size")),
            {"parallel": {"dp_size": None}},
            *(
                {"overrides": {key: value}}
                for key, value in (
                    ("is_generation", False),
                    ("enable_overlap", False),
                    ("max_running_requests", 2),
                    ("require_mlp_sync", True),
                    ("spec_algorithm", _Spec(True)),
                    ("disaggregation_mode", _Disaggregation.DECODE),
                    ("dllm_config", object()),
                    ("enable_hisparse", True),
                    ("enable_unified_memory", True),
                )
            ),
        ]
        for case in cases:
            with self.subTest(case=case):
                runtime = self._runtime(**case)
                self.assertEqual(len(runtime.launches), 3)
                self.assertEqual(
                    runtime.logs,
                    [
                        (
                            "warning",
                            "C1 terminal decode drain requested but disabled: "
                            "unsupported scheduler configuration.",
                        )
                    ],
                )

    def test_pending_result_identity_and_unsupported_requests(self):
        methods = _load_methods(_SCHEDULER_SOURCE, self.envs, SimpleNamespace())

        def pending():
            req = _Req("pending", 2)
            req.output_ids = [1]
            batch = _Batch([req], iteration=7)
            runtime = _Runtime(methods)
            runtime.running_batch = batch
            runtime.last_batch = batch.copy()
            runtime.result_queue = deque([(batch.copy(), object())])
            return runtime, req

        runtime, _ = pending()
        self.assertTrue(runtime._should_drain_c1_terminal_decode())
        changes = {
            "empty_queue": lambda s, r: s.result_queue.clear(),
            "two_results": lambda s, r: s.result_queue.append(s.result_queue[0]),
            "missing_last": lambda s, r: setattr(s, "last_batch", None),
            "missing_running": lambda s, r: setattr(s, "running_batch", None),
            "different_req": lambda s, r: setattr(
                s.running_batch, "reqs", [_Req("other", 2)]
            ),
            "two_reqs": lambda s, r: s.running_batch.reqs.append(_Req("other", 2)),
            "prefill": lambda s, r: setattr(
                s.result_queue[0][0], "forward_mode", _Mode.EXTEND
            ),
            "speculative": lambda s, r: setattr(
                s.running_batch, "spec_algorithm", _Spec(True)
            ),
            "missing_iteration": lambda s, r: setattr(
                s.result_queue[0][0], "forward_iter", None
            ),
            "last_iteration": lambda s, r: setattr(s.last_batch, "forward_iter", 6),
            "running_iteration": lambda s, r: setattr(
                s.running_batch, "forward_iter", 6
            ),
            "beam": lambda s, r: setattr(r, "beam_group", object()),
            "grammar": lambda s, r: setattr(r, "grammar", object()),
            "finished": lambda s, r: setattr(r, "_finished", True),
            "abort": lambda s, r: setattr(r, "to_finish", object()),
            "retracted": lambda s, r: setattr(r, "is_retracted", True),
            "middle_chunk": lambda s, r: setattr(r, "inflight_middle_chunks", 1),
            "no_cap": lambda s, r: setattr(r.sampling_params, "max_new_tokens", None),
            "not_terminal": lambda s, r: setattr(
                r.sampling_params, "max_new_tokens", 4
            ),
            "past_cap": lambda s, r: r.output_ids.append(2),
            "no_committed_token": lambda s, r: (
                r.output_ids.clear(),
                setattr(r.sampling_params, "max_new_tokens", 1),
            ),
        }
        for name, change in changes.items():
            with self.subTest(name=name):
                runtime, req = pending()
                change(runtime, req)
                self.assertFalse(runtime._should_drain_c1_terminal_decode())

    def test_existing_consecutive_prefill_boundary_still_drains_once(self):
        runtime = self._runtime(prefill_chunks=2, consecutive=True)
        self.assertEqual(
            runtime.launches,
            [("0", _Mode.EXTEND), ("0", _Mode.EXTEND), ("0", _Mode.DECODE)],
        )
        self.assertLess(
            runtime.trace.index(("process", "0", 1)),
            runtime.trace.index(("launch", "0", 2)),
        )
        self.assertLess(
            runtime.trace.index(("launch", "0", 3)),
            runtime.trace.index(("process", "0", 2)),
        )


if __name__ == "__main__":
    unittest.main()
