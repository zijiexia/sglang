"""Regress terminal lookahead through the production planner and batch filter.

The loop, admission-to-decode transitions, budget filter, batch filtering,
merging and result snapshots run from their unchanged production AST. Only
external device, token-admission and result-consumption boundaries are modeled.
The fixture detects extra/missing forwards, lost results and early slot reuse;
CUDA streams, FutureMap and real KV allocation remain integration-test work.
"""

import __future__

import ast
import importlib.util
import unittest
from collections import Counter, deque
from enum import Enum, IntEnum, auto
from http import HTTPStatus
from pathlib import Path
from types import MethodType, SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

_ROOT = Path(__file__).resolve().parents[4]
_SCHEDULER_SOURCE = _ROOT / "python/sglang/srt/managers/scheduler.py"
_BATCH_SOURCE = _ROOT / "python/sglang/srt/managers/schedule_batch.py"
_ENVIRON_SOURCE = _ROOT / "python/sglang/srt/environ.py"


class _Mode(Enum):
    EXTEND = auto()
    DECODE = auto()
    MIXED = auto()
    IDLE = auto()

    def is_extend(self):
        return self in (self.EXTEND, self.MIXED)

    def is_decode(self):
        return self == self.DECODE


class _Disaggregation(Enum):
    NULL = auto()
    PREFILL = auto()
    DECODE = auto()


class _Capture(IntEnum):
    NULL = 0

    def need_capture(self):
        return False


class _Spec:
    def __init__(self, enabled=False):
        self.enabled = enabled

    def is_none(self):
        return not self.enabled


class _Rows:
    """Device-array boundary with observable row identities, not tensor math."""

    def __init__(self, values):
        self.values = list(values)

    def __getitem__(self, indices):
        if isinstance(indices, _Rows):
            indices = indices.values
        if isinstance(indices, list):
            return _Rows(self.values[i] for i in indices)
        return self.values[indices]

    def to(self, *args, **kwargs):
        return self


class _SamplingRows:
    def __init__(self, reqs):
        self.rows = [r.rid for r in reqs]

    def filter_batch(self, indices, device_indices):
        assert indices == device_indices.values
        self.rows = [self.rows[i] for i in indices]

    def merge_batch(self, other):
        self.rows = self.rows + other.rows


def _class_methods(path, class_name, names):
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    methods = [
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert {node.name for node in methods} == names
    return methods


def _compile_methods(nodes, path, namespace):
    exec(
        compile(
            ast.Module(body=nodes, type_ignores=[]),
            str(path),
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        namespace,
    )
    return {node.name: namespace[node.name] for node in nodes}


def _load_environ():
    spec = importlib.util.spec_from_file_location("_output_budget_env", _ENVIRON_SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.envs


class _Req:
    def __init__(self, rid, cap, chunks=1):
        self.rid = rid
        self.sampling_params = SimpleNamespace(max_new_tokens=cap)
        self.output_ids = []
        self.beam_group = self.grammar = self.to_finish = None
        self.is_retracted = self._finished = False
        self.inflight_middle_chunks = 0
        self.prefills_left = chunks
        self.tokens_issued = 0
        self.token_indices_to_pool = None
        self.return_logprob = False
        self.prefix_indices = []
        self.extend_range = SimpleNamespace(end=1, length=1)
        self.origin_input_ids = [0]
        self.kv = SimpleNamespace(req_pool_idx=None, kv_committed_len=0)

    def finished(self):
        return self._finished

    def init_next_round_input(self, *args):
        pass


class _Pool:
    def __init__(self, capacity):
        self.capacity = capacity
        self.owners = {}
        self.peak_owners = 0

    def available_size(self):
        return self.capacity - len(self.owners)

    def allocate(self, req):
        if req.rid not in self.owners:
            assert self.available_size() > 0, (
                "request slot reused before result release"
            )
            self.owners[req.rid] = req
            self.peak_owners = max(self.peak_owners, len(self.owners))
            req.kv.req_pool_idx = req.rid

    def release(self, req):
        assert self.owners.pop(req.rid) is req, "released wrong/already released slot"


class _TokenAllocator:
    """Allocation/release boundary; production code decides capacity/retraction."""

    page_size = 1

    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.owned = {}
        self.checks = []
        self.retractions = []

    def available_size(self):
        return self.capacity - sum(self.owned.values())

    def check_decode_capacity(self, num_tokens, requests, **kwargs):
        available = self.available_size()
        self.checks.append((tuple(r.rid for r in requests), num_tokens, available))
        return num_tokens <= available

    def reserve(self, req, count):
        assert count <= self.available_size(), (
            "allocated before memory became available"
        )
        self.owned[req] = self.owned.get(req, 0) + count
        req.kv.kv_committed_len = self.owned[req]

    def release(self, req):
        self.owned.pop(req)
        req.kv.kv_committed_len = 0

    def flush_opportunistic(self):
        pass


class _Abort:
    def __init__(self, message, **kwargs):
        self.message = message

    def to_json(self):
        return {"message": self.message}


class _Batch:
    """Batch construction/device preparation; filter/copy/merge are production code."""

    _copy_fields = ()

    def __init__(self, reqs=(), **kwargs):
        for name in self._copy_fields:
            setattr(self, name, None)
        self.reqs = list(reqs)
        self.forward_mode = _Mode.DECODE
        self.spec_algorithm = _Spec()
        self.forward_iter = None
        self.chunked_req = None
        self.is_extend_in_batch = self.is_prefill_only = False
        self.batch_is_full = False
        self.model_config = SimpleNamespace(is_encoder_decoder=False)
        self.multimodal_inputs = None
        self.device = "cpu"
        self.return_logprob = self.has_grammar = self.return_hidden_states = False
        self.return_hidden_states_mode = _Capture.NULL
        self.spec_info = self.input_embeds = None
        self.sampling_info = _SamplingRows(self.reqs)
        for name in (
            "req_pool_indices",
            "req_pool_indices_cpu",
            "seq_lens",
            "seq_lens_cpu",
            "orig_seq_lens",
            "input_ids",
        ):
            setattr(self, name, _Rows(r.rid for r in self.reqs))
        self.top_logprobs_nums = [r.rid for r in self.reqs]
        self.token_ids_logprobs = [[r.rid] for r in self.reqs]
        for name, value in kwargs.items():
            setattr(self, name, value)

    @classmethod
    def init_new(cls, reqs, pool, allocator, cache, config, overlap, spec, **kwargs):
        for req in reqs:
            pool.allocate(req)
        return cls(
            reqs,
            model_config=config,
            spec_algorithm=spec,
            token_to_kv_pool_allocator=allocator,
            tree_cache=cache,
            req_to_token_pool=pool,
            **kwargs,
        )

    def prepare_for_extend(self):
        self.forward_mode = _Mode.EXTEND
        self.is_extend_in_batch = True

    def prepare_for_decode(self):
        self.forward_mode = _Mode.DECODE
        self.is_extend_in_batch = False
        self.assert_rows()

    def mix_with_running(self, other):
        self.merge_batch(other)
        self.forward_mode = _Mode.MIXED

    def release_req(self, idx, remaining, **kwargs):
        req = self.reqs[idx]
        self.token_to_kv_pool_allocator.retractions.append(req.rid)
        self.token_to_kv_pool_allocator.release(req)
        self.req_to_token_pool.release(req)
        req.is_retracted = True
        return True

    def grammar_needs_sync(self):
        return False

    def assert_rows(self):
        ids = [r.rid for r in self.reqs]
        assert self.req_pool_indices.values == ids
        assert self.req_pool_indices_cpu.values == ids
        assert self.seq_lens.values == ids
        assert self.seq_lens_cpu.values == ids
        assert self.orig_seq_lens.values == ids
        assert self.sampling_info.rows == ids


class _Adder:
    """Token/KV admission boundary; no output-budget or finish prediction."""

    def __init__(self, *args, **kwargs):
        self.can_run_list = []
        self.preempt_list = []
        self.new_chunked_req = None
        self.rem_chunk_tokens = 1

    def add_one_req(self, req, **kwargs):
        self.can_run_list.append(req)
        if req.prefills_left > 1:
            self.new_chunked_req = req
        return "CONTINUE"

    def add_chunked_req(self, req):
        self.can_run_list.append(req)
        return req if req.prefills_left > 1 else None


def _load_methods(envs, parallel, *, cuda=True, logs=None):
    logs = [] if logs is None else logs
    namespace = dict(
        deque=deque,
        envs=envs,
        get_parallel=lambda: parallel,
        get_schedule=lambda: SimpleNamespace(
            prefill_max_requests=None, retraction_policy="length"
        ),
        NewTokenRatioTracker=SimpleNamespace(
            estimate_new_token_ratio_after_retract=lambda reqs: 1.0
        ),
        beam_retraction_order=lambda indices, reqs: indices,
        num_beam_member_rows=lambda reqs: 0,
        FINISH_ABORT=_Abort,
        HTTPStatus=HTTPStatus,
        _make_abort_req=lambda req, **kwargs: req,
        is_cuda=lambda: cuda,
        DisaggregationMode=_Disaggregation,
        ForwardMode=_Mode,
        DynamicGradMode=lambda: lambda fn: fn,
        scheduler_stage_method=lambda stage: lambda fn: fn,
        SCHEDULER_STAGE_GET_NEXT_BATCH=0,
        NextBatchPlan=lambda **kwargs: SimpleNamespace(**kwargs),
        ScheduleBatch=_Batch,
        Req=_Req,
        PrefillAdder=_Adder,
        AddReqResult=SimpleNamespace(CONTINUE="CONTINUE", NO_TOKEN="NO_TOKEN"),
        PrefillStats=SimpleNamespace(from_adder=lambda *args, **kwargs: None),
        set_time_batch=lambda *args: None,
        set_schedule_time_batch=lambda *args: None,
        TEST_RETRACT=False,
        CaptureHiddenMode=_Capture,
        get_batch_return_hidden_states_mode=lambda reqs: _Capture.NULL,
        is_pin_memory_available=lambda device: False,
        strip_beam_tail=lambda batch: None,
        torch=SimpleNamespace(
            tensor=lambda values, **kwargs: _Rows(values),
            int64="int64",
            cat=lambda rows: _Rows(v for row in rows for v in row.values),
        ),
        logger=SimpleNamespace(
            info=lambda message: logs.append(("info", message)),
            warning=lambda message, *args: logs.append(
                ("warning", message % args if args else message)
            ),
        ),
    )
    batch_nodes = _class_methods(
        _BATCH_SOURCE,
        "ScheduleBatch",
        {
            "filter_batch",
            "copy",
            "merge_batch",
            "batch_size",
            "is_empty",
            "check_decode_mem",
            "new_tokens_required_next_decode",
            "retract_decode",
            "_get_decode_retraction_order",
        },
    )
    copy_node = next(node for node in batch_nodes if node.name == "copy")
    _Batch._copy_fields = {
        node.attr
        for node in ast.walk(copy_node)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    }
    for name, method in _compile_methods(batch_nodes, _BATCH_SOURCE, namespace).items():
        setattr(_Batch, name, method)
    scheduler_names = {
        "event_loop_overlap",
        "_init_overlap_output_budget",
        "is_disable_overlap_for_batch",
        "get_next_batch_to_run",
        "get_num_allocatable_reqs",
        "get_new_batch_prefill",
        "_get_new_batch_prefill_raw",
        "update_running_batch",
        "_filter_running_batch",
    }
    return _compile_methods(
        _class_methods(_SCHEDULER_SOURCE, "Scheduler", scheduler_names),
        _SCHEDULER_SOURCE,
        namespace,
    )


class _Runtime:
    def __init__(self, methods, caps=(2,), *, capacity=None, chunks=1, mixed=False):
        for name, method in methods.items():
            setattr(self, name, MethodType(method, self))
        self.is_generation = self.enable_overlap = True
        self.enable_pdmux = self.enable_overlap_mlx = False
        self.enable_overlap_output_budget = False
        self.max_running_requests = capacity or len(caps)
        self.require_mlp_sync = self.enable_hisparse = False
        self.enable_unified_memory = self.enable_priority_preemption = False
        self.enable_priority_scheduling = self.is_hybrid_swa = self.is_hybrid_ssm = (
            False
        )
        self.enable_fpm = self.enable_lora = False
        self.enable_hicache_storage = self.enable_lmcache = False
        self.enable_hierarchical_cache = self.enable_unified_cache_external_linker = (
            False
        )
        self.is_mixed_chunk = mixed
        self.spec_algorithm = _Spec()
        self.disaggregation_mode = _Disaggregation.NULL
        self.dllm_config = None
        self.model_config = SimpleNamespace(is_encoder_decoder=False)
        self.gracefully_exit = self._engine_paused = False
        self.running_batch, self.last_batch = _Batch(), None
        self.requests = [_Req(str(i), cap, chunks) for i, cap in enumerate(caps)]
        self.waiting_queue = list(self.requests)
        self.chunked_req = None
        self.trace, self.launches, self.processed = [], [], set()
        self.releases = Counter()
        self.iterations = self.serial = self.idle_count = 0
        # The physical pool is larger than the scheduling limit, as on a server.
        self.req_to_token_pool = _Pool(self.max_running_requests + 1)
        self.token_to_kv_pool_allocator = _TokenAllocator()
        self.aborted = []
        self.decode_offload_manager = None
        self.metrics_reporter = SimpleNamespace(enable_metrics=False)
        self.ipc_channels = SimpleNamespace(
            send_to_tokenizer=SimpleNamespace(
                send_output=lambda output, req: self.aborted.append(req.rid)
            )
        )
        self.beam_coordinator = SimpleNamespace(
            pending_member_rows=lambda batch: 0, retire_group=lambda req: None
        )
        self.dp_attn_adapter = SimpleNamespace(
            maybe_prepare_mlp_sync_batch=lambda batch, **kwargs: batch,
            maybe_convert_decode_to_extend=lambda batch: batch,
        )
        self.ngram_embedding_manager = SimpleNamespace(
            prepare_for_forward=lambda batch, **kwargs: batch
        )
        self.grammar_manager = SimpleNamespace(has_waiting_grammars=lambda: False)
        self.prefill_delayer = self.min_free_slots_delayer = None
        self.dynamic_chunk_sizer = None
        self.chunked_prefill_size = self.page_size = 1
        self.max_prefill_tokens = self.max_prefill_bs = 1024
        self.priority_scheduling_preemption_threshold = None
        self.new_token_ratio_tracker = SimpleNamespace(
            current=1.0, decay_step=lambda: None
        )
        self.processed_tokens_counter = self.truncation_align_size = 0
        self.tp_worker = SimpleNamespace(
            model_runner=SimpleNamespace(attn_backend=object(), prefill_aware_swa=False)
        )
        self.policy = SimpleNamespace(
            calc_priority=lambda *args, **kwargs: None,
            shortest_prefill_chunk_limit=lambda *args: 1,
        )
        self.tree_cache = SimpleNamespace(
            buffer_pipeline=None,
            storage_prefetch_retries=None,
            req_to_token_pool=self.req_to_token_pool,
        )
        self.load_inquirer = SimpleNamespace(_get_num_pending_tokens=lambda **kwargs: 0)
        self.on_ingest = None

    def _add_request_to_queue(self, req, **kwargs):
        self.waiting_queue.append(req)

    def process_pending_chunked_abort(self):
        pass

    def _process_hicache_events(self):
        pass

    def _should_defer_prefill(self):
        return False

    def _arm_prefill_decode_interval(self, batch):
        pass

    def stash_chunked_request(self, req):
        pass

    def ingest_requests(self):
        self.iterations += 1
        assert self.iterations < 150, "loop lost a result or failed to reach idle"
        if self.on_ingest:
            self.on_ingest(self)

    def run_batch(self, batch):
        batch.assert_rows()
        self.serial += 1
        batch.forward_iter = self.serial
        rows = []
        for req in batch.reqs:
            assert not req.finished(), "planned work after final result commit"
            assert self.req_to_token_pool.owners.get(req.rid) is req
            self.token_to_kv_pool_allocator.reserve(req, 1)
            if req.prefills_left > 0:
                req.prefills_left -= 1
            produces_token = req.prefills_left == 0
            req.tokens_issued += int(produces_token)
            rows.append((req, req.tokens_issued, produces_token))
            self.launches.append((req.rid, batch.forward_mode, self.serial))
        self.trace.append(("launch", self.serial, tuple(r.rid for r in batch.reqs)))
        return SimpleNamespace(
            serial=self.serial, rows=rows, sampled=not self.is_generation
        )

    def _apply_war_barrier(self):
        self.trace.append(("barrier", self.serial))

    def launch_batch_sample_if_needed(self, result, batch):
        if result is not None:
            result.sampled = True
            self.trace.append(("sample", result.serial))

    def process_batch_result(self, batch, result):
        assert result.sampled, "consumed result before its sampling launch"
        assert result.serial not in self.processed, "consumed result twice"
        assert batch.reqs == [row[0] for row in result.rows], (
            "pending row snapshot changed"
        )
        self.processed.add(result.serial)
        self.trace.append(("process", result.serial))
        for req, token, produces_token in result.rows:
            if req.finished() or req.is_retracted:
                continue
            if req.inflight_middle_chunks > 0:
                assert not produces_token
                req.inflight_middle_chunks -= 1
                continue
            assert produces_token
            req.output_ids.append(token)
            if (
                req.to_finish
                or len(req.output_ids) >= req.sampling_params.max_new_tokens
            ):
                req._finished = True
                self.req_to_token_pool.release(req)
                self.token_to_kv_pool_allocator.release(req)
                self.releases[req.rid] += 1
                self.trace.append(("release", req.rid, result.serial))

    def on_idle(self):
        assert not self.result_queue and not self.running_batch.reqs
        self.idle_count += 1
        if not self.waiting_queue:
            self.gracefully_exit = True


class TestSchedulerOverlapOutputBudget(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.envs = _load_environ()

    def _make_runtime(self, caps=(2,), *, parallel=None, cuda=True, **kwargs):
        topology = SimpleNamespace(
            tp_size=1,
            pp_size=1,
            dp_size=1,
            pp_max_micro_batch_size=kwargs.get("capacity") or len(caps),
        )
        for key, value in (parallel or {}).items():
            setattr(topology, key, value)
        logs = []
        methods = _load_methods(self.envs, topology, cuda=cuda, logs=logs)
        runtime = _Runtime(methods, caps, **kwargs)
        runtime.logs = logs
        return runtime

    def _run(self, runtime, *, enabled=True, consecutive=False):
        with (
            self.envs.SGLANG_ENABLE_OVERLAP_OUTPUT_BUDGET.override(enabled),
            self.envs.SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP.override(consecutive),
            self.envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.override(0),
        ):
            runtime._init_overlap_output_budget()
            runtime.event_loop_overlap()
        self.assertEqual(len(runtime.processed), runtime.serial)
        self.assertEqual(
            runtime.releases, Counter({r.rid: 1 for r in runtime.requests})
        )
        self.assertFalse(runtime.result_queue)
        self.assertIsNone(runtime.last_batch)
        self.assertFalse(runtime.req_to_token_pool.owners)
        self.assertFalse(runtime.token_to_kv_pool_allocator.owned)
        self.assertFalse(runtime.aborted)
        self.assertLessEqual(
            runtime.req_to_token_pool.peak_owners, runtime.max_running_requests
        )
        self.assertGreater(runtime.idle_count, 0)
        for req in runtime.requests:
            expected = (
                req.sampling_params.max_new_tokens
                if not req.to_finish
                else len(req.output_ids)
            )
            self.assertEqual(req.output_ids, list(range(1, expected + 1)))
        return runtime

    def test_exact_output_work_and_useful_overlap_at_any_batch_size(self):
        for caps in (
            (1,),
            (2,),
            (4,),
            (16,),
            (1, 2, 4),
            (4, 1, 2),
            (2, 4, 1),
            (2,) * 8,
        ):
            for enabled in (False, True):
                with self.subTest(caps=caps, enabled=enabled):
                    runtime = self._run(self._make_runtime(caps), enabled=enabled)
                    self.assertEqual(
                        runtime.logs,
                        [("info", "Overlap output budget enabled.")] if enabled else [],
                    )
                    counts = Counter(rid for rid, _, _ in runtime.launches)
                    self.assertEqual(
                        counts,
                        {str(i): cap + int(not enabled) for i, cap in enumerate(caps)},
                    )
                    if max(caps) > 1:
                        self.assertLess(
                            next(
                                i
                                for i, event in enumerate(runtime.trace)
                                if event[:2] == ("launch", 2)
                            ),
                            runtime.trace.index(("process", 1)),
                        )

    def test_one_token_final_prefill_and_chunk_boundaries(self):
        for cap in (1, 2, 4):
            for consecutive in (False, True):
                with self.subTest(cap=cap, consecutive=consecutive):
                    runtime = self._run(
                        self._make_runtime((cap,), chunks=3), consecutive=consecutive
                    )
                    self.assertEqual(len(runtime.launches), 3 + cap - 1)
                    self.assertEqual(runtime.requests[0].inflight_middle_chunks, 0)

    def test_mixed_prefill_uses_same_terminal_filter(self):
        for cap in (1, 2):
            with self.subTest(cap=cap):
                runtime = self._make_runtime((cap, 4), capacity=3, mixed=True)
                successor = _Req("2", 2)
                runtime.requests.append(successor)

                def inject(s):
                    if s.iterations == cap + 1:
                        s.waiting_queue.append(successor)

                runtime.on_ingest = inject
                self._run(runtime)
                self.assertEqual(
                    Counter(rid for rid, _, _ in runtime.launches),
                    {"0": cap, "1": 4, "2": 2},
                )
                self.assertTrue(
                    any(mode == _Mode.MIXED for _, mode, _ in runtime.launches)
                )
                target_serial = cap + 1
                event = next(
                    e for e in runtime.trace if e[:2] == ("launch", target_serial)
                )
                self.assertNotIn("0", event[2])
                self.assertIn("1", event[2])
                self.assertLess(
                    runtime.trace.index(event), runtime.trace.index(("process", cap))
                )

    def test_successor_waits_for_pending_slot_release(self):
        for cap in (1, 2):
            with self.subTest(cap=cap):
                runtime = self._run(self._make_runtime((cap, 4), capacity=1))
                self.assertEqual(
                    Counter(rid for rid, _, _ in runtime.launches), {"0": cap, "1": 4}
                )
                first_successor = next(
                    e for e in runtime.trace if e[0] == "launch" and "1" in e[2]
                )
                self.assertLess(
                    runtime.trace.index(("release", "0", cap)),
                    runtime.trace.index(first_successor),
                )

    def test_filter_preserves_pending_rows_and_metadata_alignment(self):
        runtime = self._make_runtime((2, 4, 2))
        reqs = runtime.requests
        for req in reqs:
            req.output_ids = [1]
            req.return_logprob = True
        batch = _Batch(reqs, forward_iter=7, return_logprob=True, batch_is_full=True)
        runtime.enable_overlap_output_budget = True
        runtime.last_batch = batch
        snapshot = batch.copy()
        runtime.result_queue = deque([(snapshot, object())])
        runtime._filter_running_batch(batch)
        self.assertEqual(batch.reqs, [reqs[1]])
        batch.assert_rows()
        self.assertEqual(batch.top_logprobs_nums, ["1"])
        self.assertEqual(batch.token_ids_logprobs, [["1"]])
        self.assertEqual(snapshot.reqs, reqs)
        self.assertEqual(snapshot.req_pool_indices.values, ["0", "1", "2"])
        self.assertFalse(batch.batch_is_full)
        self.assertTrue(all(not req.finished() for req in reqs))
        self.assertEqual([req.output_ids for req in reqs], [[1], [1], [1]])

    def test_terminal_pending_cancellation_is_consumed_once(self):
        runtime = self._make_runtime((2, 4))

        def abort(s):
            if s.iterations == 3:
                s.requests[0].to_finish = "cancel"

        runtime.on_ingest = abort
        self._run(runtime)
        self.assertEqual(runtime.releases["0"], 1)
        self.assertEqual(runtime.requests[0].output_ids, [1, 2])

    def test_static_fallback_does_not_change_original_work(self):
        self.assertIs(self.envs.SGLANG_ENABLE_OVERLAP_OUTPUT_BUDGET.default, False)
        cases = [
            ("parallel", {key: 2}) for key in ("tp_size", "pp_size", "dp_size")
        ] + [
            ("cuda", False),
            ("enable_overlap", False),
            ("enable_priority_preemption", True),
            ("is_hybrid_swa", True),
            ("is_hybrid_ssm", True),
            ("spec_algorithm", _Spec(True)),
            ("enable_unified_memory", True),
            ("model_config", SimpleNamespace(is_encoder_decoder=True)),
        ]
        for name, value in cases:
            with self.subTest(name=name, value=value):
                kwargs = {name: value} if name in ("parallel", "cuda") else {}
                runtime = self._make_runtime(**kwargs)
                if not kwargs:
                    setattr(runtime, name, value)
                if name == "spec_algorithm":
                    # Specialized allocation math is outside this ordinary fixture.
                    with self.envs.SGLANG_ENABLE_OVERLAP_OUTPUT_BUDGET.override(True):
                        runtime._init_overlap_output_budget()
                else:
                    self._run(runtime)
                self.assertFalse(runtime.enable_overlap_output_budget)
                self.assertEqual(
                    runtime.logs,
                    [
                        (
                            "warning",
                            "Overlap output budget requested but disabled: unsupported scheduler configuration.",
                        )
                    ],
                )
                if name != "spec_algorithm":
                    self.assertEqual(len(runtime.launches), 3)

    def _pressure_runtime(self, caps=(2, 8)):
        runtime = self._make_runtime(caps)
        runtime.waiting_queue.clear()
        runtime.token_to_kv_pool_allocator.capacity = 11 + 7 * (len(caps) - 1)

        def seed_pending(s):
            if s.iterations != 1:
                return
            rows = []
            for index, req in enumerate(s.requests):
                committed = 1 if index == 0 else 3
                req.output_ids = list(range(1, committed + 1))
                req.origin_input_ids = [0] * (10 if index == 0 else 4)
                req.prefills_left = 0
                req.tokens_issued = committed + 1
                s.req_to_token_pool.allocate(req)
                s.token_to_kv_pool_allocator.reserve(req, 11 if index == 0 else 7)
                rows.append((req, committed + 1, True))
            batch = _Batch(
                s.requests,
                forward_iter=1,
                token_to_kv_pool_allocator=s.token_to_kv_pool_allocator,
                req_to_token_pool=s.req_to_token_pool,
                tree_cache=s.tree_cache,
            )
            s.running_batch = s.last_batch = batch
            s.serial = 1
            s.launches = [(req.rid, _Mode.DECODE, 1) for req in s.requests]
            s.result_queue = deque(
                [(batch.copy(), SimpleNamespace(serial=1, rows=rows, sampled=True))]
            )

        runtime.on_ingest = seed_pending
        return runtime

    def test_pending_terminal_memory_defers_survivors_then_resumes(self):
        for caps in ((2, 8), (2, 8, 8), (2,)):
            with self.subTest(caps=caps):
                runtime = self._pressure_runtime(caps)
                planner = runtime.get_next_batch_to_run
                observations = []

                def observe_plan(**kwargs):
                    plan = planner(**kwargs)
                    observations.append(
                        (
                            plan.batch_to_run,
                            tuple(req.rid for req in plan.running_batch.reqs),
                            dict(runtime.token_to_kv_pool_allocator.owned),
                        )
                    )
                    return plan

                runtime.get_next_batch_to_run = observe_plan
                self._run(runtime)
                first_compute, retained, owned = observations[0]
                self.assertIsNone(first_compute)
                self.assertEqual(retained, tuple(str(i) for i in range(1, len(caps))))
                self.assertEqual(owned[runtime.requests[0]], 11)
                self.assertFalse(runtime.token_to_kv_pool_allocator.retractions)
                self.assertFalse(runtime.aborted)
                for req in runtime.requests:
                    self.assertIsNone(req.to_finish)
                    self.assertEqual(
                        len(req.output_ids), req.sampling_params.max_new_tokens
                    )
                self.assertEqual(runtime.trace.count(("process", 1)), 1)
                self.assertEqual(runtime.trace.count(("release", "0", 1)), 1)
                if len(caps) > 1:
                    second_launch = next(
                        e for e in runtime.trace if e[:2] == ("launch", 2)
                    )
                    self.assertLess(
                        runtime.trace.index(("process", 1)),
                        runtime.trace.index(second_launch),
                    )
                    self.assertEqual(set(second_launch[2]), set(retained))
                    self.assertEqual(runtime.token_to_kv_pool_allocator.checks[0][2], 0)
                else:
                    self.assertEqual(runtime.serial, 1)
                    self.assertFalse(runtime.token_to_kv_pool_allocator.checks)

    def test_ordinary_pressure_keeps_existing_retraction(self):
        for enabled, caps in ((False, (2, 8)), (True, (8, 8))):
            with self.subTest(enabled=enabled, caps=caps):
                runtime = self._pressure_runtime(caps)
                runtime.ingest_requests()
                with self.envs.SGLANG_ENABLE_OVERLAP_OUTPUT_BUDGET.override(enabled):
                    runtime._init_overlap_output_budget()
                plan = runtime.get_next_batch_to_run(
                    running_batch=runtime.running_batch, last_batch=runtime.last_batch
                )
                self.assertEqual(plan.batch_to_run.reqs, [runtime.requests[1]])
                self.assertEqual(runtime.token_to_kv_pool_allocator.retractions, ["0"])
                self.assertEqual(runtime.waiting_queue, [runtime.requests[0]])
                self.assertEqual(
                    runtime.token_to_kv_pool_allocator.owned, {runtime.requests[1]: 7}
                )
                self.assertFalse(runtime.aborted)
                self.assertEqual(runtime.result_queue[0][0].reqs, runtime.requests)

    def _stubbed_dispatch(self, runtime):
        source = ast.parse(_SCHEDULER_SOURCE.read_text())
        dispatch_node = next(
            node
            for node in source.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_dispatch_event_loop_once"
        )
        dispatch = _compile_methods(
            [dispatch_node],
            _SCHEDULER_SOURCE,
            runtime._init_overlap_output_budget.__func__.__globals__,
        )["_dispatch_event_loop_once"]
        selected = []
        for loop in (
            "event_loop_overlap",
            "event_loop_normal",
            "event_loop_pp",
            "event_loop_pdmux",
            "event_loop_overlap_mlx",
            "event_loop_overlap_disagg_prefill",
            "event_loop_normal_disagg_prefill",
            "event_loop_pp_disagg_prefill",
            "event_loop_overlap_disagg_decode",
            "event_loop_normal_disagg_decode",
            "event_loop_pp_disagg_decode",
        ):
            setattr(runtime, loop, lambda name=loop: selected.append(name))
        return lambda: dispatch(runtime), selected

    def test_real_dispatch_selects_loop_and_reports_eligibility(self):
        cases = [
            ({}, {}, "event_loop_overlap", True),
            ({"enable_overlap": False}, {}, "event_loop_normal", False),
            ({}, {"pp_size": 2}, "event_loop_pp", False),
            ({"enable_pdmux": True}, {}, "event_loop_pdmux", False),
            (
                {"enable_overlap": False, "enable_overlap_mlx": True},
                {},
                "event_loop_overlap_mlx",
                False,
            ),
        ]
        for mode, role in (
            (_Disaggregation.PREFILL, "prefill"),
            (_Disaggregation.DECODE, "decode"),
        ):
            cases.extend(
                [
                    (
                        {"disaggregation_mode": mode},
                        {},
                        f"event_loop_overlap_disagg_{role}",
                        False,
                    ),
                    (
                        {"disaggregation_mode": mode, "enable_overlap": False},
                        {},
                        f"event_loop_normal_disagg_{role}",
                        False,
                    ),
                    (
                        {"disaggregation_mode": mode},
                        {"pp_size": 2},
                        f"event_loop_pp_disagg_{role}",
                        False,
                    ),
                ]
            )
        for attrs, parallel, loop, supported in cases:
            for requested in (False, True):
                with self.subTest(loop=loop, requested=requested):
                    runtime = self._make_runtime(parallel=parallel)
                    for key, value in attrs.items():
                        setattr(runtime, key, value)
                    dispatch, selected = self._stubbed_dispatch(runtime)
                    with self.envs.SGLANG_ENABLE_OVERLAP_OUTPUT_BUDGET.override(
                        requested
                    ):
                        dispatch()
                    self.assertEqual(selected, [loop])
                    self.assertIs(
                        runtime.enable_overlap_output_budget, requested and supported
                    )
                    expected = []
                    if requested:
                        expected = (
                            [("info", "Overlap output budget enabled.")]
                            if supported
                            else [
                                (
                                    "warning",
                                    "Overlap output budget requested but disabled: unsupported scheduler configuration.",
                                )
                            ]
                        )
                    self.assertEqual(runtime.logs, expected)

    def test_redispatch_clears_stale_enabled_state(self):
        runtime = self._make_runtime()
        dispatch, selected = self._stubbed_dispatch(runtime)
        with self.envs.SGLANG_ENABLE_OVERLAP_OUTPUT_BUDGET.override(True):
            dispatch()
            self.assertTrue(runtime.enable_overlap_output_budget)
            runtime.disaggregation_mode = _Disaggregation.PREFILL
            dispatch()
            self.assertFalse(runtime.enable_overlap_output_budget)
            runtime.disaggregation_mode = _Disaggregation.NULL
            dispatch()
            self.assertTrue(runtime.enable_overlap_output_budget)
        previous_logs = list(runtime.logs)
        with self.envs.SGLANG_ENABLE_OVERLAP_OUTPUT_BUDGET.override(False):
            dispatch()
        self.assertFalse(runtime.enable_overlap_output_budget)
        self.assertEqual(runtime.logs, previous_logs)
        self.assertEqual(
            selected,
            [
                "event_loop_overlap",
                "event_loop_overlap_disagg_prefill",
                "event_loop_overlap",
                "event_loop_overlap",
            ],
        )

    def test_pending_identity_and_request_fallbacks(self):
        def setup():
            runtime = self._make_runtime()
            req = runtime.requests[0]
            req.output_ids = [1]
            batch = _Batch([req], forward_iter=7)
            runtime.last_batch = batch
            runtime.result_queue = deque([(batch.copy(), object())])
            runtime.enable_overlap_output_budget = True
            return runtime, batch, req

        changes = {
            "empty_queue": lambda s, b, r: s.result_queue.clear(),
            "two_results": lambda s, b, r: s.result_queue.append(s.result_queue[0]),
            "missing_last": lambda s, b, r: setattr(s, "last_batch", None),
            "mismatched_iteration": lambda s, b, r: setattr(b, "forward_iter", 8),
            "unknown_iteration": lambda s, b, r: setattr(
                s.result_queue[0][0], "forward_iter", None
            ),
            "idle": lambda s, b, r: setattr(
                s.result_queue[0][0], "forward_mode", _Mode.IDLE
            ),
            "speculative": lambda s, b, r: setattr(
                s.result_queue[0][0], "spec_algorithm", _Spec(True)
            ),
            "beam": lambda s, b, r: setattr(r, "beam_group", object()),
            "grammar": lambda s, b, r: setattr(r, "grammar", object()),
            "abort": lambda s, b, r: setattr(r, "to_finish", object()),
            "retracted": lambda s, b, r: setattr(r, "is_retracted", True),
            "middle_chunk": lambda s, b, r: setattr(r, "inflight_middle_chunks", 1),
            "no_cap": lambda s, b, r: setattr(
                r.sampling_params, "max_new_tokens", None
            ),
            "zero_cap": lambda s, b, r: setattr(r.sampling_params, "max_new_tokens", 0),
        }
        for name, change in changes.items():
            with self.subTest(name=name):
                runtime, batch, req = setup()
                change(runtime, batch, req)
                runtime._filter_running_batch(batch)
                self.assertEqual(batch.reqs, [req])
                self.assertFalse(req.finished())


if __name__ == "__main__":
    unittest.main()
