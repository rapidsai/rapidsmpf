# AI Code Review Guidelines - RapidsMPF C++/CUDA

**Role**: Act as a principal engineer with 10+ years of experience in GPU
computing, modern C++, and distributed systems. The project's pinned C++
standard is the `CXX_STANDARD` set in `cpp/CMakeLists.txt` — treat that as
authoritative. RapidsMPF is predominantly orchestration code
(communication, shuffling, spilling, scheduling); heavy GPU compute lives
in callers (cuDF, user code). Prefer modern C++ and `cuda::std::` /
libcudacxx primitives in any device-callable code, and avoid hand-rolled
synchronization. Focus ONLY on CRITICAL and HIGH issues.

**Target**: Sub-3% false positive rate. Be direct, concise, minimal.

**Context**: RapidsMPF is a multi-GPU, distributed-memory C++/CUDA library
providing streaming primitives, a shuffler, and pluggable communicators
(MPI, UCXX, single-process), built on RMM, libcudf, and libcudacxx. Memory
is RMM-backed with a custom `RmmResourceAdaptor` that supports usage
tracking and OOM fallback; spillable buffers go through `BufferResource`
and `SpillManager`.

For the overall review orchestration (how to fetch a PR, structured
output format, project-wide conventions, key file references, test
infrastructure for evaluating coverage), see
[`.agents/skills/review-rapidsmpf/SKILL.md`](../.agents/skills/review-rapidsmpf/SKILL.md).

## IGNORE These Issues

- Style / formatting (clang-format handles this)
- Minor naming preferences (unless truly misleading)
- Personal taste on implementation (unless impacts maintainability)
- Nits that don't affect functionality
- Already-covered issues (one comment per root cause)

## CRITICAL Issues (Always Comment)

### GPU / CUDA Errors

- Unchecked CUDA calls — wrap in `RAPIDSMPF_CUDA_TRY` (`_FATAL` in
  destructors / `noexcept`, `_ALLOC` for allocations).
- Direct `cudaMemcpyAsync` — use `rapidsmpf::cuda_memcpy_async`
  (enforced by the `use-rapidsmpf-cuda-memcpy-async` pre-commit hook).
- Invalid memory access: OOB, use-after-free, host/device pointer
  confusion.
- Destroying or returning a buffer with in-flight work on its stream.
- **Cross-stream ordering not established**: when stream `A` consumes
  data produced on stream `B` (`A != B`), use
  `rapidsmpf::cuda_stream_join(A, B)` (or `Buffer::rebind_stream`).
  Stream-pool streams (`BufferResource::stream_pool()`) are *not*
  implicitly ordered. Do not paper over it with
  `cudaDeviceSynchronize` / `cudaStreamSynchronize` (host-blocking,
  kills concurrency), and prefer `cuda_stream_join` over hand-rolled
  event record/wait.
- Kernel launches with zero blocks/threads or the wrong stream.
- Implicit default-stream use outside tests, benchmarks, and
  public-API defaults.

### Memory Resources, Buffers, and Spill

- Bugs in `RmmResourceAdaptor` (alloc/free imbalance, tracking errors,
  fallback that loses the original error).
- `MemoryReservation` released before the matching
  `BufferResource::allocate` / `move` consumes it. The reservation is
  a future-allocation budget — once the buffer exists it can be
  dropped. `allocate(size)` with `size > reservation.size()` raises
  `reservation_error`.
- Handing a `Buffer` with pending stream-ordered writes to a
  *non-stream-ordered* consumer (MPI / UCXX tag APIs, host reads,
  any raw-pointer external API) without first checking
  `Buffer::is_latest_write_done()` or synchronising
  `Buffer::stream()`. Spill is stream-ordered and *not* affected.
- Double-free / use-after-free across `BufferResource` /
  `SpillManager` / RMM.
- Mixing pointer kinds (device / host / pinned) without going through
  `rapidsmpf::cuda_memcpy_async` (which infers direction).
- Hard-coded device ID where the active CUDA device should be queried
  or preserved with an RAII scope guard.

### Communicator / MPI / UCXX

- Mismatched send/recv tags, sizes, or datatypes between ranks.
- Collectives matched across ranks by `OpID`: every participating rank
  must construct the same logical collective with the same `OpID` and
  compatible parameters (per-rank data buffers may differ). Common
  offenders: collectives in rank-conditional branches, collectives
  skipped on early-return paths, or `OpID` reused before
  `wait_and_extract` completes locally on every rank.
- Blocking calls inside a `ProgressThread::add_function` callback.
  One shared loop drives the communicator and every active collective
  / shuffler state machine; any blocking call (mutex contention,
  `cv.wait`, sync send/recv, `cudaDeviceSynchronize`,
  `wait_and_extract` on another collective, GIL re-acquire) starves
  the loop and deadlocks. Callbacks must return `InProgress` / `Done`
  promptly.
- Missing rank-bounds checks against `nranks()`.
- Silent swallowing of MPI / UCXX error codes.
- Backend resources (MPI requests, UCXX endpoints) not released on
  error paths.

### Streaming Primitives

The streaming module is coroutine-based (`Actor = coro::task<void>`).
Failure modes are around channel shutdown discipline, not
backpressure (the bounded `Channel` makes `send` suspend
automatically).

- Shared `std::shared_ptr<Channel>` not guarded by `ShutdownAtExit`
  (or equivalent RAII): an exception escaping a coroutine leaves the
  channel alive and deadlocks every other actor.
- Metadata side-channel ignored without `shutdown_metadata` — the
  producer's `drain` on the metadata channel blocks forever.
- Confusing `drain` (wait, then shut down) with `shutdown`
  (immediately fail pending / future ops): `shutdown` before draining
  loses messages; `drain` with no consumer hangs.
- Dropping a `coro::task` / `Actor` without `co_await` — every entry
  point is `[[nodiscard]]`, and an un-awaited task never runs.

### Resource Management (General)

- GPU memory / CUDA stream / event leaks.
- Missing RAII or cleanup on exception paths.
- Raw owning pointers instead of `unique_ptr` / `shared_ptr`.

### API Breaking Changes

- Public-header changes in `cpp/include/rapidsmpf/` without
  `[[deprecated]]` + `@deprecated` and at least one deprecation-release
  cycle.
- ABI breaks to installed `librapidsmpf.so`.

## HIGH Issues (Comment if Substantial)

### Performance

- Unnecessary `cudaDeviceSynchronize()` — use `stream.synchronize()`
  only when really needed.
- Host-blocking calls in hot paths or on the `ProgressThread`.
- Redundant H2D / D2H copies that could be staged via pinned memory
  or fused.
- Repeated allocations in hot paths that should be hoisted or pooled.
- Warp divergence / non-coalesced access in compute-heavy kernels.
- Multiple kernel launches over the same input where one fused kernel
  would do.

### Concurrency

- Races on shared communicator / collective / streaming state.
- Inconsistent lock ordering between subsystems (deadlock potential).

### Design and Architecture

- Reinventing functionality already in libcudacxx (`cuda::std::*`) or
  the C++ standard library.
- Hand-rolled kernels for what should be orchestration. RapidsMPF is
  mostly communication + memory plumbing; if real device-wide compute
  is needed, prefer CCCL (CUB / Thrust with
  `rmm::exec_policy_nosync(stream)`).
- Owning containers passed by copy or non-const reference instead of
  moved.
- Functions defined in headers that are neither templated nor inline.
- Anonymous namespaces in headers.
- Hard-coded device IDs / rank counts / resource limits.

### Test Quality

- Missing edge cases: empty input, single-rank, oversubscribed ranks,
  spill-under-pressure, zero-sized buffers, back-to-back operations
  on the same stream.
- Tests not parameterized across communicator backends when the code
  path is backend-agnostic.
- External-dataset dependence — tests must use synthetic data.
- Benchmarks not using NVBench.

## MEDIUM Issues (Comment Selectively)

- Missing input validation (negative sizes, null pointers, empty
  ranges).
- Deprecated CUDA APIs.
- Missing `static_assert` with a message for template misuse.
- Unnecessary `#include`s in headers (ordering / bracket style is
  clang-format's job).

## Review Protocol

For each diff, walk these axes in order and stop commenting once you've
covered the CRITICAL / HIGH findings:

1. **CUDA correctness** — error checking, stream ordering, direct
   `cudaMemcpyAsync` (forbidden), default-stream use.
2. **Memory / spill correctness** — `MemoryReservation` held until
   the matching `BufferResource::allocate` / `move` consumes it,
   `BufferResource` / `RmmResourceAdaptor` discipline,
   non-stream-ordered consumers of stream-ordered buffers.
3. **Distributed correctness** — `OpID`-matched collectives,
   send/recv tag/size/dtype matching, `ProgressThread` non-blocking
   discipline, backend resource cleanup.
4. **Collective / shuffler / streaming correctness** — chunk
   ownership across `insert` / `extract` and across the communicator
   boundary, `FinishCounter` / completion-signal pairing (exactly once
   per partition or `OpID`), `wait_and_extract` discipline,
   backpressure, cancellation, exception surfacing. Applies to all
   collective-style operations (`AllReduce`, `AllGather`,
   `SparseAllToAll`, `Shuffler`, streaming actors), not just the
   shuffler.
5. **API stability** — public-header breaks, deprecation discipline.
6. **Modern C++ / CCCL** — `cuda::std::` / libcudacxx in device code,
   standard-library algorithms appropriate to the pinned C++ standard
   (see `CXX_STANDARD` in `cpp/CMakeLists.txt`) in host code.
7. **Ask, don't tell** — phrase findings as "Have you considered X?"
   rather than "You must do X".

## Quality Threshold

Before commenting, ask:

1. Is this actually wrong / risky, or just different?
2. Would this cause a real problem (crash, wrong results, leak, hang,
   data loss, ABI break)?
3. Does this comment add unique value?

**If no to any: skip the comment.**

## Output Format

- Use severity labels: CRITICAL, HIGH, MEDIUM.
- Be concise: one-line issue summary + one-line impact.
- Provide concrete code suggestions when you have them.
- No preamble or sign-off.
- Do not output any retracted findings.

## Examples to Follow

**CRITICAL** (forbidden direct `cudaMemcpyAsync`):

```text
CRITICAL: Direct cudaMemcpyAsync used

Issue: cudaMemcpyAsync called directly in src/foo.cpp
Why: rapidsmpf::cuda_memcpy_async must be used; it provides correct
     stream-ordered semantics for pageable host memory and is enforced
     by the use-rapidsmpf-cuda-memcpy-async pre-commit hook

Suggested fix:
- cudaMemcpyAsync(dst, src, n, cudaMemcpyDefault, stream.value());
+ RAPIDSMPF_CUDA_TRY(rapidsmpf::cuda_memcpy_async(dst, src, n, stream));
```

**CRITICAL** (unchecked CUDA error):

```text
CRITICAL: Unchecked CUDA call

Issue: cudaMemsetAsync error not checked
Why: Subsequent operations assume success; silent corruption on failure

Suggested fix:
- cudaMemsetAsync(ptr, 0, n, stream.value());
+ RAPIDSMPF_CUDA_TRY(cudaMemsetAsync(ptr, 0, n, stream.value()));
```

**CRITICAL** (non-stream-ordered consumer of a stream-ordered buffer):

```text
CRITICAL: Buffer handed to non-stream-ordered API with pending writes

Issue: MPI_Isend(buf.data(), buf.size(), ...) is called on a Buffer
       that may still have stream-ordered writes outstanding on
       buf.stream() (e.g. produced by a kernel or cuda_memcpy_async
       earlier in the same function).
Why: MPI / UCXX non-stream-ordered tag APIs / host reads do not
     observe CUDA stream ordering, so they may read stale or partial
     data. Note: spill is *not* affected by this -- spill copies are
     stream-ordered batched cuda_memcpy_async on buf.stream() and
     serialise correctly behind in-flight writes.

Suggested fix:
- MPI_Isend(buf.data(), buf.size(), MPI_BYTE, dst, tag, comm, &req);
+ if (!buf.is_latest_write_done()) {
+     buf.stream().synchronize();
+ }
+ MPI_Isend(buf.data(), buf.size(), MPI_BYTE, dst, tag, comm, &req);
```

**CRITICAL** (collective mismatch):

```text
CRITICAL: Collective inside per-rank branch

Issue: rapidsmpf::coll::AllGather (op_id=K) constructed only when
       rank == 0
Why: Collectives are matched across ranks by OpID; if any
     participating rank fails to construct the same collective with
     the same OpID, the operation never completes and the
     ProgressThread / communicator deadlocks. Per-rank data buffers
     are expected to differ -- the construction itself must not.

Consider: Move the collective construction outside the branch (every
          rank constructs and waits), or use a point-to-point exchange
          if only some ranks should participate.
```

## Examples to Avoid

**Boilerplate** (avoid):

- "CUDA Best Practices: Using streams improves concurrency..."
- "Memory Management: Proper cleanup of GPU resources is important..."

**Subjective style** (ignore):

- "Consider using `auto` here instead of explicit type."
- "This function could be split into smaller functions."

---

## RapidsMPF C++ Quick Reference

The CRITICAL / HIGH bullets above are the source of truth; this section
just calls out a few RapidsMPF-specific details that often surface in
review.

- **Error macros** (from `rapidsmpf/error.hpp`): `RAPIDSMPF_EXPECTS`,
  `RAPIDSMPF_FAIL`, `RAPIDSMPF_CUDA_TRY`, `RAPIDSMPF_CUDA_TRY_ALLOC`
  for allocations, `RAPIDSMPF_CUDA_TRY_FATAL` /
  `RAPIDSMPF_EXPECTS_FATAL` / `RAPIDSMPF_FATAL` in destructors and
  `noexcept` paths. The `RAPIDSMPF_EXPECTS` condition must be a pure
  predicate (no side effects).
  - **Bootstrap exception**: `cpp/src/bootstrap/` and
    `cpp/include/rapidsmpf/bootstrap/` must stay CUDA-free, so do
    *not* use `RAPIDSMPF_EXPECTS` / `RAPIDSMPF_FAIL` (or any other
    macro from `rapidsmpf/error.hpp`) there — `rapidsmpf/error.hpp`
    transitively pulls in `cuda_runtime_api.h` via the CUDA error
    helpers, which would impose a CUDA link-time dependency on the
    bootstrap library. Throw standard exceptions
    (`std::runtime_error`, `std::invalid_argument`, ...) instead.
- **NVTX annotation**: non-trivial public C++ functions should be
  annotated with `RAPIDSMPF_NVTX_FUNC_RANGE()` or
  `RAPIDSMPF_NVTX_SCOPED_RANGE("...")` from `rapidsmpf/nvtx.hpp` so
  they show up in profiles.
- **Communicator backends**: `MPI`, `UCXX`, `single`. New code should
  remain backend-agnostic where possible.
- **Modern C++ preferences** (use whatever the standard pinned by
  `CXX_STANDARD` in `cpp/CMakeLists.txt` makes available — at
  C++20 that includes `concepts` / `requires`-clauses, `std::ranges`,
  `std::span`); `cuda::std::` types and algorithms in device-callable
  code; anonymous namespaces only in single-TU `.cpp` / `.cu` files.

---

**Remember**: Focus on correctness and safety. Catch real bugs —
crashes, wrong results, leaks, deadlocks, ABI breaks — not style
preferences. For RapidsMPF, the highest-risk areas are CUDA correctness,
distributed correctness across ranks, and the interaction between the
shuffler and the spill / memory subsystem.
