# Link loss, print state, and file operations

Status: implemented architecture decision.

## Problem

Some printers temporarily lose their local control link while continuing a
physical print or file transfer. Treating that transport edge as proof that the
printer stopped can fail a server-side job and return its queue item. When the
device reconnects, the same item may be dispatched and printed again.

Production evidence from a Bambu P1 fleet showed short MQTT gaps during live
prints, repeated five-minute redispatches, and queue items physically printed
more than once. The exact device is useful evidence, but the failure is not
brand-specific: transport reachability and physical job state are different
facts.

## Domain boundaries

Three state owners must remain separate:

| Owner | Knows | Must not infer |
|---|---|---|
| Device driver | Link generation, reachability, reconnect and send ability | Physical print outcome |
| Printer client | Device reports and their projection to cloud printer status | File-operation completion from link state alone |
| File transfer | One accepted FILE or START operation, its cancellation and terminal | General device allocation or reconnect policy |

A disconnected link is an observation, not a printer status. A client may
project it to `OFFLINE` only when doing so cannot contradict protected device or
operation state.

## Invariants

- One accepted FILE or START operation produces at most one terminal:
  `READY` or `ERROR`.
- Repeating the same active job is idempotent. A different job preempts it and
  receives a distinct operation.
- LAN upload, printer-side URL download, and staged START use the same admission,
  cancellation, and terminal path.
- Link loss alone does not complete, fail, or replay a file operation.
- A prepared file is cleared only after the device accepts its START command.
- Every operation and watchdog is cancelled and awaited during client teardown.
- Drivers publish each true connection transition once. They suppress only a
  duplicate or stale observation of the same session.
- Brand code supplies protocol behavior through public typed methods; it does
  not schedule transfer tasks or mutate transfer internals.
- Blocking work uses the app-owned executor lanes. No brand creates a thread.

## File-operation surface

`FileTransfer` owns the operation lane. Brand clients use only these commands:

- `submit(data)` accepts a FILE demand.
- `start_staged()` accepts START for the prepared file.
- `started()` / `rejected(message)` project a device start outcome.
- `progress(percent)` projects printer-side URL download progress.
- `close()` cancels and awaits owned work.

`FileTransfer` is final. A brand composes a `PrintFileDriver` implementing the
protocol operations it supports: `upload`, `start_uploaded`,
`upload_and_start`, `start_url`, `cancel_start`, and `close`. `PreparationKind`
selects upload, atomic upload-and-start, or printer-side URL without brand labels
or string comparisons; staged START reuses the accepted upload. A start returns
`StartDisposition.COMPLETE` or `AWAIT_DEVICE`. Only `RetryableFileError` consumes
the controller's one retry budget; permanent `FileOperationError` failures do
not retry. Link state is deliberately not an input to this object.

## Current architecture slice

The shared operation lane and composed driver surface are implemented. The old
parallel dispatch locks, task registries, subclass hooks, per-brand retry loops,
and Bambu URL scheduler are gone. Peripheral action/state semantics are also
shared by the library rather than copied across brands.

Liveness is one immutable `DeviceSession` per driver:

- generation identifies a continuous connection;
- reachability is `never_seen`, `up`, `down`, or `stopped`;
- a source token pairs lease identity with wire generation, so a stale event
  cannot clobber a replacement transport whose generation restarted at one;
- `observed_at` is last activity while up and the first observation while down;
- duplicate edges preserve that first down observation;
- reconnect starts a new generation and clears the outage once;
- teardown awaits the driver, transitions to `stopped`, and detaches the lease.

The driver records facts only. `PrinterClient` aggregates all declared drivers
and derives a fixed deadline from the most recent down observation plus
`device_loss_grace`. There is no outage timer, flag cluster, or optional
capability lookup: the normal client tick performs the projection. Idle loss is
immediate; printing and a `DOWNLOADING` file operation retain their status until
the deadline. A reconnect leaves status restoration to the next real device
report. File transfer sees the raw edge immediately but remains non-terminal.

The projection is covered by behavior, not field-presence tests:

1. A sub-grace flap during transfer remains non-terminal and keeps the same
   operation.
2. A sub-grace flap during printing never emits a job-killing status.
3. Reconnect resumes normal device-report projection without duplicate handlers.
4. Deadline expiry produces one bounded offline outcome.
5. Teardown makes the session terminal and awaits owned work.

## Rejected shapes

- Suppressing disconnect events in the wire or driver hides true edges from
  device bookkeeping and reconnect setup.
- A brand override that coordinates client status and transfer cancellation
  splits one policy between two owners.
- Reflective optional capabilities (`getattr`, `hasattr`) make misspelling and
  partial initialization valid runtime states.
- Independent booleans/timestamps for “down”, “holding”, “reported”, and
  “deadline” permit impossible combinations. The session record is the sole
  reachability fact; projection policy is derived from it.
- Tests that search source text or assert private field names preserve a past
  implementation, not the safety contract above.

## Verification

The library remains Python 3.9 compatible and brand-neutral. Integration tests
exercise public demands and observable messages/state. Import-DAG and
brand-neutrality guards remain structural because they protect package
boundaries; lifecycle tests remain behavioral.
