# Target printer-client architecture

Status: proposal. This describes the target, not the current code.

The goal is one runtime path that stays connected to every configured printer
until removal. It must be direct enough that an integration author can understand
the whole lifecycle without reading the framework internals.

## What is wrong now

The 2.0 work moved code around but did not really collapse the system.

| Current area | Size/shape | Problem |
|---|---:|---|
| `Client` | 771 lines, 50 methods | Cloud protocol, membership, state and printer behavior are mixed. |
| `PrinterClient` | 772 lines, 64 methods | Brands inherit another large class; 41 methods are trivial wrappers or no-op hooks. |
| Device drivers | 774 lines | `DeviceDriver -> LeaseDriver -> MqttDriver/WsDriver`, even though every current printer has one driver. |
| Wire connection stack | 2,817 lines in the main transport/retry/pool/lease modules | More than one layer owns start, stop, reconnect, generation and state. |
| `FileTransfer` | 753 lines, 37 methods, 24 private | A nine-field lifecycle record, task, watchdog, `ContextVar`, thread event and several flags describe one operation. |
| Brand file code | 2,132 lines | Integrations implement framework routing and confirmation state instead of device capabilities. |
| Current brand printer classes | 5,311 lines | The large shared base did not make the brand classes small. |
| Brand `integration.py` files | 1,100 lines, 46 private factories | `IntegrationSpec` is a bag of optional callbacks. |

The inherited authoring surface is roughly 1,500 lines before an integration
author writes a printer. The live device message path is also too long:

```text
Paho callback
  -> Paho courier/events
  -> pool endpoint
  -> lease courier/events
  -> lease driver
  -> PrinterClient
  -> brand printer
```

That is why small connection fixes grow another lock, flag, timer or retry loop.

## Do not preserve these bugs

The current file lifecycle has two correctness bugs that the replacement must
not copy.

1. `FileTransfer` calls `cancel_start()` when device confirmation is missing for
   60 seconds. Bambu implements `cancel_start()` with `StopCommand`. If the print
   started and its report was lost during a disconnect, the client can stop a
   valid physical print. Missing confirmation means unknown. Reconnect and
   reconcile it. Only an explicit user cancel may send stop.
2. Bambu FTPS calls the upload progress callback from an offload thread. The
   callback mutates `FileProgressState` and lifecycle timestamps directly. All
   shared printer state must be mutated on its owning event loop.

There are other concrete bad states to remove:

- `LeaseDriver` has `_restart_lock`, `_restarting`, `_restart_again`, `_stopped`
  and `_start_failure` beside the transport retry state.
- Several protocol objects are invalid until code assigns `device.sender` after
  construction.
- Pool identity omits connection-affecting retry, TLS and logging policy, so the
  first lease can silently choose policy for later leases.
- File capability metadata has already drifted from runtime behavior: Moonraker
  and Snapmaker attach file support but advertise only camera support.

## Target shape

```text
Runtime
├── CloudSession
├── configured printers: dict[id, PrinterRuntime]
│   └── PrinterRuntime
│       ├── PrinterAdapter             brand-owned, shallow public API
│       ├── PersistentConnection       one desired-state supervisor
│       │   └── TransportSession       Paho, WebSocket or poll implementation
│       ├── PrintLifecycle
│       │   └── print/file capabilities
│       └── CameraService
└── extensions                         PaperCut and other non-printer services
```

The hot message path is exactly:

```text
TransportSession -> PersistentConnection -> PrinterRuntime -> adapter.device_event
```

Do not insert a mapper, effect bus, middleware chain, service locator or generic
command router into that path.

## `Runtime`

`Runtime` replaces the orchestration split between `Host`, `ClientApp`,
`Scheduler`, `ClientList` and the orchestration half of
`ClientConnectionManager`.

It owns:

- configuration persistence;
- the configured-printer mapping;
- one task per `PrinterRuntime`, rather than a fleet-wide tick loop;
- discovery and onboarding;
- process-owned executors and shared host telemetry;
- extensions;
- startup and shutdown order.

Its public lifecycle is `add`, `remove`, `start` and `close`. A configured entry
remains desired until `remove`.

`RuntimeServices` contains required process services. It is not an all-optional
service locator and integrations do not look themselves up by a string key.

## `CloudSession`

`CloudSession` owns only the SimplyPrint side:

- perpetual cloud connection supervision;
- desired printer membership;
- inbound routing by printer ID;
- registration rate limits;
- idempotent membership replay after reconnect;
- outbound writes that clear dirty state only after a successful send.

Delete `Client.active`, `ClientState`, `_pending_action_*`, `ensure_added` and
`ensure_removed`. Allocation, cloud membership and device reachability are
different facts and must not share a flag.

## `PrinterRuntime`

`PrinterRuntime` is final. Brands do not subclass it.

It owns:

- `PrinterState`;
- device connection and reachability projection;
- dispatch of typed cloud demands;
- print/job state and offline grace;
- file and camera lifecycle;
- per-printer cloud timers;
- teardown of everything owned by the configured printer.

Common demand behavior is called directly. Required runtime control flow does
not travel through the general event bus.

## `PrinterAdapter`

This is the class an integration author writes. It is small and all extension
methods are public.

The exact Python spelling should be proved in the first vertical slice, but the
surface is constrained to:

- `connection() -> ConnectionSpec`: a pure description of the current endpoint,
  evaluated for every fresh attempt and after a config change;
- `device_event(io, event, state) -> PrinterObservation`: one required typed
  entry point for connected, message and disconnected events;
- `demand(io, demand, state) -> DemandResult`: one required typed entry point for
  brand device commands;
- explicit instance capability values for prints/files and camera.

`DeviceIO` is passed into calls. It is never assigned later as `device.sender`.
The adapter mutates the passed `PrinterState` explicitly for ordinary telemetry.
`PrinterObservation` contains only facts that the shared lifecycle owns: mapped
status and an optional `JobObservation`. It must not grow into an effects bag.

The adapter constructor creates data and protocol objects only. It starts no
socket, task, thread, discovery listener or registration.

Do not model multiple device connections until a real integration needs them.
Every current production integration has one control connection. HTTP command
requests beside an MQTT/WebSocket/poll session do not make a second supervised
connection.

One adapter can still be large. Split its local protocol, model and pure mapping
code into ordinary modules when that reads better. Do not expose every local
split as another framework layer.

## Persistent device connection

Replace the public driver hierarchy with three things:

- `ConnectionSpec`: immutable endpoint, auth, liveness and transport factory;
- `DeviceIO`: the live send/publish surface;
- `PersistentConnection`: the one final supervisor.

`PersistentConnection` owns desired state, one attempt generation, retry,
reconfiguration, keepalive failure, event delivery and teardown.

The rules are simple:

- Its loop exits only on `close`.
- Transport identity lasts until the configured endpoint changes or the printer
  is removed. Recovery does not replace an MQTT client to erase unexplained
  state.
- A transient open error, EOF or wire-level keepalive timeout stays inside the
  transport's one supported reconnect path with deterministic bounded backoff.
- Application-level silence reports device inactivity and continues probing. It
  never closes an otherwise healthy transport.
- Paho retains one client. Paho owns ordinary socket reconnects; the surrounding
  worker only catches `loop_forever()` exiting or raising and re-enters it on the
  same client. A worker failure cannot silently end supervision.
- A config change fences the old generation, closes it, and starts the new spec.
- Stale events from an old generation are ignored.
- Parser/adapter failure is reported without killing supervision.
- `close` cancels the attempt/backoff and awaits the backend exactly once.
- Each failure has one retry owner. Do not layer a generic retry loop over a
  library that already owns the same reconnect lifecycle.

Pooling may remain an internal optimization only where several logical MQTT
consumers really share one broker session. Pool identity must include every
connection-affecting setting. Pool/lease types are not part of the integration
API.

## Print lifecycle and generic file access

Storage and printing are separate capabilities.

```python
@dataclass(frozen=True)
class DeviceFile:
    path: PurePosixPath       # relative to the store root
    size: Optional[int] = None
    digest: Optional[FileDigest] = None


class FileUploader(Protocol):
    async def upload(
        self,
        source: Path,
        destination: PurePosixPath,
        *,
        overwrite: bool,
        progress: ProgressCallback,
    ) -> DeviceFile: ...

    async def close(self) -> None: ...


class FileStore(FileUploader, Protocol):
    async def list(self, directory: PurePosixPath) -> Tuple[DeviceFile, ...]: ...
    async def stat(self, path: PurePosixPath) -> Optional[DeviceFile]: ...
    async def delete(self, path: PurePosixPath) -> None: ...
```

An upload-only HTTP endpoint implements `FileUploader`. FTP, Moonraker, Duet,
PrusaLink and OctoPrint can implement the full `FileStore`. There are no default
unsupported methods and no feature detection.

The shared library provides three direct print compositions:

- `StoredPrints(uploader, start_file)`: download, upload, optionally stage, then
  start the returned `DeviceFile`;
- `UrlPrints(start_url)`: hand the URL to the printer; no fake staging support;
- `DirectPrints(start_local)`: download locally, then call an atomic
  upload-and-start endpoint.

They use one small `PrintLifecycle`. Integrations do not subclass it or choose a
hidden route for every operation.

One accepted operation has one coroutine and one state enum. Do not rebuild the
current task plus watchdog task plus `ContextVar` plus `threading.Event` plus flag
record. Timeout and cancellation stay with the operation that owns them.

Every accepted device start returns a `StartReceipt`. It means accepted, not
proven started. A canonical device job observation later confirms or rejects the
receipt. An ambiguous send is never blindly retried. A missing observation never
sends stop.

Normalize `FileDemandData` once into `PrintRequest`. Keep display name separate
from filesystem-safe storage name. Replace start-option dictionaries with typed
settings. Keep vendor enums at the vendor protocol boundary.

Progress entering from a synchronous adapter is marshalled to the printer loop
and applied monotonically.

### Route mapping

| Integration | Storage | Shared print path | Start transport |
|---|---|---|---|
| Anycubic | upload-only temporary HTTP URL | `StoredPrints` | MQTT |
| Bambu LAN | full FTPS store | `StoredPrints` | MQTT |
| Bambu cloud | none | `UrlPrints` | MQTT |
| Creality | upload-only HTTP | `StoredPrints` | WebSocket |
| Elegoo CC1 | upload-only chunked HTTP | `StoredPrints` | WebSocket |
| Elegoo CC2 | upload-only ranged HTTP | `StoredPrints` | MQTT |
| Duet | full HTTP store | `StoredPrints` | HTTP G-code |
| Moonraker | full HTTP store | `StoredPrints` | WebSocket RPC |
| Snapmaker | Moonraker store plus a typed pre-start operation | `StoredPrints` | WebSocket RPC |
| Ultimaker | no independently addressable file | `DirectPrints` | atomic HTTP |
| PrusaLink branch | full USB-backed HTTP store | `StoredPrints` | HTTP |
| OctoPrint branch | full `/api/files` store | `StoredPrints` | HTTP |
| Qidi branch | Moonraker store and starter | `StoredPrints` | WebSocket RPC |

Bambu changing between cloud and LAN explicitly replaces its print capability
during reconfiguration. There is no per-operation `route()` switch hidden inside
one file class.

Do not retain Bambu's size-only file deduplication: equal name and length do not
prove equal content. Always overwrite explicitly or compare a verified digest.
Do not treat an arbitrary HTTP ETag as MD5 unless that protocol guarantees it.

## Camera

An integration exposes one immutable `CameraSource`: an async URI resolver and,
only when the device requires it, a typed activation operation.

`CameraService` owns resolution tasks, demand admission, cache, retries, workers
and shutdown. This removes the repeated resolved-URI, probe-in-flight, task and
cancel flags in brand printers. Standard MJPEG protocols are registered once;
integrations contribute only genuinely custom openers.

Do not create public capability types for materials, peripherals and protocol
helpers yet. Keep them in the adapter until two real implementations share the
same semantics. Capability objects can become the next class explosion if every
method gets one.

## Integration declaration

Shrink `IntegrationSpec` to static data and one adapter construction seam:

```python
@dataclass(frozen=True, slots=True)
class PrinterIntegration:
    id: IntegrationId
    product: ProductMetadata
    config_type: Type[PrinterConfig]
    create_adapter: AdapterFactory
    discovery: Optional[DiscoverySpec] = None
    onboarding: Optional[Onboarding] = None
    presentation: Optional[Presentation] = None
    accounts: Optional[Type[AccountProvider]] = None
    tasks: Tuple[TaskSpec, ...] = ()
```

Use an ordinary immutable mapping keyed by `IntegrationId`. Delete the
`SpecRegistry` projection methods.

Normal discovery comes from one `DiscoverySpec`. MQTT diagnostics come from the
adapter's `connection()`. Camera comes from the adapter. Runtime capabilities are
the source of truth; do not maintain a second hand-written feature registry that
can drift.

This removes most current private `_create_*`, `_build_*`, `discover`, camera,
MQTT URL and verification wrappers.

## Discovery and onboarding

Each integration has one discovery plan. `Runtime` owns the discovery service;
there is no process-global active service.

- Bind `IntegrationId` once at composition.
- Use one canonical discovered-printer value and one event carrying that ID.
- Merge the current record/device/result wrappers.
- Promote model, hardware ID, serial and LAN mode out of `extra` dictionaries.
- Use enums for closed transport and purpose values.
- Delete brand-specific discovery event classes and the repeated
  `discover(service, timeout)` wrappers.

The current flow engine is also too generic: 1,289 lines, many tiny outcome
classes, sync-or-async callback unions and untyped state. Keep onboarding direct:
async callbacks only, one discriminated result, typed session state and explicit
prompt construction. Do not hide flow behavior in decorators or mutate frozen
prompts with `object.__setattr__`.

## Events

Required runtime flow is direct calls. Keep a general event bus only for genuine
one-to-many observers.

An observer cannot rewrite the arguments later observers receive by returning a
value. Delete unused uniqueness/lifetime/priority modes instead of preserving
test-only features. A small broadcast bus is enough.

## PaperCut and other extensions

PaperCut is not a printer type. Do not merge the current `papercut_integration`
generic framework into the printer spec.

Use one instance-owned extension with `start`, `close`, `status`, typed settings
and bound actions. The extension owns its service and health. `Action` stores its
bound callable; there is no string action switch. Reuse the existing editable
field and status domains instead of adding parallel strings and models.

The app owns a plain extension mapping. Delete context dictionaries such as
`ctx.integrations["papercut"]`, duplicated health maps and service lookup helpers.

## Delete, do not wrap

The mergeable result removes these paths after all callers move:

- `Client -> PrinterClient -> BrandPrinter` inheritance;
- `DeviceDriver`, `LeaseDriver`, `MqttDriver`, `WsDriver` and the tick-based
  `ensure_started` sweep;
- `FileTransfer`, `PrintFileDriver`, `PreparationKind`, `StartDisposition`,
  unsupported route stubs and filename-based start history;
- `JobEdge`, raw-payload hooks and `apply_status` flag arguments;
- public camera controller/handle/backend wiring from the integration API;
- `IntegrationSpec` optional callback bag and `SpecRegistry` projections;
- `ClientContext` optional service-locator shape;
- sender injection after construction;
- compatibility aliases and type-shape fallbacks used only by tests.

Pooling, transport adapters, model reflection, Pydantic boundary models and
substantial pure protocol code can stay where they have a real job.

## Migration order

1. Record the baseline: production LoC, classes, public extension methods,
   mutable fields, tasks, threads, file descriptors, loop lag and throughput.
   Replace tests that pin source/private shape with behavior tests.
2. Align the library revision used by the client. New Qidi, OctoPrint, PrusaLink
   and other integration branches target the new API instead of copying the old
   one.
3. Build `ConnectionSpec`, `DeviceIO` and `PersistentConnection`. Prove perpetual
   retry, fresh sessions, generation fencing, reconfiguration and teardown with
   fake transports and Paho. Move Bambu first.
4. Build `Runtime`, `CloudSession` and final `PrinterRuntime` against a fake
   adapter. Characterize current external cloud messages and ordering before the
   switch.
5. Build `PrintLifecycle`, file capabilities and `CameraService`. Complete the
   Bambu vertical slice, including connection loss during a print and file start.
6. Migrate Moonraker and Snapmaker; then Anycubic, Creality and Elegoo; then Duet
   and Ultimaker. Rebase Qidi, OctoPrint and PrusaLink directly onto the result.
7. Switch the app composition root only after every shipped printer uses
   `PrinterRuntime`. Delete the old client, scheduler/manager, driver, transfer,
   registry and camera authoring paths in that same cut. Ship one runtime path.
8. Rebase PaperCut business logic onto the small extension seam and discard its
   generic spec/registry/context framework.
9. Run the local P1 soak, deterministic fault suite, 100-printer replay/farm test
   and production canary. Compare every measurement to the baseline.

Temporary development commits may contain both implementations. The mergeable
result may not.

## Behavior gates

Connection tests use fake time and transports and cover:

- initial refusal, EOF, keepalive failure and backend thread exit;
- repeated disconnect/reconnect without ending the supervisor;
- auth/config failure followed by config replacement;
- stale callbacks from an old generation;
- adapter/parser failure;
- stop during connect and stop during backoff;
- exactly one live session and exactly-once close;
- no device edge changing cloud membership or deciding physical job state.

Print/file tests cover:

- list, stat, upload, delete and explicit overwrite for full stores;
- stored, URL and direct print paths;
- staging only on a stored-file path;
- accepted start plus lost report never sending stop;
- reconnect and a running-job observation resolving one receipt once;
- definitive rejection without invented cancellation;
- no retry after an ambiguous start send;
- progress from a worker thread appearing only on the printer loop;
- superseded work never mutating the replacement operation;
- every task, session, temporary file and worker closing once.

The 30-minute local and farm suites are acceptance tests after deterministic fault
coverage, not substitutes for it.

## Bad tests to remove or replace

Keep tests of wire bytes, generated artifacts, public behavior and real package
boundaries. Remove tests whose only purpose is to freeze today's implementation.

Known targets:

- ws-client `test_discovery_has_no_process_global_service_accessor`: replace the
  filename/string scan with two runtime instances proving no shared state.
- ws-client brand-token and one-file import AST tests: replace with one repo-wide
  import-linter graph.
- ws-client polyfill filename test: delete it.
- ws-client lease-driver, camera-controller and scheduler tests that mutate or
  wait on private flags: rewrite around public lifecycle and fake time.
- client `test_cli_modules_do_not_import_config_manager`: delete it; adjacent
  behavior already proves the boundary.
- client exact Moonraker/Snapmaker `__bases__`, constructor signature and Duet
  signature tests: delete them; retain construction and ownership behavior.
- client command-root base-class assertion: delete it; retain serialization and
  direct dispatch behavior.
- client duplicate registry brand sets/order and monkeypatched constructor
  internals: consolidate around the public registry/factory behavior.
- manual-model tests that inspect phase IDs or `__annotations__`: run the flow and
  assert that the selected model is persisted.
- release-build tests that grep `.iss`, workflow, Dockerfile, CMake, spec or Python
  source for preferred tokens: execute the tool or parse structured output where
  the contract matters.

Tests may read files produced by the unit under test. That is behavior, not source
inspection.

## Review scoreboard

For each slice, report before and after:

- production LoC;
- classes and public methods;
- durable instance attributes;
- duplicated lifecycle/retry loops;
- tasks, threads and file descriptors per printer;
- loop lag and message throughput at farm scale;
- tests deleted, replaced and added.

The scoreboard is a review aid. Never turn it into tests that search source.
