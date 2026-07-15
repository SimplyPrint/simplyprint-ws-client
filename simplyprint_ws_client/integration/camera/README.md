# Camera composition

Each `PrinterClient` owns one final `CameraController` at `client.camera`.
Printer-facing protocol routes remain on `PrinterClient`, so a brand can prepare
its hardware in `on_stream_on()` and then delegate to `super()`. The controller
owns the camera URI and status, its `CameraHandle`, bounded demand queue, frame
delivery, and teardown.

Set or clear a source through `client.camera.uri`; inspect setup state through
`client.camera.set_uri(uri)`. A configured URI is submitted to the shared `CameraPool`,
which chooses a protocol and returns a handle. The controller serializes frame
reads and pauses or stops that handle as cloud demand and client lifecycle change.
