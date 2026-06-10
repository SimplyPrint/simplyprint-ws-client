# Camera process pool

- Define a camera protocol
- Define a camera (config + protocol)
- Submit to pool
- Receive frames
- Pause/Resume

Async thread request frame from camera X

Camera X has a derived protocol and configuration

The async thread submits the configuration to the pool and gets a future bound to
its event loop that will be fulfilled once the pool can fulfil the frame request.
