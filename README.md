# SimplyPrint Websocket Client

> A python package to simplify the use of the SimplyPrint websocket API to easily integrate with
> the <https://simplyprint.io> platform.

This package version v1.0.0 is under major development and is not yet ready for use. API is not stable and will change.

## Usage

See [docs/index.md](docs/index.md) to get started.

## Dependencies

The open printer SDK, [`simplyprint-sdk`](https://pypi.org/p/simplyprint-sdk/),
is a separate package installed from PyPI (`simplyprint-sdk[pydantic]>=0.1,<0.2`).
The framework re-exports its merge-and-diff models and G-code parser. CI and
users always resolve it from PyPI, so the committed `pyproject.toml` and
`uv.lock` must not point at a local checkout.

### Developing against a local SDK

For a quick test against an SDK checkout next to this one, install it over the
locked release and run without re-syncing:

```sh
uv sync
uv pip install -e ../simplyprint-sdk
uv run --no-sync pytest
```

A plain `uv run` or `uv sync` puts the PyPI release back.

To keep the override longer, uncomment the `[tool.uv.sources]` block in
`pyproject.toml` (adjust the path) and run `uv lock`. Do not commit that change
or the `uv.lock` it produces. Release the SDK first, then bump the pin here
(see the client's `docs/release-sdk.md` for the release order).

## TODO

- [ ] Add usage documentation
- [ ] Add examples
- [ ] Add tests for event system and reactivity
- [ ] Unify event bus api with printer events.
- [ ] Expand client to a plugin/module based system
- [ ] Update config backend (Only pydantic) + custom settings provider + more settings
- [ ] Provide nicer interface for sending especially job_info (e.i. job management logic) so we can bundle important
  steps together for consistency.
- [ ] Improve `tick` hook with something like the OctoPrint-SimplyPrint FlexTimer solution to avoid having to manually
  keep track of time
