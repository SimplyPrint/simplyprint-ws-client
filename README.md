# SimplyPrint Websocket Client

> A python package to simplify the use of the SimplyPrint websocket API to easily integrate with
> the <https://simplyprint.io> platform.

This package version v1.0.0 is under major development and is not yet ready for use. API is not stable and will change.

## Usage

See [docs/index.md](docs/index.md) to get started.

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
