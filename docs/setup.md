To get started first install the client library:

```bash
pip install "simplyprint-ws-client==1.0.1rc15"
```

Import the authoring types your integration uses explicitly:

```python
from simplyprint_ws_client import (
    ClientContext,
    IntegrationSpec,
    PrinterClient,
    PrinterConfig,
)
```

The package root contains the supported authoring API. Protocol/runtime internals
are imported from their owning modules; there is no wildcard compatibility
fallback.
