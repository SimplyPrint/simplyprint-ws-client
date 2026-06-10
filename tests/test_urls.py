from simplyprint_ws_client.cloud.api.url_builder import (
    SimplyPrintURL,
    SimplyPrintBackend,
)


def test_url_builder():
    urls = SimplyPrintURL()

    urls.set_backend(SimplyPrintBackend.PRODUCTION)

    assert str(urls.main_url) == "https://simplyprint.io"
    assert str(urls.api_url) == "https://api.simplyprint.io"
    assert str(urls.ws_url) == "wss://ws.simplyprint.io/0.2"

    urls.set_backend(SimplyPrintBackend.TESTING)

    assert str(urls.main_url) == "https://test.simplyprint.io"
    assert str(urls.api_url) == "https://testapi.simplyprint.io"
    assert str(urls.ws_url) == "wss://testws3.simplyprint.io/0.2"

    urls.set_backend(SimplyPrintBackend.STAGING)

    assert str(urls.main_url) == "https://staging.simplyprint.io"
    assert str(urls.api_url) == "https://apistaging.simplyprint.io"
    assert str(urls.ws_url) == "wss://wsstaging.simplyprint.io/0.2"

    # modifies same static variable
    assert urls._active_backend == SimplyPrintURL._active_backend
