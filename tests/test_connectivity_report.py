from simplyprint_ws_client.common.debug.connectivity import ConnectivityReport


def test_generate_default_report():
    ConnectivityReport.generate_default()
