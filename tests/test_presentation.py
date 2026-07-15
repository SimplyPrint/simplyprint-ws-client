from simplyprint_ws_client import PrinterConfig
from simplyprint_ws_client.integration.presentation import default_printer_presentation


def test_default_printer_presentation_does_not_infer_config_fields():
    config = PrinterConfig.get_new()
    config.custom_webcam_url = "http://camera.local/stream"

    presentation = default_printer_presentation("/img/printer.svg", config)

    assert presentation.image_url == "/img/printer.svg"
    assert presentation.connection is None
    assert [field.key for field in presentation.editable_fields] == ["webcam_url"]

    webcam = presentation.editable_fields[0]
    assert webcam.value == "http://camera.local/stream"
    webcam.write("")
    assert config.custom_webcam_url is None
    webcam.write("http://camera.local/new")
    assert config.custom_webcam_url == "http://camera.local/new"
