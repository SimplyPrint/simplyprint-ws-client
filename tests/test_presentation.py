from simplyprint_ws_client.contrib.presentation import default_printer_presentation


def test_default_printer_presentation_does_not_infer_config_fields():
    presentation = default_printer_presentation("/img/printer.svg")

    assert presentation.image_url == "/img/printer.svg"
    assert presentation.connection is None
