from ipaddress import IPv4Address

from simplyprint_ws_client.contrib.onboarding import (
    ManualAddressStep,
    ModelChoiceCatalog,
)


def test_model_choice_catalog_uses_explicit_extractors():
    class Item:
        def __init__(self, code, title, image=None, enabled=True):
            self.code = code
            self.title = title
            self.image = image
            self.enabled = enabled

    catalog = ModelChoiceCatalog.from_items(
        [Item("a", "Alpha", "/alpha.webp"), Item("b", "Beta", enabled=False)],
        value=lambda item: item.code,
        label=lambda item: item.title,
        image=lambda item: item.image,
        include=lambda item: item.enabled,
    )

    assert [choice.value for choice in catalog.choices] == ["a", ""]
    assert catalog.choices[0].image == "/alpha.webp"


def test_model_choice_catalog_builds_identify_step_with_configurable_state_key():
    step = ModelChoiceCatalog.from_rows(
        [("model-a", "Model A", "/a.webp")]
    ).identify_step(state_key="model")

    schema = step._input_model.model_json_schema()
    options = schema["properties"]["model"]["ui"]["options"]

    assert [option["value"] for option in options] == ["model-a", ""]
    assert step._include({}) is True
    assert step._include({"model": "model-a"}) is False


def test_manual_address_step_can_collect_dns_or_ip_with_custom_key():
    step = ManualAddressStep(
        state_key="address",
        value_type=str | IPv4Address,
        placeholder="printer.local",
    ).build()

    parsed = step._input_model(address="printer.local")

    assert parsed.address == "printer.local"
    assert step._input_values({"address": "printer.local"}) == {
        "address": "printer.local"
    }
