"""Network facts are read once per TTL, not once per printer built."""

from simplyprint_ws_client.common.hardware import physical_machine as pm


def test_a_fact_is_reused_within_its_ttl_and_read_again_after(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(pm.time, "monotonic", lambda: now[0])
    reads = []

    @pm.ttl_cached(30.0)
    def fact():
        reads.append(1)
        return len(reads)

    assert fact() == 1 and fact() == 1
    now[0] += 29.9
    assert fact() == 1
    now[0] += 0.2
    assert fact() == 2
    fact.cache_clear()
    assert fact() == 3
