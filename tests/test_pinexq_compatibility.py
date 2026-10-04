"""Exercise the real PineXQ models and version check without a live server."""
import importlib
import os
import subprocess
import sys
import warnings
from types import SimpleNamespace

import pytest
from pinexq.client.job_management.model.sirenentities import InfoEntity, InputDataSlotEntity


def test_api_10_fields_are_recognized_without_extra_property_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        info = InfoEntity.model_validate({
            "properties": {"ApiVersion": "10.1.0", "TenantId": "test-tenant"},
        })
        slot = InputDataSlotEntity.model_validate({
            "properties": {"Name": "experiment", "IsMaterialized": True},
        })
    assert info.properties.tenant_id == "test-tenant"
    assert slot.properties.is_materialized is True
    assert not info.properties.model_extra
    assert not slot.properties.model_extra


@pytest.mark.parametrize("mismatched_component", [0, 1], ids=["major", "minor"])
def test_client_accepts_matching_protocol_and_warns_for_mismatch(monkeypatch, mismatched_component):
    module = importlib.import_module("pinexq.client.job_management.enterjma")
    # CI installs the newest supported client, whose protocol minor version may
    # advance. Test its real version-check behavior rather than pinning the mock
    # server to the protocol of pinexq-client 2.0.0.
    protocol = list(importlib.import_module("pinexq.client.job_management").__jma_version__)
    info = SimpleNamespace(api_version=".".join(map(str, protocol)))
    entry = SimpleNamespace(info_link=SimpleNamespace(navigate=lambda: info))
    monkeypatch.setattr(module, "enter_api", lambda *args: entry)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        assert module.enter_jma(object()) is entry
        # The client compares only major/minor; a patch change is compatible.
        patch_version = protocol.copy()
        patch_version[2] += 1
        info.api_version = ".".join(map(str, patch_version))
        assert module.enter_jma(object()) is entry
    different_protocol = protocol.copy()
    different_protocol[mismatched_component] += 1
    info.api_version = ".".join(map(str, different_protocol))
    with pytest.warns(UserWarning, match="API-Version mismatch"):
        assert module.enter_jma(object()) is entry


def test_import_does_not_suppress_pinexq_warnings():
    # Test in a fresh process so previously imported SDK modules cannot mask an
    # import-time filter. The old opt-out setting must no longer be necessary.
    env = os.environ.copy()
    env.pop("Q_ALCHEMY_API_VERSION_WARNING", None)
    subprocess.run([sys.executable, "-c", '''
import warnings
import q_alchemy
with warnings.catch_warnings(record=True) as caught:
    warnings.warn("Version mismatch between 'pinexq_client' and server", UserWarning)
    warnings.warn("API-Version mismatch between 'pinexq_client' and server", UserWarning)
    warnings.warn("Entity with extra properties received!", UserWarning)
    assert len(caught) == 3, [str(w.message) for w in caught]
'''], env=env, check=True)
