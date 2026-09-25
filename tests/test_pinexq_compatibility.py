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
            "properties": {"ApiVersion": "10.0.0", "TenantId": "test-tenant"},
        })
        slot = InputDataSlotEntity.model_validate({
            "properties": {"Name": "experiment", "IsMaterialized": True},
        })
    assert info.properties.tenant_id == "test-tenant"
    assert slot.properties.is_materialized is True
    assert not info.properties.model_extra
    assert not slot.properties.model_extra


def test_client_accepts_api_10_and_warns_for_different_protocol(monkeypatch):
    module = importlib.import_module("pinexq.client.job_management.enterjma")
    info = SimpleNamespace(api_version="10.0.0")
    entry = SimpleNamespace(info_link=SimpleNamespace(navigate=lambda: info))
    monkeypatch.setattr(module, "enter_api", lambda *args: entry)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        assert module.enter_jma(object()) is entry
    info.api_version = "11.0.0"
    with pytest.warns(UserWarning, match="API-Version mismatch"):
        module.enter_jma(object())


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
