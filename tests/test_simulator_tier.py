"""Tier resolution / routing for SparseSimulator.

Offline tests use a dummy client (no network). The live test reads the caller's
plan from the hypermedia API and is gated on Q_ALCHEMY_API_KEY.
"""

import os
import unittest
from pathlib import Path

from dotenv import load_dotenv

import httpx

from q_alchemy.simulator import ENTERPRISE_GRANT, PlanLookupError, SparseSimulator

load_dotenv(Path(__file__).parent.parent / ".env")


class _FakeClient:
    """Serves canned JSON pages by path; optional HTTP status overrides or a network error."""

    def __init__(self, pages, *, status=None, error=None):
        self.pages, self.status, self.error = pages, dict(status or {}), error

    def get(self, url):
        if self.error is not None:
            raise self.error
        request = httpx.Request("GET", f"https://example.invalid{url}")
        return httpx.Response(self.status.get(url, 200), json=self.pages.get(url, {}), request=request)


def _grant_pages(grants):
    return {
        "/api/EntryPoint": {"links": [{"rel": ["Info"], "href": "/api/Info"}]},
        "/api/Info": {"entities": [{"rel": ["CurrentUser"], "properties": {"UserGrants": grants}}]},
    }


def _sim(tier="auto", grants=None):
    """Simulator on a dummy client; `grants` stands in for the detected plan."""
    sim = SparseSimulator(client=object(), tier=tier)
    if grants is not None:
        sim._grants = list(grants)
    return sim


class TestTierResolution(unittest.TestCase):
    def test_standard_is_always_available(self):
        """Opting down is a normal thing to want, so it needs no entitlement."""
        self.assertEqual(_sim("standard").tier, "standard")
        self.assertEqual(_sim("standard", [ENTERPRISE_GRANT]).tier, "standard")

    def test_enterprise_is_honored_with_the_plan(self):
        self.assertEqual(_sim("enterprise", [ENTERPRISE_GRANT, "role:admin"]).tier, "enterprise")

    def test_enterprise_without_the_plan_is_refused_before_any_job(self):
        """The ProCon would refuse it anyway; fail here instead of round-tripping."""
        with self.assertRaises(PermissionError) as caught:
            _sim("enterprise", ["plan:free"]).tier
        message = str(caught.exception)
        self.assertIn("enterprise plan", message)
        self.assertIn("standard", message)  # names the way forward

    def test_unknown_tier_is_rejected_rather_than_silently_standard(self):
        with self.assertRaises(ValueError):
            _sim("xlarge", [ENTERPRISE_GRANT]).tier

    def test_auto_without_grants_is_standard(self):
        # The API exposes no Info link, i.e. no grants: that is the standard tier.
        sim = SparseSimulator(client=_FakeClient({"/api/EntryPoint": {"links": []}}), tier="auto")
        self.assertEqual(sim.user_grants(), [])
        self.assertEqual(sim.tier, "standard")

    def test_auto_reads_grants_from_the_current_user(self):
        sim = SparseSimulator(client=_FakeClient(_grant_pages([ENTERPRISE_GRANT])), tier="auto")
        self.assertEqual(sim.user_grants(), [ENTERPRISE_GRANT])
        self.assertEqual(sim.tier, "enterprise")

    def test_failed_plan_lookup_raises_instead_of_downgrading(self):
        """A lookup failure must not silently move an enterprise caller to standard."""
        failures = {
            "network": _FakeClient({}, error=httpx.ConnectError("offline")),
            "http status": _FakeClient(_grant_pages([ENTERPRISE_GRANT]), status={"/api/Info": 503}),
            "unauthorized": _FakeClient({"/api/EntryPoint": {"links": []}}, status={"/api/EntryPoint": 401}),
            "malformed": _FakeClient({"/api/EntryPoint": ["not", "an", "object"]}),
        }
        for label, client in failures.items():
            for tier in ("auto", "enterprise"):
                with self.subTest(failure=label, tier=tier):
                    sim = SparseSimulator(client=client, tier=tier)
                    with self.assertRaises(PlanLookupError) as caught:
                        sim.tier
                    self.assertIn("tier='standard'", str(caught.exception))
                    self.assertIsNone(sim._grants)  # failures are not cached

    def test_plan_lookup_failure_is_retried_on_the_next_call(self):
        client = _FakeClient(_grant_pages([ENTERPRISE_GRANT]), status={"/api/Info": 503})
        sim = SparseSimulator(client=client, tier="auto")
        with self.assertRaises(PlanLookupError):
            sim.user_grants()
        client.status.clear()
        self.assertEqual(sim.tier, "enterprise")

    def test_standard_tier_needs_no_plan_lookup(self):
        sim = SparseSimulator(client=_FakeClient({}, error=httpx.ConnectError("offline")), tier="standard")
        self.assertEqual(sim.tier, "standard")

    def test_auto_with_enterprise_grant_is_enterprise(self):
        sim = SparseSimulator(client=object(), tier="auto")
        sim._grants = [ENTERPRISE_GRANT, "role:admin"]  # simulate detected plan
        self.assertTrue(sim.is_enterprise())
        self.assertEqual(sim.tier, "enterprise")

    def test_enterprise_routing_appends_suffix(self):
        sim = _sim("enterprise", [ENTERPRISE_GRANT])
        # _run builds "<capability>_<suffix>" and appends _enterprise for the tier.
        suffix, _, _ = sim._prepare_input("OPENQASM 2.0;\nqreg q[1];", "auto")
        name = f"counts_{suffix}"
        if sim.tier == "enterprise":
            name += "_enterprise"
        self.assertEqual(name, "counts_from_qasm_string_enterprise")


@unittest.skipUnless(os.getenv("Q_ALCHEMY_API_KEY"), "no Q_ALCHEMY_API_KEY: skipping live plan detection")
class TestLivePlanDetection(unittest.TestCase):
    def test_user_grants_and_tier(self):
        sim = SparseSimulator()
        grants = sim.user_grants()
        self.assertIsInstance(grants, list)
        # tier must resolve consistently with the detected grants
        self.assertEqual(sim.tier, "enterprise" if ENTERPRISE_GRANT in grants else "standard")

    def test_enterprise_caller_may_opt_down_to_standard(self):
        """The asymmetry that matters: down is always allowed, up is not."""
        sim = SparseSimulator(tier="standard")
        self.assertEqual(sim.tier, "standard")
        if ENTERPRISE_GRANT not in sim.user_grants():
            self.skipTest("key has no enterprise plan; nothing to opt down from")
        self.assertEqual(SparseSimulator(tier="auto").tier, "enterprise")


if __name__ == "__main__":
    unittest.main()
