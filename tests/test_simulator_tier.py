"""Tier resolution / routing for SparseSimulator.

Offline tests use a dummy client (no network). The live test reads the caller's
plan from the hypermedia API and is gated on Q_ALCHEMY_API_KEY.
"""

import os
import unittest

from dotenv import load_dotenv

from q_alchemy.simulator import SparseSimulator, ENTERPRISE_GRANT

load_dotenv("../.env")


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
        sim = SparseSimulator(client=object(), tier="auto")  # dummy client -> grant fetch fails -> []
        self.assertEqual(sim.user_grants(), [])
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
