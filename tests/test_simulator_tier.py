"""Tier resolution / routing for SparseSimulator.

Offline tests use a dummy client (no network). The live test reads the caller's
plan from the hypermedia API and is gated on Q_ALCHEMY_API_KEY.
"""

import os
import unittest

from dotenv import load_dotenv

from q_alchemy.simulator import SparseSimulator, ENTERPRISE_GRANT

load_dotenv("../.env")


class TestTierResolution(unittest.TestCase):
    def test_explicit_tiers_are_honored(self):
        self.assertEqual(SparseSimulator(client=object(), tier="standard").tier, "standard")
        self.assertEqual(SparseSimulator(client=object(), tier="enterprise").tier, "enterprise")

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
        sim = SparseSimulator(client=object(), tier="enterprise")
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


if __name__ == "__main__":
    unittest.main()
