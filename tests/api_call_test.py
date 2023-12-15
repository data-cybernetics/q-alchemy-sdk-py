# Copyright 2022-2023 data cybernetics ssc GmbH.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import time
import unittest

import numpy as np
import qiskit

from q_alchemy import Client


class TestApiIntegration(unittest.TestCase):

    def test_integration(self):

        # The API key is read from the environment variable Q_ALCHEMY_API_KEY
        client = Client()
        root = client.get_jobs_root()

        # Let us create a new and empty job, but rename it right away!
        job = root.create_job()
        job.rename("Your first Test-Job!")

        # First, let us configure the job with the job's config
        # for that we summon up the config resource!
        config = job.get_config()

        # Creating the quantum state
        qb = 12
        state = np.load(f"./data/test_baa_state.{qb}.1.npy")

        # Upload the state vector now:
        state_vector = job.get_state_vector()
        state_vector.upload_vector(state)

        # Set the fidelity loss and configure the job with tags to find it later
        fid_loss = 0.21
        config.set_config(fid_loss, [f"{qb}qb", str(fid_loss)])

        # Start the Job
        job.update().schedule()

        # Wait for the result
        while not job.update().is_success and not job.has_error:
            print("Waiting for completion...")
            time.sleep(1)

        # Now get the best result and plot it as given
        final_node = job.get_result().get_best_node()
        qc = final_node.to_circuit()
        print(
            qiskit.transpile(qc, basis_gates=['u1', 'u2', 'u3', 'cx'], optimization_level=3).draw(output="text", fold=-1)
        )
        self.assertEquals(final_node.total_saved_cnots, 2307)

