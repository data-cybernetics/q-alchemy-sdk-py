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
from typing import Callable


class PollingException(Exception):
    pass


def wait_until(
        condition: Callable,
        polling_interval_ms: int = 200,
        timeout_ms: int = 5000,
        timeout_message: str | None = None,
        error_condition: Callable | None = None,
        error_condition_message: str | None = None,

) -> None:
    start = time.time()
    timeout = start + timeout_ms / 1000
    while True:
        now = time.time()
        next_due = now + polling_interval_ms / 1000
        success = condition()
        if success:
            return

        if error_condition:
            error = error_condition()
            if error:
                raise PollingException(
                    f"{f': {error_condition_message}' if error_condition_message else 'Error condition meet while waiting'}")

        if (timeout > 0) and (now > timeout):
            raise TimeoutError(
                f"{f': {timeout_message}' if timeout_message else f'Timeout while waiting. Waited: {timeout_ms}ms'}")

        time.sleep(next_due - now)
