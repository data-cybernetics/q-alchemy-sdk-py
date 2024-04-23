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
import logging
import random

from hypermedia_client.core.exceptions import RateLimitError

logger = logging.getLogger(__name__)


# Custom wait generator to respect 'Retry-After' header
def wait_retry_generator(retry_state):
    exception = retry_state.outcome.exception()
    if isinstance(exception, RateLimitError):
        if exception.retry_after_sec is not None:
            retry_after_sec = exception.retry_after_sec + random.random() / 2
            print(f"Rate limited. Retrying after {retry_after_sec} seconds.")
            return retry_after_sec
    else:
        return 0.5
