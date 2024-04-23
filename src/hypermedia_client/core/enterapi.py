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
from typing import TypeVar, Type

import httpx
from tenacity import retry, retry_if_exception_type, wait_fixed, stop_after_attempt

from hypermedia_client.core import Entity, RateLimitError
from hypermedia_client.core.hco.hco_base import Hco
from hypermedia_client.core.resilience_handling import wait_retry_generator

THco = TypeVar('THco', bound=Hco)


@retry(retry=retry_if_exception_type(RateLimitError), wait=wait_retry_generator, stop=stop_after_attempt(4))
def enter_api(client: httpx.Client, entrypoint_hco_type: Type[THco], entrypoint_entity_type: Type[Entity] = Entity,
              entrypoint: str = "api/EntryPoint") -> THco:
    entry_point_response = client.get(url=entrypoint)
    if entry_point_response.status_code == 429:
        raise RateLimitError(entry_point_response.headers)
    entry_point_response.raise_for_status()
    entrypoint_entity = entrypoint_entity_type.model_validate_json(entry_point_response.read())

    return entrypoint_hco_type.from_entity(entrypoint_entity, client)
