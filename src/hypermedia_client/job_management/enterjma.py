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

import logging

from hypermedia_client.core import Entity
from hypermedia_client.core.enterapi import enter_api
from hypermedia_client.core.hco.hco_base import Hco
from hypermedia_client.job_management.hcos.entrypoint_hco import EntryPointHco
from hypermedia_client.job_management.model.sirenentities import EntryPointEntity
import hypermedia_client.job_management

LOG = logging.getLogger(__name__)

THco = TypeVar("THco", bound=Hco)


def _version_match(ver1: list[int], ver2: list[int]) -> bool:
    return all([v1 == v2 for v1, v2 in zip(ver1, ver2)])


def enter_jma(
    client: httpx.Client,
    entrypoint_hco_type: Type[THco] = EntryPointHco,
    entrypoint_entity_type: Type[Entity] = EntryPointEntity,
    entrypoint: str = "api/EntryPoint",
) -> EntryPointHco:
    entry_point_hco = enter_api(client, entrypoint_hco_type, entrypoint_entity_type, entrypoint)

    info = entry_point_hco.info_link.navigate()

    # Check for matching protocol versions
    client_version = hypermedia_client.job_management.__jma_version__
    jma_version = [int(i) for i in str.split(info.api_version, '.')]
    if not _version_match(jma_version, client_version):
        LOG.warning(
            f"Version mismatch between 'hypermedia_client' (v{'.'.join(map(str ,client_version))}) "
            f"and 'JobManagementAPI' (v{'.'.join(map(str, jma_version))})! "
        )

    return entry_point_hco
