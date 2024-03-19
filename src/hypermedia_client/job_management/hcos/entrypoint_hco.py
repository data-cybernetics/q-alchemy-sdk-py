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
from enum import StrEnum
from typing import Self

import httpx

from hypermedia_client.core.hco.hco_base import Hco
from hypermedia_client.core.hco.link_hco import LinkHco
from hypermedia_client.job_management.hcos.info_hco import InfoLink
from hypermedia_client.job_management.hcos.jobsroot_hco import JobsRootLink
from hypermedia_client.job_management.hcos.processingsteproot_hco import ProcessingStepsRootLink
from hypermedia_client.job_management.hcos.workdataroot_hco import WorkDataRootLink
from hypermedia_client.job_management.known_relations import Relations
from hypermedia_client.job_management.model.sirenentities import EntryPointEntity


class EntryPointLink(LinkHco):
    def navigate(self) -> 'EntryPointHco':
        return EntryPointHco.from_entity(self._navigate_internal(EntryPointEntity), self._client)


class EntrypointRelations(StrEnum):
    JOBS_ROOT = "JobsRoot"
    WORKDATA_ROOT = "WorkDataRoot"
    PROCESSINGSTEPS_ROOT = "ProcessingStepsRoot"
    INFO = "Info"
    ADMIN = "Admin"


class EntryPointHco(Hco[EntryPointEntity]):
    self_link: EntryPointLink
    job_root_link: JobsRootLink
    work_data_root_link: WorkDataRootLink
    processing_step_root_link: ProcessingStepsRootLink
    info_link: InfoLink

    admin_link: LinkHco | None

    @classmethod
    def from_entity(cls, entity: EntryPointEntity, client: httpx.Client) -> Self:
        instance = cls(client, entity)
        Hco.check_classes(instance._entity.class_, ["EntryPoint"])

        instance.self_link = EntryPointLink.from_entity(
            instance._client, instance._entity, Relations.SELF)
        instance.info_link = InfoLink.from_entity(
            instance._client, instance._entity, EntrypointRelations.INFO)
        instance.job_root_link = JobsRootLink.from_entity(
            instance._client, instance._entity, EntrypointRelations.JOBS_ROOT)
        instance.work_data_root_link = WorkDataRootLink.from_entity(
            instance._client, instance._entity, EntrypointRelations.WORKDATA_ROOT)
        instance.processing_step_root_link = ProcessingStepsRootLink.from_entity(
            instance._client, instance._entity, EntrypointRelations.PROCESSINGSTEPS_ROOT)

        instance.admin_link = LinkHco.from_entity_optional(
            instance._client, instance._entity, EntrypointRelations.ADMIN)

        return instance
