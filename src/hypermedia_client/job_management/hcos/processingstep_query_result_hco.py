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
from typing import Self

import httpx

from hypermedia_client.core.hco.hco_base import Hco, Property
from hypermedia_client.core.hco.link_hco import LinkHco
from hypermedia_client.job_management.hcos.processing_step_hco import ProcessingStepHco
from hypermedia_client.job_management.known_relations import Relations
from hypermedia_client.job_management.model.sirenentities import (
    ProcessingStepQueryResultEntity,
    ProcessingStepEntity
)


class ProcessingStepQueryResultPaginationLink(LinkHco):
    def navigate(self) -> 'ProcessingStepQueryResultHco':
        return ProcessingStepQueryResultHco.from_entity(self._navigate_internal(ProcessingStepQueryResultEntity),
                                                        self._client)


class ProcessingStepQueryResultLink(LinkHco):
    def navigate(self) -> 'ProcessingStepQueryResultHco':
        return ProcessingStepQueryResultHco.from_entity(self._navigate_internal(ProcessingStepQueryResultEntity),
                                                        self._client)


class ProcessingStepQueryResultHco(Hco[ProcessingStepQueryResultEntity]):
    total_entities: int = Property()
    current_entities_count: int = Property()
    processing_steps: list[ProcessingStepHco]

    self_link: ProcessingStepQueryResultLink
    all_link: ProcessingStepQueryResultPaginationLink | None
    first_link: ProcessingStepQueryResultPaginationLink | None
    last_link: ProcessingStepQueryResultPaginationLink | None
    next_link: ProcessingStepQueryResultPaginationLink | None
    previous_link: ProcessingStepQueryResultPaginationLink | None

    @classmethod
    def from_entity(cls, entity: ProcessingStepQueryResultEntity, client: httpx.Client) -> Self:
        instance = cls(client, entity)

        Hco.check_classes(instance._entity.class_, ["ProcessingStepQueryResult"])

        # pagination links
        instance.self_link = ProcessingStepQueryResultLink.from_entity(
            instance._client, instance._entity, Relations.SELF)
        instance.all_link = ProcessingStepQueryResultPaginationLink.from_entity_optional(
            instance._client, instance._entity, Relations.ALL)
        instance.first_link = ProcessingStepQueryResultPaginationLink.from_entity_optional(
            instance._client, instance._entity, Relations.FIRST)
        instance.last_link = ProcessingStepQueryResultPaginationLink.from_entity_optional(
            instance._client, instance._entity, Relations.LAST)
        instance.next_link = ProcessingStepQueryResultPaginationLink.from_entity_optional(
            instance._client, instance._entity, Relations.NEXT)
        instance.previous_link = ProcessingStepQueryResultPaginationLink.from_entity_optional(
            instance._client, instance._entity, Relations.PREVIOUS)

        instance._extract_processing_steps()

        return instance

    def _extract_processing_steps(self):
        self.processing_steps = []
        processing_steps = self._entity.find_all_entities_with_relation(Relations.ITEM, ProcessingStepEntity)
        for processing_step in processing_steps:
            processing_step_hco: ProcessingStepHco = ProcessingStepHco.from_entity(processing_step, self._client)
            self.processing_steps.append(processing_step_hco)
