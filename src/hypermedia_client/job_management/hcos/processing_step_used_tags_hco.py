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
from typing import List, Self

import httpx

from hypermedia_client.core.hco.hco_base import Hco, Property
from hypermedia_client.core.hco.link_hco import LinkHco
from hypermedia_client.job_management.known_relations import Relations
from hypermedia_client.job_management.model.sirenentities import ProcessingStepUsedTagsEntity


class ProcessingStepUsedTagsHco(Hco[{ProcessingStepUsedTagsEntity}]):
    tags: List[str] | None = Property()

    self_link: 'ProcessingStepUsedTagsLink'

    @classmethod
    def from_entity(cls, entity: ProcessingStepUsedTagsEntity, client: httpx.Client) -> Self:
        instance = cls(client, entity)

        Hco.check_classes(instance._entity.class_, ["ProcessingStepUsedTags"])

        instance.self_link = ProcessingStepUsedTagsLink.from_entity(instance._client, instance._entity, Relations.SELF)

        return instance


class ProcessingStepUsedTagsLink(LinkHco):
    def navigate(self) -> ProcessingStepUsedTagsHco:
        return ProcessingStepUsedTagsHco.from_entity(self._navigate_internal(ProcessingStepUsedTagsEntity), self._client)

