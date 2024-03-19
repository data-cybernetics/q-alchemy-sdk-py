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
from hypermedia_client.job_management.hcos.workdata_hco import WorkDataHco
from hypermedia_client.job_management.known_relations import Relations
from hypermedia_client.job_management.model.sirenentities import (
    OutputDataSlotEntity,
    WorkDataEntity
)


class OutputDataSlotLink(LinkHco):
    def navigate(self) -> 'OutputDataSlotHco':
        return OutputDataSlotHco.from_entity(self._navigate_internal(OutputDataSlotEntity), self._client)


class OutputDataSlotHco(Hco[OutputDataSlotEntity]):
    title: str | None = Property()
    description: str | None = Property()
    media_type: str | None = Property()
    assigned_workdata: WorkDataHco | None

    @classmethod
    def from_entity(cls, entity: OutputDataSlotEntity, client: httpx.Client) -> Self:
        instance = cls(client, entity)
        Hco.check_classes(instance._entity.class_, ["OutputDataSlot"])

        instance._extract_workdata()

        return instance

    def _extract_workdata(self):
        self.assigned_workdata = None

        workdata: WorkDataEntity | None = self._entity.find_first_entity_with_relation(Relations.ASSIGNED,
                                                                                       WorkDataEntity)
        if not workdata:
            return

        self.assigned_workdata = WorkDataHco.from_entity(workdata, self._client)
