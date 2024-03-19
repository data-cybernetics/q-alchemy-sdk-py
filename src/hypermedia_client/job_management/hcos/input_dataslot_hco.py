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

from hypermedia_client.core import Link, MediaTypes
from hypermedia_client.core.hco.action_hco import ActionHco
from hypermedia_client.core.hco.action_with_parameters_hco import ActionWithParametersHco
from hypermedia_client.core.hco.hco_base import Hco, Property
from hypermedia_client.core.hco.link_hco import LinkHco
from hypermedia_client.core.hco.upload_action_hco import UploadAction, UploadParameters
from hypermedia_client.job_management.hcos.workdata_hco import WorkDataLink, WorkDataHco
from hypermedia_client.job_management.known_relations import Relations
from hypermedia_client.job_management.model.open_api_generated import SelectWorkDataForDataSlotParameters
from hypermedia_client.job_management.model.sirenentities import InputDataSlotEntity, WorkDataEntity


class InputDataSlotLink(LinkHco):
    def navigate(self) -> 'InputDataSlotHco':
        return InputDataSlotHco.from_entity(self._navigate_internal(InputDataSlotEntity), self._client)


class InputDataSlotSelectWorkDataAction(ActionWithParametersHco[SelectWorkDataForDataSlotParameters]):
    def execute(self, parameters: SelectWorkDataForDataSlotParameters):
        self._execute(parameters)


class InputDataSlotUploadWorkDataAction(UploadAction):
    def execute(self, parameters: UploadParameters) -> WorkDataLink:
        url = self._upload(parameters)
        link = Link.from_url(url, [str(Relations.CREATED_RESSOURCE)], "Uploaded workdata", MediaTypes.SIREN)
        return WorkDataLink.from_link(self._client, link)


class InputDataSlotClearDataAction(ActionHco):
    def execute(self):
        self._execute_internal()


class InputDataSlotHco(Hco[InputDataSlotEntity]):
    is_configured: bool | None = Property()
    title: str | None = Property()
    description: str | None = Property()
    media_type: str | None = Property()
    selected_workdata: WorkDataHco | None

    select_workdata_action: InputDataSlotSelectWorkDataAction | None
    clear_workdata_action: InputDataSlotClearDataAction | None

    @classmethod
    def from_entity(cls, entity: InputDataSlotEntity, client: httpx.Client) -> Self:
        instance = cls(client, entity)
        Hco.check_classes(instance._entity.class_, ["InputDataSlot"])

        # actions
        instance.select_workdata_action = InputDataSlotSelectWorkDataAction.from_entity_optional(
            client, instance._entity, "SelectWorkData")
        instance.clear_workdata_action = InputDataSlotClearDataAction.from_entity_optional(
            client, instance._entity, "Clear")

        instance._extract_workdata()

        return instance

    def _extract_workdata(self):
        self.selected_workdata = None

        workdata: WorkDataEntity | None = self._entity.find_first_entity_with_relation(Relations.SELECTED,
                                                                                       WorkDataEntity)
        if not workdata:
            return

        self.selected_workdata = WorkDataHco.from_entity(workdata, self._client)
