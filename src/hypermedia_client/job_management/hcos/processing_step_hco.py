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
from datetime import datetime
from typing import List, Self

import httpx

from hypermedia_client.core.hco.action_with_parameters_hco import ActionWithParametersHco
from hypermedia_client.core.hco.download_link_hco import DownloadLinkHco
from hypermedia_client.core.hco.hco_base import Hco, Property
from hypermedia_client.core.hco.link_hco import LinkHco
from hypermedia_client.job_management.known_relations import Relations
from hypermedia_client.job_management.model.open_api_generated import DataSpecificationHto, \
    SetProcessingStepTagsParameters
from hypermedia_client.job_management.model.sirenentities import ProcessingStepEntity


class ProcessingStepLink(LinkHco):
    def navigate(self) -> 'ProcessingStepHco':
        return ProcessingStepHco.from_entity(self._navigate_internal(ProcessingStepEntity), self._client)


class ProcessingStepEditTagsAction(ActionWithParametersHco[SetProcessingStepTagsParameters]):
    def execute(self, parameters: SetProcessingStepTagsParameters):
        self._execute(parameters)

    def default_parameters(self) -> SetProcessingStepTagsParameters:
        # todo check why we have to manually set tags
        return self._get_default_parameters(SetProcessingStepTagsParameters, SetProcessingStepTagsParameters(tags=[]))


class ProcessingStepHco(Hco[ProcessingStepEntity]):
    title: str = Property()
    version: str | None = Property()
    function_name: str | None = Property()
    short_description: str | None = Property()
    long_description: str | None = Property()
    tags: list[str] | None = Property()
    has_parameters: bool | None = Property()
    is_public: bool | None = Property()
    is_configured: bool | None = Property()
    created_at: datetime | None = Property()
    last_modified_at: datetime | None = Property()
    parameter_schema: str | None = Property()
    default_parameters: str | None = Property()
    return_schema: str | None = Property()
    error_schema: str | None = Property()
    input_data_slot_specification: List[DataSpecificationHto] | None = Property()
    output_data_slot_specification: List[DataSpecificationHto] | None = Property()
    edit_tags_action: ProcessingStepEditTagsAction | None

    self_link: ProcessingStepLink
    download_link: DownloadLinkHco

    @classmethod
    def from_entity(cls, entity: ProcessingStepEntity, client: httpx.Client) -> Self:
        instance = cls(client, entity)
        Hco.check_classes(instance._entity.class_, ["ProcessingStep"])

        instance.self_link = ProcessingStepLink.from_entity(instance._client, instance._entity, Relations.SELF)
        instance.download_link = DownloadLinkHco.from_entity(instance._client, instance._entity, Relations.DOWNLOAD)

        # todo tests

        # actions
        # todo
        #   "EditProperties"
        #   "Upload"
        #   "ConfigureDefaultParameters"
        #   "ClearDefaultParameters"
        instance.edit_tags_action = ProcessingStepEditTagsAction.from_entity_optional(
            client, instance._entity, "EditTags")
        return instance
