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
from typing import Self, List

import httpx
from pydantic import BaseModel, ConfigDict

from hypermedia_client.core.hco.action_hco import ActionHco
from hypermedia_client.core.hco.action_with_parameters_hco import ActionWithParametersHco
from hypermedia_client.core.hco.hco_base import Hco, Property
from hypermedia_client.core.hco.link_hco import LinkHco
from hypermedia_client.job_management.hcos import InputDataSlotHco
from hypermedia_client.job_management.hcos.output_dataslot_hco import OutputDataSlotHco
from hypermedia_client.job_management.hcos.processing_step_hco import ProcessingStepLink
from hypermedia_client.job_management.known_relations import Relations
from hypermedia_client.job_management.model.open_api_generated import JobStates, ProcessingView, RenameJobParameters, \
    SelectProcessingParameters, SetJobTagsParameters
from hypermedia_client.job_management.model.sirenentities import JobEntity, InputDataSlotEntity, OutputDataSlotEntity


class JobRenameAction(ActionWithParametersHco[RenameJobParameters]):
    def execute(self, parameters: RenameJobParameters):
        self._execute(parameters)

    def default_parameters(self) -> RenameJobParameters:
        return self._get_default_parameters(RenameJobParameters, RenameJobParameters())


class JobSelectProcessingAction(ActionWithParametersHco[SelectProcessingParameters]):
    def execute(self, parameters: SelectProcessingParameters):
        self._execute(parameters)

    def default_parameters(self) -> SelectProcessingParameters:
        return self._get_default_parameters(SelectProcessingParameters, SelectProcessingParameters())


class JobHideAction(ActionHco):
    def execute(self):
        self._execute_internal()


class JobUnHideAction(ActionHco):
    def execute(self):
        self._execute_internal()


class JobAllowOutputDataDeletionAction(ActionHco):
    def execute(self):
        self._execute_internal()


class JobDisAllowOutputDataDeletionAction(ActionHco):
    def execute(self):
        self._execute_internal()


class GenericProcessingConfigureParameters(BaseModel):
    """Generic parameter model, that can be set with any dictionary"""
    model_config = ConfigDict(extra='allow')


class JobConfigureProcessingAction(ActionWithParametersHco[GenericProcessingConfigureParameters]):
    def execute(self, parameters: GenericProcessingConfigureParameters):
        self._execute(parameters)

    def default_parameters(self) -> GenericProcessingConfigureParameters:
        return self._get_default_parameters(GenericProcessingConfigureParameters,
                                            GenericProcessingConfigureParameters())


class JobStartProcessingAction(ActionHco):
    def execute(self):
        self._execute_internal()


class JobLink(LinkHco):
    def navigate(self) -> 'JobHco':
        return JobHco.from_entity(self._navigate_internal(JobEntity), self._client)


class ParentJobLink(LinkHco):
    def navigate(self) -> 'JobHco':
        return JobHco.from_entity(self._navigate_internal(JobEntity), self._client)


class JobEditTagsAction(ActionWithParametersHco[SetJobTagsParameters]):
    def execute(self, parameters: SetJobTagsParameters):
        self._execute(parameters)

    def default_parameters(self) -> SetJobTagsParameters:
        # todo check why we have to manually set tags
        return self._get_default_parameters(SetJobTagsParameters, SetJobTagsParameters(tags=[]))


class JobHco(Hco[JobEntity]):
    name: str = Property()
    state: JobStates = Property()
    hidden: bool = Property()
    tags: list[str] | None = Property()
    output_is_deletable: bool = Property()
    created_on: datetime = Property()
    completed_on: datetime = Property()
    error_description: str = Property()
    processing: ProcessingView = Property()
    result: str = Property()

    self_link: JobLink
    parent_link: ParentJobLink | None
    selected_processing_step_link: ProcessingStepLink | None

    rename_action: JobRenameAction
    select_processing_action: JobSelectProcessingAction
    configure_processing_action: JobConfigureProcessingAction
    start_processing_action: JobStartProcessingAction
    hide_action: JobHideAction
    unhide_action: JobUnHideAction
    allow_output_data_deletion_action: JobAllowOutputDataDeletionAction
    disallow_output_data_deletion_action: JobDisAllowOutputDataDeletionAction
    edit_tags_action: JobEditTagsAction | None

    input_dataslots: List[InputDataSlotHco]
    output_dataslots: List[OutputDataSlotHco]

    @classmethod
    def from_entity(cls, entity: JobEntity, client: httpx.Client) -> Self:
        instance = cls(client, entity)

        Hco.check_classes(instance._entity.class_, ["Job"])

        instance.self_link = JobLink.from_entity(
            instance._client, instance._entity, Relations.SELF)
        instance.parent_link = ParentJobLink.from_entity_optional(
            instance._client, instance._entity, Relations.PARENT_JOB)
        instance.selected_processing_step_link = ProcessingStepLink.from_entity_optional(
            instance._client, instance._entity, Relations.SELECTED_PROCESSING_STEP)

        # actions
        instance.hide_action = JobHideAction.from_entity_optional(
            client, instance._entity, "Hide")
        instance.unhide_action = JobUnHideAction.from_entity_optional(
            client, instance._entity, "UnHide")
        instance.rename_action = JobRenameAction.from_entity_optional(
            client, instance._entity, "Rename")
        instance.select_processing_action = JobSelectProcessingAction.from_entity_optional(
            client, instance._entity, "SelectProcessing")
        instance.configure_processing_action = JobConfigureProcessingAction.from_entity_optional(
            client, instance._entity, "ConfigureProcessing")
        instance.start_processing_action = JobStartProcessingAction.from_entity_optional(
            client, instance._entity, "StartProcessing")
        instance.allow_output_data_deletion_action = JobAllowOutputDataDeletionAction.from_entity_optional(
            client, instance._entity, "AllowOutputDataDeletion")
        instance.disallow_output_data_deletion_action = JobDisAllowOutputDataDeletionAction.from_entity_optional(
            client, instance._entity, "DisallowOutputDataDeletion")
        instance.edit_tags_action = JobEditTagsAction.from_entity_optional(
            client, instance._entity, "EditTags")

        # entities
        instance._extract_input_dataslots()
        instance._extract_output_dataslots()

        return instance

    def _extract_input_dataslots(self):
        self.input_dataslots = []
        input_dataslots = self._entity.find_all_entities_with_relation(Relations.INPUT_DATASLOT, InputDataSlotEntity)
        for input_dataslot in input_dataslots:
            input_dataslot_hco: InputDataSlotHco = InputDataSlotHco.from_entity(input_dataslot, self._client)
            self.input_dataslots.append(input_dataslot_hco)

    def _extract_output_dataslots(self):
        self.output_dataslots = []
        output_dataslots = self._entity.find_all_entities_with_relation(Relations.OUTPUT_DATASLOT, OutputDataSlotEntity)
        for output_dataslot in output_dataslots:
            output_dataslot_hco: OutputDataSlotHco = OutputDataSlotHco.from_entity(output_dataslot, self._client)
            self.output_dataslots.append(output_dataslot_hco)
