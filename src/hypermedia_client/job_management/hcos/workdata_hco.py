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
from typing import Self

import httpx

from hypermedia_client.core.hco.action_hco import ActionHco
from hypermedia_client.core.hco.action_with_parameters_hco import ActionWithParametersHco
from hypermedia_client.core.hco.download_link_hco import DownloadLinkHco
from hypermedia_client.core.hco.hco_base import Hco, Property
from hypermedia_client.core.hco.link_hco import LinkHco
from hypermedia_client.job_management.known_relations import Relations
from hypermedia_client.job_management.model.open_api_generated import (
    SetNameWorkDataParameters,
    SetCommentWorkDataParameters,
    SetTagsWorkDataParameters,
    WorkDataKind
)
from hypermedia_client.job_management.model.sirenentities import WorkDataEntity


class WorkDataLink(LinkHco):
    def navigate(self) -> 'WorkDataHco':
        return WorkDataHco.from_entity(self._navigate_internal(WorkDataEntity), self._client)


class WorkDataDeleteAction(ActionHco):
    def execute(self):
        self._execute_internal()


class WorkDataRenameAction(ActionWithParametersHco[SetNameWorkDataParameters]):
    def execute(self, parameters: SetNameWorkDataParameters):
        self._execute(parameters)

    def default_parameters(self) -> SetNameWorkDataParameters:
        return self._get_default_parameters(SetNameWorkDataParameters, SetNameWorkDataParameters())


class WorkDataEditCommentAction(ActionWithParametersHco[SetCommentWorkDataParameters]):
    def execute(self, parameters: SetCommentWorkDataParameters):
        self._execute(parameters)

    def default_parameters(self) -> SetCommentWorkDataParameters:
        return self._get_default_parameters(SetCommentWorkDataParameters, SetCommentWorkDataParameters())


class WorkDataEditTagsAction(ActionWithParametersHco[SetTagsWorkDataParameters]):
    def execute(self, parameters: SetTagsWorkDataParameters):
        self._execute(parameters)

    def default_parameters(self) -> SetTagsWorkDataParameters:
        # todo check why we have to manually set tags
        return self._get_default_parameters(SetTagsWorkDataParameters, SetTagsWorkDataParameters(tags=[]))


class WorkDataAllowDeletionAction(ActionHco):
    def execute(self):
        self._execute_internal()


class WorkDataDisallowAction(ActionHco):
    def execute(self):
        self._execute_internal()


class WorkDataHideAction(ActionHco):
    def execute(self):
        self._execute_internal()


class WorkDataUnHideAction(ActionHco):
    def execute(self):
        self._execute_internal()


class WorkDataHco(Hco[WorkDataEntity]):
    name: str | None = Property()
    created_on: datetime | None = Property()
    size_in_bytes: int | None = Property()
    tags: list[str] | None = Property()
    media_type: str | None = Property()
    kind: WorkDataKind | None = Property()
    comments: str | None = Property()
    is_deletable: bool | None = Property()
    hidden: bool | None = Property()

    delete_action: WorkDataDeleteAction | None
    hide_action: WorkDataHideAction | None
    unhide_action: WorkDataUnHideAction | None
    allow_deletion_action: WorkDataAllowDeletionAction | None
    disallow_deletion_action: WorkDataDisallowAction | None
    rename_action: WorkDataRenameAction | None
    edit_comment_action: WorkDataEditCommentAction | None
    edit_tags_action: WorkDataEditTagsAction | None

    self_link: WorkDataLink
    download_link: DownloadLinkHco

    @classmethod
    def from_entity(cls, entity: WorkDataEntity, client: httpx.Client) -> Self:
        instance = cls(client, entity)
        Hco.check_classes(instance._entity.class_, ["WorkData"])

        # actions
        instance.hide_action = WorkDataHideAction.from_entity_optional(
            client, instance._entity, "Hide")
        instance.unhide_action = WorkDataUnHideAction.from_entity_optional(
            client, instance._entity, "UnHide")
        instance.delete_action = WorkDataDeleteAction.from_entity_optional(
            client, instance._entity, "Delete")
        instance.rename_action = WorkDataRenameAction.from_entity_optional(
            client, instance._entity, "Rename")
        instance.edit_comment_action = WorkDataEditCommentAction.from_entity_optional(
            client, instance._entity, "EditComment")
        instance.edit_tags_action = WorkDataEditTagsAction.from_entity_optional(
            client, instance._entity, "EditTags")
        instance.allow_deletion_action = WorkDataAllowDeletionAction.from_entity_optional(
            client, instance._entity, "AllowDeletion")
        instance.disallow_deletion_action = WorkDataDisallowAction.from_entity_optional(
            client, instance._entity, "DisallowDeletion")

        # links
        instance.self_link = WorkDataLink.from_entity(
            instance._client, instance._entity, Relations.SELF)
        instance.download_link = DownloadLinkHco.from_entity(
            instance._client, instance._entity, Relations.DOWNLOAD)
        return instance
