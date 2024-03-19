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

from hypermedia_client.core import MediaTypes, Link
from hypermedia_client.core.hco.action_with_parameters_hco import ActionWithParametersHco
from hypermedia_client.core.hco.hco_base import Hco
from hypermedia_client.core.hco.link_hco import LinkHco
from hypermedia_client.core.hco.upload_action_hco import UploadAction, UploadParameters
from hypermedia_client.job_management.hcos.workdata_used_tags_hco import WorkDataUsedTagsLink
from hypermedia_client.job_management.hcos.workdata_hco import WorkDataLink
from hypermedia_client.job_management.hcos.workdata_query_result_hco import (
    WorkDataQueryResultHco,
    WorkDataQueryResultLink,
    WorkDataQueryResultPaginationLink
)
from hypermedia_client.job_management.known_relations import Relations
from hypermedia_client.job_management.model import WorkDataQueryParameters
from hypermedia_client.job_management.model.sirenentities import WorkDataRootEntity


class WorkDataQueryAction(ActionWithParametersHco[WorkDataQueryParameters]):
    def execute(self, parameters: WorkDataQueryParameters) -> WorkDataQueryResultHco:
        url = self._execute_returns_url(parameters)
        link = Link.from_url(url, [str(Relations.CREATED_RESSOURCE)], "Created query", MediaTypes.SIREN)
        # resolve link immediately
        return WorkDataQueryResultLink.from_link(self._client, link).navigate()

    def default_parameters(self) -> WorkDataQueryParameters:
        return self._get_default_parameters(WorkDataQueryParameters, WorkDataQueryParameters())


class WorkDataUploadAction(UploadAction):
    def execute(self, parameters: UploadParameters) -> WorkDataLink:
        url = self._upload(parameters)
        link = Link.from_url(url, [str(Relations.CREATED_RESSOURCE)], "Uploaded workdata", MediaTypes.SIREN)
        return WorkDataLink.from_link(self._client, link)


class WorkDataRootLink(LinkHco):
    def navigate(self) -> 'WorkDataRootHco':
        return WorkDataRootHco.from_entity(self._navigate_internal(WorkDataRootEntity), self._client)


class WorkDataRootHco(Hco[WorkDataRootEntity]):
    query_action: WorkDataQueryAction | None
    upload_action: WorkDataUploadAction | None

    self_link: WorkDataRootLink
    all_link: WorkDataQueryResultPaginationLink | None
    used_tags_link: WorkDataUsedTagsLink | None

    @classmethod
    def from_entity(cls, entity: WorkDataRootEntity, client: httpx.Client) -> Self:
        instance = cls(client, entity)
        Hco.check_classes(instance._entity.class_, ["WorkDataRoot"])

        instance.query_action = WorkDataQueryAction.from_entity_optional(
            client, instance._entity, "CreateWorkDataQuery")
        instance.upload_action = WorkDataUploadAction.from_entity_optional(
            client, instance._entity, "Upload")

        instance.self_link = WorkDataRootLink.from_entity(
            instance._client, instance._entity, Relations.SELF)

        instance.all_link = WorkDataQueryResultPaginationLink.from_entity_optional(
            instance._client, instance._entity, Relations.ALL)

        instance.used_tags_link = WorkDataUsedTagsLink.from_entity_optional(
            instance._client, instance._entity, Relations.USED_TAGS)
        return instance
