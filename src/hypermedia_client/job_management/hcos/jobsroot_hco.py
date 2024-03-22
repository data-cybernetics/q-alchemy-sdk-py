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
from httpx import URL

from hypermedia_client.core import MediaTypes, Link
from hypermedia_client.core.hco.action_with_parameters_hco import ActionWithParametersHco
from hypermedia_client.core.hco.hco_base import Hco
from hypermedia_client.core.hco.link_hco import LinkHco
from hypermedia_client.job_management.hcos.job_hco import JobLink
from hypermedia_client.job_management.hcos.job_query_result_hco import (
    JobQueryResultHco,
    JobQueryResultLink
)
from hypermedia_client.job_management.hcos.job_used_tags_hco import JobUsedTagsLink
from hypermedia_client.job_management.known_relations import Relations
from hypermedia_client.job_management.model.open_api_generated import (
    CreateJobParameters,
    JobQueryParameters,
    CreateSubJobParameters
)
from hypermedia_client.job_management.model.sirenentities import JobsRootEntity


class CreateJobAction(ActionWithParametersHco[CreateJobParameters]):
    def execute(self, parameters: CreateJobParameters) -> JobLink:
        url: URL = self._execute_returns_url(parameters)
        link = Link.from_url(url, [str(Relations.CREATED_RESSOURCE)], "Created job", MediaTypes.SIREN)
        return JobLink.from_link(self._client, link)

    def default_parameters(self) -> CreateJobParameters:
        return self._get_default_parameters(CreateJobParameters, CreateJobParameters())


class CreateSubJobAction(ActionWithParametersHco[CreateSubJobParameters]):
    def execute(self, parameters: CreateSubJobParameters) -> JobLink:
        url = self._execute_returns_url(parameters)
        link = Link.from_url(url, [str(Relations.CREATED_RESSOURCE)], "Created sub-job", MediaTypes.SIREN)
        return JobLink.from_link(self._client, link)

    def default_parameters(self) -> CreateSubJobParameters:
        return self._get_default_parameters(CreateSubJobParameters, CreateSubJobParameters())


class JobQueryAction(ActionWithParametersHco):
    def execute(self, parameters: JobQueryParameters) -> JobQueryResultHco:
        url = self._execute_returns_url(parameters)
        link = Link.from_url(url, [str(Relations.CREATED_RESSOURCE)], "Created job query", MediaTypes.SIREN)
        # resolve link immediately
        return JobQueryResultLink.from_link(self._client, link).navigate()

    def default_parameters(self) -> JobQueryParameters:
        return self._get_default_parameters(JobQueryParameters, JobQueryParameters())


class JobsRootHco(Hco[JobsRootEntity]):
    create_job_action: CreateJobAction | None
    job_query_action: JobQueryAction | None
    create_subjob_action: CreateSubJobAction | None
    used_tags_link: JobUsedTagsLink | None

    self_link: 'JobsRootLink'

    @classmethod
    def from_entity(cls, entity: JobsRootEntity, client: httpx.Client) -> Self:
        instance = cls(client, entity)

        Hco.check_classes(instance._entity.class_, ["JobsRoot"])
        instance.create_job_action = CreateJobAction.from_entity_optional(client, instance._entity, "CreateJob")
        instance.create_subjob_action = CreateSubJobAction.from_entity_optional(client, instance._entity, "CreateSubJob")
        instance.job_query_action = JobQueryAction.from_entity_optional(client, instance._entity, "CreateJobQuery")
        instance.used_tags_link = JobUsedTagsLink.from_entity_optional(
            instance._client, instance._entity, Relations.USED_TAGS)
        instance.self_link = JobsRootLink.from_entity(instance._client, instance._entity, Relations.SELF)
        return instance


class JobsRootLink(LinkHco):
    def navigate(self) -> JobsRootHco:
        return JobsRootHco.from_entity(self._navigate_internal(JobsRootEntity), self._client)
