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
from typing import List

import httpx

from hypermedia_client.core.hco.hco_base import Hco, Property
from hypermedia_client.core.hco.link_hco import LinkHco
from hypermedia_client.job_management.hcos.job_hco import JobHco
from hypermedia_client.job_management.known_relations import Relations
from hypermedia_client.job_management.model.sirenentities import JobQueryResultEntity, JobEntity


class JobQueryResultPaginationLink(LinkHco):
    def navigate(self) -> 'JobQueryResultHco':
        return JobQueryResultHco.from_entity(self._client, self._navigate_internal(JobQueryResultEntity))


class JobQueryResultLink(LinkHco):
    def navigate(self) -> 'JobQueryResultHco':
        return JobQueryResultHco.from_entity(self._client, self._navigate_internal(JobQueryResultEntity))


class JobQueryResultHco(Hco[JobQueryResultEntity]):
    self_link: JobQueryResultLink
    all_link: JobQueryResultPaginationLink | None
    first_link: JobQueryResultPaginationLink | None
    last_link: JobQueryResultPaginationLink | None
    next_link: JobQueryResultPaginationLink | None
    previous_link: JobQueryResultPaginationLink | None

    total_entities: int = Property()
    current_entities_count: int = Property()
    jobs: List[JobHco]

    @classmethod
    def from_entity(cls, client: httpx.Client, entity: JobQueryResultEntity) -> 'JobQueryResultHco':
        instance = cls(client, entity)

        Hco.check_classes(instance._entity.class_, ["JobQueryResult"])

        # pagination links
        instance.self_link = JobQueryResultLink.from_entity(instance._client, instance._entity, Relations.SELF)
        instance.all_link = JobQueryResultPaginationLink.from_entity_optional(instance._client, instance._entity,
                                                                              Relations.ALL)
        instance.first_link = JobQueryResultPaginationLink.from_entity_optional(instance._client, instance._entity,
                                                                                Relations.FIRST)
        instance.last_link = JobQueryResultPaginationLink.from_entity_optional(instance._client, instance._entity,
                                                                               Relations.LAST)
        instance.next_link = JobQueryResultPaginationLink.from_entity_optional(instance._client, instance._entity,
                                                                               Relations.NEXT)
        instance.previous_link = JobQueryResultPaginationLink.from_entity_optional(instance._client, instance._entity,
                                                                                   Relations.PREVIOUS)

        # entities
        instance._extract_jobs()

        return instance

    def _extract_jobs(self):
        self.jobs = []
        jobs = self._entity.find_all_entities_with_relation(Relations.ITEM, JobEntity)
        for job in jobs:
            job_hco: JobHco = JobHco.from_entity(job, self._client)
            self.jobs.append(job_hco)
