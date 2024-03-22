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
from hypermedia_client.job_management.hcos.user_hco import UserHco
from hypermedia_client.job_management.known_relations import Relations
from hypermedia_client.job_management.model.sirenentities import InfoEntity, UserEntity


class InfoLink(LinkHco):
    def navigate(self) -> "InfoHco":
        return InfoHco.from_entity(self._navigate_internal(InfoEntity), self._client)


class InfoHco(Hco[InfoEntity]):
    api_version: str = Property()
    build_version: str = Property()
    current_user: UserHco

    self_link: InfoLink

    @classmethod
    def from_entity(cls, entity: InfoEntity, client: httpx.Client) -> Self:
        instance = cls(client, entity)

        Hco.check_classes(instance._entity.class_, ["Info"])

        instance.self_link = InfoLink.from_entity(
            instance._client, instance._entity, Relations.SELF
        )

        instance._extract_current_user()

        return instance

    def _extract_current_user(self):
        user_entity = self._entity.find_first_entity_with_relation(
            Relations.CURRENT_USER, UserEntity)
        self.current_user = UserHco.from_entity(user_entity, self._client)
