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
from io import BytesIO
from typing import Self

import httpx
from httpx import Response

from hypermedia_client.core import Link, Entity, get_resource, SirenException
from hypermedia_client.core.hco.link_hco import LinkHco


class DownloadLinkHco(LinkHco):

    @classmethod
    def from_link_optional(cls, client: httpx.Client, link: Link | None) -> Self | None:
        return super(DownloadLinkHco, cls).from_link_optional(client, link)

    @classmethod
    def from_entity_optional(cls, client: httpx.Client, entity: Entity, link_relation: str) -> Self | None:
        return super(DownloadLinkHco, cls).from_entity_optional(client, entity, link_relation)

    @classmethod
    def from_link(cls, client: httpx.Client, link: Link) -> Self:
        return super(DownloadLinkHco, cls).from_link(client, link)

    @classmethod
    def from_entity(cls, client: httpx.Client, entity: Entity, link_relation: str) -> Self:
        return super(DownloadLinkHco, cls).from_entity(client, entity, link_relation)

    def download(self) -> bytes:
        response: Response = get_resource(self._client, self._link.href, self._link.type)
        if not isinstance(response, Response):
            raise SirenException(
                f"Error while downloading resource: did not get response type")

        return response.content

    # TODO: download for large files
