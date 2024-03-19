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
import json
import logging
from io import BytesIO
from typing import Any, BinaryIO, Iterable, AsyncIterable

import httpx
from httpx import Response
from httpx import URL
from pydantic import BaseModel

from .exceptions import SirenException
from .http_headers import Headers
from .base_relations import BaseRelations
from .media_types import MediaTypes
from .model.error import ProblemDetails
from .model.sirenmodels import Entity, Link, Action, TEntity

logger = logging.getLogger(__name__)


# for now, we do not support navigations to non siren links (e.g. external)
def get_resource(client: httpx.Client, href: str, media_type: str = MediaTypes.SIREN,
                 parse_type: type[TEntity] = Entity) -> TEntity | ProblemDetails | Response:
    try:
        # assume get for links
        response = client.get(href)
    except (httpx.ConnectTimeout, httpx.ConnectError) as exc:
        raise SirenException(f"Http-client error requesting resource: {href}") from exc
    expected_type = media_type or MediaTypes.SIREN  # if not specified expect siren

    if response.status_code == httpx.codes.OK:
        if (media_type := response.headers.get(Headers.CONTENT_TYPE, MediaTypes.SIREN)) != expected_type:
            logger.warning(f"Expected type {expected_type} not matched by response: "
                           f"' got: '{media_type}'")

        if media_type == MediaTypes.SIREN.value or media_type is None:  # assume siren if not specified
            resp = response.content.decode()
            entity = parse_type.model_validate_json(resp)
            return entity
        else:
            return response

    elif response.status_code >= 400:
        return handle_error_response(response)
    else:
        logger.warning(f"Unexpected return code: {response.status_code}")
        return response


def navigate(client: httpx.Client, link: Link,
             parse_type: type[TEntity] = Entity) -> TEntity | ProblemDetails | Response:
    return get_resource(client, link.href, link.type, parse_type)


def ensure_siren_response(response: TEntity | ProblemDetails | Response) -> TEntity:
    if isinstance(response, ProblemDetails):
        raise SirenException(
            f"Error while navigating: {response}")
    if isinstance(response, Response):
        raise SirenException(
            f"Error while navigating, unexpected response: {response}")
    return response


def upload_json(client: httpx.Client, action: Action, json_payload: Any,
                filename: str) -> None | URL | ProblemDetails | Response:
    return upload_binary(client, action, json.dumps(json_payload), filename, MediaTypes.APPLICATION_JSON)


def upload_binary(client: httpx.Client, action: Action, content: str | bytes | Iterable[bytes] | AsyncIterable[bytes],
                  filename: str,
                  mediatype: str = MediaTypes.OCTET_STREAM) -> None | URL | ProblemDetails | Response:

    if isinstance(content, str):
        payload = BytesIO(content.encode(encoding="utf-8"))
    elif isinstance(content, bytes):
        payload = BytesIO(content)
    else:
        # Fixme: iterable are not supported. Use the 'content' instead of the 'file' parameter of the request call?
        raise NotImplemented('Iterables are not supported as payload (yet)! Convert to bytes or string.')

    return upload_file(client, action, payload, filename, mediatype)


# for now no support for multi file upload
def upload_file(client: httpx.Client, action: Action, file: BinaryIO, filename: str,
                mediatype: str = MediaTypes.OCTET_STREAM) -> None | URL | ProblemDetails | Response:
    if action.type != MediaTypes.MULTIPART_FORM_DATA:
        raise SirenException(
            f"Action with upload requires type: {MediaTypes.MULTIPART_FORM_DATA} but found: {action.type}")

    files = {'upload-file': (filename, file, mediatype)}
    try:
        response = client.request(method=action.method, url=action.href, files=files)
    except httpx.RequestError as exc:
        raise SirenException(f"Error from httpx while uploading data to: {action.href}") from exc
    return handle_action_result(response)


def execute_action_on_entity(client: httpx.Client, entity: Entity, name: str, parameters: BaseModel | None = None):
    action = entity.find_first_action_with_name(name)
    if action is None:
        raise SirenException(f"Entity does not contain expected action: {name}")

    return execute_action(client, action, parameters)


def execute_action(client: httpx.Client, action: Action,
                   parameters: BaseModel | None = None) -> None | URL | ProblemDetails | Response:
    if action.has_parameters() is False:
        # no parameters required
        if parameters is not None:
            raise SirenException(f"Action requires no parameters but got some")
    else:
        # parameters required
        if parameters is None:
            raise SirenException(f"Action requires parameters but non provided")

    action_parameters = None
    if parameters is not None:
        action_parameters = parameters.model_dump_json(by_alias=True, exclude_none=True)

    try:
        response = client.request(
            method=action.method,
            url=action.href,
            content=action_parameters,
            headers={Headers.CONTENT_TYPE.value: MediaTypes.APPLICATION_JSON.value}
        )
    except httpx.RequestError as exc:
        raise SirenException(f"Error from httpx while executing action: {action.href}") from exc

    return handle_action_result(response)


def handle_action_result(response: Response) -> None | URL | ProblemDetails | Response:
    if response.status_code == httpx.codes.OK:
        return
    if response.status_code == httpx.codes.CREATED:
        location_header = response.headers.get(Headers.LOCATION_HEADER, None)
        if location_header is None:
            logger.warning(f"Got created response without location header")
            return

        return URL(location_header)

    elif response.status_code >= 400:
        return handle_error_response(response)
    else:
        logger.warning(f"Unexpected return code: {response.status_code}")
        return response


def handle_error_response(response: Response) -> ProblemDetails | Response:
    content_type = response.headers.get(Headers.CONTENT_TYPE, '')
    if content_type.startswith(MediaTypes.PROBLEM_DETAILS.value):
        return ProblemDetails.model_validate(response.json())
    else:
        logger.warning(
            f"Error case did not return media type: '{MediaTypes.PROBLEM_DETAILS.value}', "
            f"got '{response.headers.get(Headers.CONTENT_TYPE, None)}' instead!")
        return response


def navigate_self(client: httpx.Client, entity: Entity) -> Entity | ProblemDetails | None:
    if entity is None:
        return None

    self_link = entity.find_first_link_with_relation(BaseRelations.SELF)
    if self_link is None:
        return None

    return navigate(client, self_link)
