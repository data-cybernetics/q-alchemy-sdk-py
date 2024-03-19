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
import warnings
from typing import List, Any, Union, Type, TypeVar, Self

from httpx import URL
from pydantic import BaseModel, ConfigDict, constr, Field, field_validator, model_validator

TParameters = TypeVar('TParameters', bound='BaseModel')
TEntity = TypeVar('TEntity', bound='Entity')


class SirenBaseModel(BaseModel):
    model_config = ConfigDict(
        extra='forbid'
    )


class ActionField(SirenBaseModel):
    name: constr(min_length=1)
    type: str | None = None
    value: Any | None = None
    class_: List[str] | None = Field(None, alias='class')
    title: str | None = None
    accept: str | None = None
    maxFileSizeBytes: int | None = None
    allowMultiple: bool | None = None


class Action(SirenBaseModel):
    name: constr(min_length=1)
    href: constr(min_length=1)
    class_: List[str] | None = Field(None, alias='class')
    method: str | None = None
    title: str | None = None
    type: str | None = None
    fields: List[ActionField] | None = None

    def has_parameters(self) -> bool:
        return (self.fields is not None) and (len(self.fields) > 0)

    def get_default_parameters(self, parameter_type: Type[TParameters] = Any,
                               default_if_none: TParameters | None = None) -> TParameters | None:
        if not self.has_parameters():
            raise Exception("Can not get default parameters for action without parameters")
        if len(self.fields) > 1:
            raise Exception("Action has more than one field, can not determine default parameters")
        if not self.fields[0].value:
            return default_if_none
        dump = self.fields[0].model_dump(by_alias=True)['value']
        return parameter_type.model_validate(dump)


class EmbeddedLinkEntity(SirenBaseModel):
    class_: List[str] | None = Field(None, alias='class')
    rel: List[str]
    href: constr(min_length=1)
    title: str | None = None
    type: str | None = None


class Link(SirenBaseModel):
    href: constr(min_length=1)
    rel: List[str]
    class_: List[str] | None = Field(None, alias='class')
    title: str | None = None
    type: str | None = None

    @classmethod
    def from_url(cls,
                 url: URL,
                 relation: list[str],
                 title: str | None = None,
                 mediatype: str | None = None,
                 class_: list[str] | None = None) -> Self:
        instance = cls(href=str(url), rel=relation, title=title, type=mediatype)
        instance.class_ = class_
        return instance


class Entity(SirenBaseModel):
    rel: List[str] = []  # used when embedded
    class_: List[str] | None = Field(None, alias='class')
    title: str | None = None
    properties: Any | None = None
    entities: List[Union['Entity', EmbeddedLinkEntity]] | None = None
    actions: List[Action] | None = None
    links: List[Link] | None = None

    @model_validator(mode='after')
    def _check_extra_properties(self) -> Self:
        if type(self) is Entity:
            # skip this validation for the base class
            return self
        if isinstance(self.properties, dict):
            warnings.warn(f"Unresolved properties in Entity '{self.title}'! Implementation might be incomplete?")
        elif isinstance(self.properties, BaseModel) and self.properties.model_extra:
            warnings.warn(
                f"Entity with extra properties received! Possibly a version mismatch "
                f"between server and client? "
                f"(unexpected properties: {[n for n in self.properties.model_extra.keys()]})"
            )
        return self

    def find_first_entity_with_relation(self, searched_relation: str, entity_type: Type[TEntity] = 'Entity'
                                        ) -> Union[TEntity, None]:
        if self.entities is None:
            return None

        for entity in self.entities:
            if self.contains_relation(entity.rel, searched_relation):
                return entity_type.model_validate(entity.model_dump(by_alias=True))
        return None

    def find_all_entities_with_relation(self, searched_relation: str,
                                        entity_type: Type[TEntity] = 'Entity') -> List[TEntity]:
        if self.entities is None:
            return []

        result = []
        for entity in self.entities:
            if self.contains_relation(entity.rel, searched_relation):
                # map to requested entity type so properties are not in a dict
                mapped = entity_type.model_validate(entity.model_dump(by_alias=True))
                result.append(mapped)
        return result

    def find_first_link_with_relation(self, searched_relation: str) -> Link | None:
        if self.links is None:
            return None

        for link in self.links:
            if self.contains_relation(link.rel, searched_relation):
                return link
        return None

    def action_exists(self, name: str) -> bool:
        return self.find_first_action_with_name(name) is not None

    @staticmethod
    def contains_relation(relations: list[str] | None, searched_relation: str) -> bool:
        if relations is None:
            return False

        for relation in relations:
            if relation == searched_relation:
                return True
        return False

    def find_first_action_with_name(self, name: str):
        if self.actions is None:
            return None

        for action in self.actions:
            if action.name == name:
                return action
        return None
