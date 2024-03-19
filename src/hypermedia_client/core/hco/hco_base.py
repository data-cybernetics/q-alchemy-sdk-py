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
from dataclasses import dataclass
from typing import List, TypeVar, Generic, Callable, Any

import httpx
from httpx import URL

from hypermedia_client.core import SirenException, Entity, BaseRelations, Link

TEntity = TypeVar('TEntity', bound=Entity)
THcoEntity = TypeVar('THcoEntity', bound=Entity)


class ClientContainer:
    _client: httpx.Client

    def __init__(self, client: httpx.Client):
        self._client = client

    def __repr__(self):
        variables = [f"{k}: {repr(v)}" for k, v in vars(self).items() if not k.startswith('_')]
        var_repr = f" ({', '.join(variables)})"
        return f"<{self.__class__.__name__}{var_repr if variables else ''}>"


@dataclass
class Property:
    """
    Sentinel object marking properties in an HCO object to be copied from the entity.

    Args:
        name: Name of the property in the entity. If not provided, it's assumed to be
            the same as the property name in the Hco. [optional]
        converter: A callable that will be applied to the value in the entity before
            being assigned to the property in the Hco. [optional]
    """
    name: str | None = None
    converter: Callable[[Any], Any] | None = None


class Hco(ClientContainer, Generic[THcoEntity]):
    _entity: THcoEntity

    def __init__(self, client: httpx.Client, entity: THcoEntity):
        super().__init__(client)
        self._entity = entity
        self._set_properties()

    def __hash__(self):
        # TODO: write a unit test ;)
        candidate_self_link: Link | None = self._entity.find_first_link_with_relation(BaseRelations.SELF)
        if candidate_self_link is not None:
            return hash(URL(candidate_self_link.href).path)
        else:
            return super().__hash__()

    def __eq__(self, other):
        # TODO: write a unit test ;)
        if isinstance(other, Hco):
            return hash(self) == hash(other)
        else:
            return False

    @staticmethod
    def check_classes(existing_classes: List[str], expected_classes: List[str]):
        for expected_class in expected_classes:
            if expected_class not in existing_classes:
                raise SirenException(
                    f"Error while mapping entity:expected hco class {expected_class} is not a class of generic entity "
                    f"with classes {existing_classes}")

    def _set_properties(self):
        """Initializes Hco properties from the entity.

        Iterates over all properties defined in the class annotation. If one is initialized
        with `Property()` the variable gets set from a property in self._entity.properties
        with the same name.
        """
        for var_name, var_type in self.__annotations__.items():
            # Get the initialized object. The annotation contain only the annotated type.
            hco_property = getattr(self, var_name, None)
            # Skip everything that is not initialized as `Property()`
            if not (hco_property and isinstance(hco_property, Property)):
                continue

            if not hco_property.name:
                hco_property.name = var_name
            property_value = getattr(self._entity.properties, var_name)
            if hco_property.converter:
                property_value = hco_property.converter(property_value)
            setattr(self, var_name, property_value)

