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
from pydantic import BaseModel, ConfigDict, Field

from hypermedia_client.core import Entity
from hypermedia_client.job_management.model.open_api_generated import (
    InfoHtoOpenApiProperties,
    EntryPointHtoOpenApiProperties,
    JobsRootHtoOpenApiProperties,
    JobQueryResultHtoOpenApiProperties,
    JobHtoOpenApiProperties,
    WorkDataHtoOpenApiProperties,
    ProcessingStepHtoOpenApiProperties,
    WorkDataQueryResultHtoOpenApiProperties,
    WorkDataRootHtoOpenApiProperties,
    ProcessingStepRootHtoOpenApiProperties,
    ProcessingStepQueryResultHtoOpenApiProperties,
    WorkDataUsedTagsHtoOpenApiProperties,
    JobUsedTagsHtoOpenApiProperties,
    ProcessingStepUsedTagsHtoOpenApiProperties,
    UserHtoOpenApiProperties,
)


# ToDo: make these Generics bound to Entity


class EntryPointEntity(Entity):
    properties: EntryPointHtoOpenApiProperties | None = None


class InfoEntity(Entity):
    properties: InfoHtoOpenApiProperties | None = None


class JobsRootEntity(Entity):
    properties: JobsRootHtoOpenApiProperties | None = None


class JobQueryResultEntity(Entity):
    properties: JobQueryResultHtoOpenApiProperties | None = None


class JobEntity(Entity):
    properties: JobHtoOpenApiProperties | None = None


class WorkDataEntity(Entity):
    properties: WorkDataHtoOpenApiProperties | None = None


class WorkDataRootEntity(Entity):
    properties: WorkDataRootHtoOpenApiProperties | None = None


class WorkDataQueryResultEntity(Entity):
    properties: WorkDataQueryResultHtoOpenApiProperties | None = None


class ProcessingStepEntity(Entity):
    properties: ProcessingStepHtoOpenApiProperties | None = None


class ProcessingStepsRootEntity(Entity):
    properties: ProcessingStepRootHtoOpenApiProperties | None = None


class ProcessingStepQueryResultEntity(Entity):
    properties: ProcessingStepQueryResultHtoOpenApiProperties | None = None


class WorkDataUsedTagsEntity(Entity):
    properties: WorkDataUsedTagsHtoOpenApiProperties | None = None


class ProcessingStepUsedTagsEntity(Entity):
    properties: ProcessingStepUsedTagsHtoOpenApiProperties | None = None


class JobUsedTagsEntity(Entity):
    properties: JobUsedTagsHtoOpenApiProperties | None = None


class InputDataSlotHtoProperties(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )
    is_configured: bool | None = Field(None, alias="IsConfigured")
    title: str | None = Field(None, alias="Title")
    description: str | None = Field(None, alias="Description")
    media_type: str | None = Field(None, alias="MediaType")


class InputDataSlotEntity(Entity):
    properties: InputDataSlotHtoProperties | None = None


class OutputDataSlotHtoProperties(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )
    title: str | None = Field(None, alias="Title")
    description: str | None = Field(None, alias="Description")
    media_type: str | None = Field(None, alias="MediaType")


class OutputDataSlotEntity(Entity):
    properties: OutputDataSlotHtoProperties | None = None


class UserEntity(Entity):
    properties: UserHtoOpenApiProperties | None = None
