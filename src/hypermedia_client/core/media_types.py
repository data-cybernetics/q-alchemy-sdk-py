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
from enum import StrEnum


class MediaTypes(StrEnum):
    APPLICATION_JSON = "application/json"
    SIREN = "application/vnd.siren+json"
    PROBLEM_DETAILS = "application/problem+json"
    MULTIPART_FORM_DATA = "multipart/form-data"
    OCTET_STREAM = "application/octet-stream"

    XML = "application/xml"
    ZIP = "application/zip"
    PDF = "application/pdf"
    TEXT = "text/plain"
    HTML = "text/html"
    CSV = "text/csv"
    SVG = "image/svg+xml"
    PNG = "image/png"
    JPEG = "image/jpeg"
    BMP = "image/bmp"


class SirenClasses(StrEnum):
    FileUploadAction = "FileUploadAction"
