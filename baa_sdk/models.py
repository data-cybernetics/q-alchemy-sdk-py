# Copyright 2022 data cybernetics ssc GmbH.

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
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel

LOG = logging.getLogger(__name__)


class RenameJob(BaseModel):
    new_name: str

    class Config:

        @staticmethod
        def dumps(v, *, default):
            return json.dumps({"NewName": v["new_name"]})

        json_dumps = dumps


class Strategy(str, Enum):
    GREEDY = "Greedy"
    BRUTE_FORCE = "BruteForce"

class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"

class JobConfig(BaseModel):
    max_fidelity_loss: float = 0.0
    strategy: str = "Greedy"
    max_combination_size: int = 0
    use_low_rank: bool = False
    load_per_cyle: int = 5000
    secede_modulo: int = 1
    max_time_sec: int = 1200
    max_loops: int = 200
    max_nodes: int = 50000
    max_level: int = 2
    batch_size: int = 100
    log_level: str = "INFO"
    redis_ttl_seconds: int = 3600


class JobContext(BaseModel):
    unique_id: str
    start_time: datetime
    config: JobConfig = None
    nodes_received: int = 0
    loops: int = 0
    open_computations: Optional[int] = None
    unhandled_nodes: Optional[int] = None

    def time_since_start(self):
        return datetime.now() - self.start_time

    @property
    def log_prefix(self):
        return f"[{self.unique_id} @{self.time_since_start()}] "


class JobQuerySortBy(str, Enum):
    NAME = 'Name',
    COMPLETED_ON = 'CompletedOn',
    CREATED_ON = 'CreatedOn',
    TAGS = 'Tags',
    JOB_STATE = 'JobState'


class JobQuerySortType(str, Enum):
    NONE = 'None',
    ASCENDING = 'Ascending',
    DESCENDING = 'Descending'


class JobState(str, Enum):
    UNDEFINED = 'Undefined',
    CREATED = 'Created',
    READY_FOR_PROCESSING = 'ReadyForProcessing',
    PENDING = 'Pending',
    PROCESSING = 'Processing',
    COMPLETED = 'Completed',
    ERROR = 'Error',
    CANCELED = 'Canceled',
    DATA_MISSING = 'DataMissing',
    MARKED_FOR_DELETION = 'MarkedForDeletion'
