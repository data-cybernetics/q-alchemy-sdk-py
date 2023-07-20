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
from datetime import datetime
from enum import Enum
from typing import Optional, List

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
    """Maximum Fidelity Loss allowed by the algorithm. Default: 0.0 (none)."""
    strategy: str = Strategy.GREEDY.value
    """The strategy to find the best state preparation. Default: Greedy."""
    max_combination_size: int = 0
    """While finding new combinations in the recursive tree search, 
    this number defines the number of qubits to split maximally. E.g., if
    there are 7 qubits, the maximum of 3 qubits can be used to split. This 
    number, if given limits this generally. Default: 0 (no limit)."""
    use_low_rank: bool = False
    """Using low rank can produce even better results. Usually a full dis-entangling 
    is attempted (within the maximum fidelity loss), but if that fails, a low rank 
    split is tried too. Default: False."""
    load_per_cyle: int = 5000
    """How many sub-problems (splits) can be handled simultaneously. Default: 5000."""
    secede_modulo: int = 1
    """This programmatic setting defines how often does the computation secede to other
    computations. Sometimes the computation can be sped up if this is not at the default.
    Default: 1 (no secede)."""
    max_time_sec: int = 1200
    """The total computation time allowed for the computation in seconds. If this time is exceeded, 
    no new splits/sub-problems are generated, and the current computation is finished. 
    Default 1200 seconds."""
    max_loops: int = 200
    """How many loops will the algorithm do until it does not generate new splits? In each loop,
    load_per_cyle number of splits are undertaken. Default: 200."""
    max_nodes: int = 50000
    """How many total splits/sub-problems (these are computational nodes) are allowed maximally
    until the computation will no longer generate new splits. Default: 50000."""
    max_level: int = 2
    """How many levels in the search tree will be gone through. The more the better in terms of quality,
    but the longer the computation takes and the more splits will be traversed. Default: 2."""
    batch_size: int = 100
    """How big (in number of splits/sub-problems) is a computational job going to tackle in one go?
    Default: 100."""
    log_level: str = LogLevel.INFO.value
    """The log-level of the computation, in case it is necessary to debug the computation. Default: INFO."""
    redis_ttl_seconds: int = 3600
    """The splits/sub-problems are stored on an internal redis at most this number of seconds. Default: 3600."""
    tags: List[str] = []
    """The tags of the job are given here. These are important to finding the job once again in a query.
    Default: empty list."""


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
