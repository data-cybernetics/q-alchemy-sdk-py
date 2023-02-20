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
import logging
import warnings
from datetime import datetime
from io import BytesIO
from typing import Optional, List, Union

import dateutil.parser
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from qclib.state_preparation import LowRankInitialize
from qclib.state_preparation.util.baa import Node
from qiskit import QuantumCircuit
from retry import retry

from baa_sdk.models import JobConfig, RenameJob, JobState, JobQuerySortBy, JobQuerySortType

LOG = logging.getLogger(__name__)

_entry_point_path = "/api/EntryPoint"


class RawClient:
    added_headers: Optional[dict]
    schema: str
    api_key: str
    host: str

    def __init__(self, api_key: str, host: str, schema: str = "https", added_headers: Optional[dict] = None) -> None:
        self.schema = schema
        self.api_key = api_key
        self.host = host
        self.added_headers = added_headers

    def get_header(self):
        header = {
            "X-Api-Key": self.api_key,
            "Accept": "application/vnd.siren+json"
        }
        if self.added_headers is not None:
            header.update(self.added_headers)
        return header

    def get_url(self, path: str):
        url = f"{self.schema}://{self.host}/{path}"
        return url

    def get_document(self, url: str):
        return _get_document(self, url)


class Document:
    json: dict
    client: RawClient

    def __init__(self, client: RawClient, json: dict):
        self.json = json
        self.client = client

    def _get_section(self, section: str, name: str, all_matches=False) -> Optional[Union[dict, List[dict]]]:
        section = self.json.get(section)
        if section is None:
            return None

        possible_sections = [
            a for a in section
            if ("name" in a and a["name"].lower() == name.lower()) or
               ("rel" in a and name.lower() in [r.lower() for r in a["rel"]])
        ]
        if all_matches:
            return possible_sections
        if len(possible_sections) == 0:
            return None
        return possible_sections[0]

    def _get_property(self, name: str) -> Optional[Union[str, dict]]:
        p = self.json.get("properties")
        if p is None:
            return None
        return self.json["properties"].get(name)

    def _get_action(self, name: str):
        return self._get_section("actions", name)

    def _get_link(self, rel: str) -> dict:
        return self._get_section("links", rel)

    def _get_entity(self, rel: str) -> dict:
        return self._get_section("entities", rel)

    def _get_entities(self, rel: str):
        return self._get_section("entities", rel, True)

    def get_self_url(self) -> Optional[str]:
        return self._get_link("self").get("href")

    def update(self):
        job_doc = self.client.get_document(self.get_self_url())
        self.json = job_doc.json
        return self


class Config(Document):
    def __init__(self, client: RawClient, json: dict):
        super().__init__(client, json)

    def upload(self, config: JobConfig):
        action = self._get_action("Upload")
        if action is None:
            raise ValueError("Expected Actions 'Rename' not found!")
        response: requests.Response = requests.request(
            method=action["method"], url=action["href"], headers=self.client.get_header(),
            json=config.dict()
        )
        if response.status_code == 200:
            self.update()
            return True
        else:
            message = f"Upload has failed (code {response.status_code}) with error '{response.text}'"
            print(message)
            warnings.warn(message=message, category=UserWarning)
            return False

    def delete(self):
        action = self._get_action("Delete")
        if action is None:
            warnings.warn("Expected Actions 'Ready' not available!", UserWarning)
            return False
        response: requests.Response = requests.request(
            method=action["method"], url=action["href"], headers=self.client.get_header()
        )
        if response.status_code == 200:
            job_doc = self.client.get_document(self.get_self_url())
            self.json = job_doc.json
            return True
        else:
            message = f"Delete has failed (code {response.status_code}) with error '{response.text}'"
            warnings.warn(message=message, category=UserWarning)
            return False

    def job_config(self) -> Optional[JobConfig]:
        config = self._get_property("Config")
        if config is None:
            return None
        import re
        config_snake_case = dict([(re.sub(r'(?<!^)(?=[A-Z])', '_', k).lower(), v) for k, v in config.items()])
        return JobConfig(**config_snake_case)

    def update(self) -> 'Config':
        super().update()
        return self


class StateVector(Document):
    def __init__(self, client: RawClient, json: dict):
        super().__init__(client, json)

    def update(self) -> 'StateVector':
        super().update()
        return self

    @property
    def upload_link_expiry_date(self) -> datetime:
        v = self._get_property("LinkExpiryDate")
        return dateutil.parser.parse(v)

    def get_vector(self) -> Optional[np.ndarray]:
        entity = self._get_entity("state_vector")
        if entity is None or "href" not in entity:
            warnings.warn("Entity 'StateVector' not available!", UserWarning)
            return None
        response: requests.Response = requests.get(entity["href"])
        if response.status_code == 200:
            data = BytesIO(response.content)
            table: pa.Table = pq.read_table(data)
            pa_state_vector_real: pa.Array = table["real_amplitudes"]
            pa_state_vector_imag: pa.Array = table["imag_amplitudes"]
            vector = pa_state_vector_real.to_numpy() + 1.0j * pa_state_vector_imag.to_numpy()
            return vector
        else:
            message = f"Delete has failed (code {response.status_code}) with error '{response.text}'"
            warnings.warn(message=message, category=UserWarning)
            return None

    def delete_vector(self, state_vector: np.ndarray):
        action = self._get_action("Delete")
        if action is None:
            warnings.warn("Expected Actions 'Ready' not available!", UserWarning)
            return False
        response: requests.Response = requests.request(
            method=action["method"], url=action["href"], headers=self.client.get_header()
        )
        if response.status_code == 200:
            job_doc = self.client.get_document(self.get_self_url())
            self.json = job_doc.json
            return True
        else:
            message = f"Delete has failed (code {response.status_code}) with error '{response.text}'"
            warnings.warn(message=message, category=UserWarning)
            return False

    def upload_vector(self, state_vector: np.ndarray):
        action = self._get_action("Upload")
        if action is None:
            raise ValueError("Expected Actions 'Upload' not found!")

        pa_state_vector_real = pa.array(np.real(state_vector))
        pa_state_vector_imag = pa.array(np.imag(state_vector))
        pa_basis = pa.array(np.arange(0, state_vector.shape[0]))
        table = pa.Table.from_arrays(
            arrays=[pa_basis, pa_state_vector_real, pa_state_vector_imag],
            names=["basis", "real_amplitudes", "imag_amplitudes"],
            metadata={
                "basis": "Computational Basis",
                "real_amplitudes": "Real Part of the Amplitudes",
                "imag_amplitudes": "imaginary Part of the Amplitudes"
            }
        )
        bytes_io = BytesIO()
        pq.write_table(table, bytes_io)
        bytes_io.seek(0)

        response: requests.Response = requests.request(
            method=action["method"], url=action["href"], headers=self.client.get_header(),
            data=bytes_io.read()
        )
        if response.status_code == 200:
            job_doc = self.client.get_document(self.get_self_url())
            self.json = job_doc.json
            return True
        else:
            message = f"Upload has failed (code {response.status_code}) with error '{response.text}'"
            print(message)
            warnings.warn(message=message, category=UserWarning)
            return False


class ResultNode(Document):
    def __init__(self, client: RawClient, json: dict):
        super().__init__(client, json)

    def update(self) -> 'ResultNode':
        super().update()
        return self

    @property
    def creation_date(self) -> Optional[datetime]:
        d = self._get_property("CreationDate")
        if d is not None:
            return dateutil.parser.parse(d)
        return None

    @property
    def link_expiry_date(self) -> Optional[datetime]:
        d = self._get_property("LinkExpiryDate")
        if d is not None:
            return dateutil.parser.parse(d)
        return None

    @property
    def qubits(self) -> Optional[List[List[int]]]:
        result_summary = self._get_property("ResultNodeData")
        if result_summary is not None and "Qubits" in result_summary:
            return result_summary["Qubits"]
        return None

    @property
    def partitions(self) -> Optional[list]:
        result_summary = self._get_property("ResultNodeData")
        if result_summary is not None and "Partitions" in result_summary:
            return result_summary["Partitions"]
        return None

    @property
    def ranks(self) -> Optional[List[int]]:
        result_summary = self._get_property("ResultNodeData")
        if result_summary is not None and "Ranks" in result_summary:
            return result_summary["Ranks"]
        return None

    @property
    def total_fidelity_loss(self) -> Optional[float]:
        result_summary = self._get_property("ResultNodeData")
        if result_summary is not None and "TotalFidelityLoss" in result_summary:
            return result_summary["TotalFidelityLoss"]
        return None

    @property
    def node_fidelity_loss(self) -> Optional[float]:
        result_summary = self._get_property("ResultNodeData")
        if result_summary is not None and "NodeFidelityLoss" in result_summary:
            return result_summary["NodeFidelityLoss"]
        return None

    @property
    def total_saved_cnots(self) -> Optional[int]:
        result_summary = self._get_property("ResultNodeData")
        if result_summary is not None and "TotalSavedCnots" in result_summary:
            return result_summary["TotalSavedCnots"]
        return None

    @property
    def node_saved_cnots(self) -> Optional[int]:
        result_summary = self._get_property("ResultNodeData")
        if result_summary is not None and "NodeSavedCnots" in result_summary:
            return result_summary["NodeSavedCnots"]
        return None

    @property
    def is_final(self) -> Optional[bool]:
        result_summary = self._get_property("ResultNodeData")
        if result_summary is not None and "IsFinal" in result_summary:
            return result_summary["IsFinal"]
        return None

    @property
    def num_qubits(self) -> Optional[int]:
        result_summary = self._get_property("ResultNodeData")
        if result_summary is not None and "NumQubits" in result_summary:
            return result_summary["NumQubits"]
        return None

    @property
    def num_vectors(self) -> Optional[int]:
        result_summary = self._get_property("ResultNodeData")
        if result_summary is not None and "Qubits" in result_summary:
            return len(result_summary["Qubits"])
        return None

    def get_node_vector(self, idx: int) -> Optional[np.ndarray]:
        node_link = self._get_link(f"node_vector[{idx}]")
        if node_link is not None and "href" in node_link:
            response: requests.Response = requests.get(node_link["href"])
            data = BytesIO(response.content)
            table: pa.Table = pq.read_table(data)
            pa_state_vector_real: pa.Array = table["real_amplitudes"]
            pa_state_vector_imag: pa.Array = table["imag_amplitudes"]
            vector = pa_state_vector_real.to_numpy() + 1.0j * pa_state_vector_imag.to_numpy()
            return vector
        return None

    def get_vectors(self) -> List[Optional[np.ndarray]]:
        return [self.get_node_vector(idx) for idx in range(self.num_vectors)]

    def to_node(self) -> Optional[Node]:
        vectors = [list(v) for v in self.get_vectors() if v is not None]
        if len(vectors) != self.num_vectors:
            return None
        return Node(
            node_saved_cnots=self.node_saved_cnots,
            total_saved_cnots=self.total_saved_cnots,
            node_fidelity_loss=self.node_fidelity_loss,
            total_fidelity_loss=self.total_fidelity_loss,
            vectors=vectors,
            qubits=self.qubits,
            ranks=self.ranks,
            partitions=self.partitions,
            nodes=[]
        )

    def to_circuit(self, opt_params=None):
        """
        Return a qiskit circuit from this node.

        opt_params: Dictionary
            isometry_scheme: string
                Scheme used to decompose isometries.
                Possible values are ``'knill'`` and ``'ccd'`` (column-by-column decomposition).
                Default is ``isometry_scheme='ccd'``.

            unitary_scheme: string
                Scheme used to decompose unitaries.
                Possible values are ``'csd'`` (cosine-sine decomposition) and ``'qsd'`` (quantum
                Shannon decomposition).
                Default is ``unitary_scheme='qsd'``.
        :return: the circuit to create the state
        """
        opt_params = {} if opt_params is None else opt_params
        circuit = QuantumCircuit(self.num_qubits)

        vector: np.ndarray
        for vector, qubits, rank, partition in zip(
                self.get_vectors(), self.qubits, self.ranks, self.partitions
        ):
            opt_params = {
                "iso_scheme": opt_params.get("isometry_scheme"),
                "unitary_scheme": opt_params.get("unitary_scheme"),
                "partition": partition,
                "lr": rank,
            }

            gate = LowRankInitialize(list(vector), opt_params=opt_params)
            circuit.compose(gate, qubits[::-1], inplace=True)  # qiskit little-endian.

        return circuit.reverse_bits()


class Result(Document):
    def __init__(self, client: RawClient, json: dict):
        super().__init__(client, json)

    def update(self) -> 'Result':
        super().update()
        return self

    def get_best_node(self) -> Optional[ResultNode]:
        best_node = self._get_link("BestNode")
        if best_node is not None and "href" in best_node:
            doc = self.client.get_document(best_node["href"])
            return ResultNode(doc.client, doc.json)
        return None

    def get_result_evolution(self) -> Optional[List[dict]]:
        result_summary = self._get_property("ResultSummary")
        if result_summary is not None and "ResultEvolution" in result_summary:
            evolution = result_summary["ResultEvolution"]
            return evolution
        return None

    def get_final_qubits(self) -> Optional[List[List[int]]]:
        result_summary = self._get_property("ResultSummary")
        if result_summary is not None and "FinalQubits" in result_summary:
            return result_summary["FinalQubits"]
        return None

    def get_final_partition(self) -> Optional[list]:
        result_summary = self._get_property("ResultSummary")
        if result_summary is not None and "FinalPartitions" in result_summary:
            return result_summary["FinalPartitions"]
        return None

    def get_final_ranks(self) -> Optional[List[int]]:
        result_summary = self._get_property("ResultSummary")
        if result_summary is not None and "Ranks" in result_summary:
            return result_summary["Ranks"]
        return None

    def get_total_fidelity_loss(self) -> Optional[int]:
        result_summary = self._get_property("ResultSummary")
        if result_summary is not None and "TotalFidelityLoss" in result_summary:
            return result_summary["TotalFidelityLoss"]
        return None

    def get_total_saved_cnots(self) -> Optional[List[int]]:
        result_summary = self._get_property("ResultSummary")
        if result_summary is not None and "TotalSavedCnots" in result_summary:
            return result_summary["TotalSavedCnots"]
        return None

    def get_result_nodes(self) -> List[ResultNode]:
        entities = self._get_entities("child")
        docs = [ResultNode(self.client, self.client.get_document(d["href"]).json) for d in entities if "href" in d]
        return docs


class Job(Document):
    def __init__(self, client: RawClient, json: dict):
        super().__init__(client, json)

    def update(self) -> 'Job':
        super().update()
        return self

    @property
    def has_stopped(self):
        return self.state in ["Completed", "Error", "Canceled"]

    @property
    def has_not_started(self):
        return self.state in ["Undefined", "Created", "ReadyForProcessing", "DataMissing"]

    @property
    def has_started(self):
        return self.state in ["Pending", "Processing"]

    @property
    def has_error(self):
        return self.state in ["Error", "Canceled"]

    @property
    def is_ready_to_start(self):
        return self.state in ["ReadyForProcessing"]

    @property
    def is_success(self):
        return self.state in ["Completed"]

    @property
    def has_data_missing(self):
        return self.state in ["Created", "DataMissing"]

    @property
    def name(self) -> Optional[str]:
        return self._get_property("Name")

    @property
    def state(self) -> Optional[str]:
        return self._get_property("State")

    @property
    def created_on(self) -> Optional[datetime]:
        raw = self._get_property("CreatedOn")
        return dateutil.parser.parse(raw)

    @property
    def created_by_id(self) -> Optional[datetime]:
        raw = self._get_property("CreatedById")
        return dateutil.parser.parse(raw)

    @property
    def context(self) -> Optional[dict]:
        raw = self._get_property("Context")
        return raw

    @property
    def error(self) -> Optional[dict]:
        raw = self._get_property("Error")
        return raw

    def get_config(self) -> Optional[Config]:
        c = self._get_entity("config")
        if c is None or "href" not in c:
            return None
        d = self.client.get_document(c["href"])
        return Config(d.client, d.json)

    def get_state_vector(self) -> Optional[StateVector]:
        c = self._get_entity("StateVector")
        if c is None or "href" not in c:
            return None
        d = self.client.get_document(c["href"])
        return StateVector(d.client, d.json)

    def get_result(self) -> Optional[Result]:
        c = self._get_entity("Result")
        if c is None or "href" not in c:
            return None
        d = self.client.get_document(c["href"])
        return Result(d.client, d.json)

    def rename(self, new_name) -> bool:
        action = self._get_action("Rename")
        if action is None:
            raise ValueError("Expected Actions 'Rename' not found!")
        data = RenameJob(new_name=new_name)
        response: requests.Response = requests.request(
            method=action["method"], url=action["href"], headers=self.client.get_header(),
            data=data.json()
        )
        if response.status_code == 202:
            job_doc = self.client.get_document(self.get_self_url())
            self.json = job_doc.json
            return True
        else:
            warnings.warn(f"Rename has failed (code {response.status_code}) with {response.text}", UserWarning)
            return False

    def schedule(self) -> bool:
        action = self._get_action("Ready")
        if action is None:
            warnings.warn("Expected Actions 'Ready' not available!", UserWarning)
            return False
        response: requests.Response = requests.request(
            method=action["method"], url=action["href"], headers=self.client.get_header()
        )
        if response.status_code == 202:
            self.update()
            return True
        else:
            warnings.warn(f"Scheduling the Job has failed (code {response.status_code}) with {response.text}", UserWarning)
            return False

    def delete(self) -> bool:
        action = self._get_action("Delete")
        if action is None:
            warnings.warn("Expected Actions 'Delete' not available!", UserWarning)
            return False
        response: requests.Response = requests.request(
            method=action["method"], url=action["href"], headers=self.client.get_header()
        )
        if response.status_code == 200:
            self.update()
            return True
        else:
            warnings.warn(f"Delete has failed (code {response.status_code}) with {response.text}", UserWarning)
            return False

    def cancel(self) -> bool:
        action = self._get_action("Cancel")
        if action is None:
            warnings.warn("Expected Actions 'Cancel' not available!", UserWarning)
            return False
        response: requests.Response = requests.request(
            method=action["method"], url=action["href"], headers=self.client.get_header()
        )
        if response.status_code == 200:
            self.update()
            return True
        else:
            warnings.warn(f"Cancel has failed (code {response.status_code}) with {response.text}", UserWarning)
            return False


class JobQueryResult(Document):

    def __init__(self, client: RawClient, json: dict):
        super().__init__(client, json)

    def update(self) -> 'JobQueryResult':
        super().update()
        return self

    @property
    def total_entities(self) -> Optional[int]:
        return self._get_property("TotalEntities")

    @property
    def current_entities_count(self) -> Optional[int]:
        return self._get_property("CurrentEntitiesCount")

    @property
    def jobs(self) -> List[Job]:
        return [Job(self.client, e) for e in self._get_entities("item")]

    def first(self) -> Optional['JobQueryResult']:
        linked_doc = self._get_link("first")
        if linked_doc is not None and "href" in linked_doc:
            doc = self.client.get_document(linked_doc["href"])
            return JobQueryResult(doc.client, doc.json)
        return None

    def next(self) -> Optional['JobQueryResult']:
        linked_doc = self._get_link("next")
        if linked_doc is not None and "href" in linked_doc:
            doc = self.client.get_document(linked_doc["href"])
            return JobQueryResult(doc.client, doc.json)
        return None

    def last(self) -> Optional['JobQueryResult']:
        linked_doc = self._get_link("last")
        if linked_doc is not None and "href" in linked_doc:
            doc = self.client.get_document(linked_doc["href"])
            return JobQueryResult(doc.client, doc.json)
        return None

    def delete_all(self) -> int:
        """
        Delete all jobs in this query.

        :return: deletes that weren't successful
        """
        count = len(self.jobs)
        executed_count = sum([1 if job.delete() else 0 for job in self.jobs])
        return count - executed_count

    def cancel_all(self) -> int:
        """
        Cancel all jobs in this query.

        :return: cancels that weren't successful
        """
        count = len(self.jobs)
        executed_count = sum([1 if job.cancel() else 0 for job in self.jobs])
        return count - executed_count

    def schedule_all(self) -> int:
        """
        Schedule all jobs in this query.

        :return: schedules that weren't successful
        """
        count = len(self.jobs)
        executed_count = sum([1 if job.schedule() else 0 for job in self.jobs])
        return count - executed_count

    def __repr__(self):
        return f"Query {len(self.jobs)}/{self.total_entities}"

class JobsRoot(Document):

    def __init__(self, client: RawClient, json: dict):
        super().__init__(client, json)

    def update(self) -> 'JobsRoot':
        super().update()
        return self

    def create_job(self) -> Job:
        action = self._get_action("CreateJob")
        response: requests.Response = requests.request(
            method=action["method"], url=action["href"], headers=self.client.get_header()
        )
        if response.status_code == 201:
            location = response.headers.get("Location")
            if location is None:
                raise IOError("Job was not created for an unknown reason!")
            document: Document = self.client.get_document(location)
            return Job(document.client, document.json)
        raise IOError(response.text)

    def job_query(self, page_size: int = 20, page_offset: int = 0,
                  sort_by: JobQuerySortBy = JobQuerySortBy.CREATED_ON,
                  sort_type: JobQuerySortType = JobQuerySortType.DESCENDING,
                  tags_by_and: Optional[List[str]] = None, name_contains: Optional[str] = None,
                  states_by_or: Optional[List[JobState]] = None) -> Optional[JobQueryResult]:
        action = self._get_action("CreateJobQuery")
        if action is not None:
            data = {
                "Pagination": {
                    "PageSize": page_size,
                    "PageOffset": page_offset
                },
                "SortBy": {
                    "PropertyName": sort_by,
                    "SortType": sort_type.value
                },
                "Filter": {
                    "StatesByOr": states_by_or or [],
                    "TagsByAnd": tags_by_and or [],
                    "NameContains": name_contains
                }
            }
            response = requests.request(
                action["method"], action["href"], headers=self.client.get_header(), json=data
            )
            if response.status_code == 201 and "Location" in response.headers:
                location = response.headers["Location"]
                doc = self.client.get_document(location)
                return JobQueryResult(doc.client, doc.json)
            print(response.status_code, response.reason)
            return None

    def get_all_jobs(self) -> Optional[JobQueryResult]:
        all_jobs = self._get_link("All")
        if all_jobs is not None and"href" in all_jobs:
            doc = self.client.get_document(all_jobs["href"])
            return JobQueryResult(doc.client, doc.json)
        return None

    def get_latest_job(self) -> Optional[Job]:
        link_section = self._get_link("Latest")
        if link_section is None:
            return None
        link = link_section.get("href")
        if link is None:
            return None
        document = self.client.get_document(link)
        return Job(document.client, document.json)


class Client(RawClient):

    def __init__(self, api_key: str, host: str, schema: str = "https", added_headers: Optional[dict] = None) -> None:
        super().__init__(api_key, host, schema, added_headers)

    def get_jobs_root(self) -> JobsRoot:
        entry_point = self.get_document(self.get_url("api/EntryPoint"))
        jobs = entry_point._get_link("Jobs")
        raw_document = self.get_document(jobs["href"])
        return JobsRoot(raw_document.client, raw_document.json)


@retry(tries=100, delay=0.5, max_delay=9, backoff=1.25,logger=LOG)
def _get_document(client: RawClient, url: str) -> Document:
    response: requests.Response = requests.get(url, headers=client.get_header())
    if response.status_code == 200:
        return Document(client, response.json())
    raise IOError(response.text)
