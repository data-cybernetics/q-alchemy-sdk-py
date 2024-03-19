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
import json as json_
from typing import Any, Self

import httpx
from httpx import URL


from hypermedia_client.core import Link, MediaTypes
from hypermedia_client.core.polling import wait_until, PollingException
from hypermedia_client.job_management.enterjma import enter_jma
from hypermedia_client.job_management.hcos import WorkDataLink
from hypermedia_client.job_management.hcos.entrypoint_hco import EntryPointHco
from hypermedia_client.job_management.hcos.job_hco import (
    JobHco,
    GenericProcessingConfigureParameters,
    JobLink,
)
from hypermedia_client.job_management.hcos.job_query_result_hco import JobQueryResultHco
from hypermedia_client.job_management.hcos.jobsroot_hco import JobsRootHco
from hypermedia_client.job_management.hcos.processingsteproot_hco import (
    ProcessingStepsRootHco,
)
from hypermedia_client.job_management.known_relations import Relations
from hypermedia_client.job_management.model import (
    CreateJobParameters,
    ProcessingStepQueryParameters,
    ProcessingStepFilterParameter,
    SelectProcessingParameters,
    JobStates,
    CreateSubJobParameters,
    JobQueryParameters,
    JobSortPropertiesSortParameter,
    JobFilterParameter,
    SelectWorkDataForDataSlotParameters,
)


class Job:
    """Convenience wrapper for handling JobHcos in the JobManagement-Api.

    This wrapper allows the API to be used with a fluent-style builder pattern:

    job = (
        Job(client)
        .create(name='JobName')
        .select_processing(processing_step='job_processing')
        .configure_parameters(**job_parameters)
        .start()
        .wait_for_state(JobStates.completed)
        .delete()
    )
    """

    _client: httpx.Client
    _entrypoint: EntryPointHco
    _jobs_root: JobsRootHco
    _job: JobHco | None = None
    _processing_step_root: ProcessingStepsRootHco

    def __init__(self, client: httpx.Client):
        """

        Args:
            client: An httpx.Client instance initialized with the api-host-url as `base_url`
        """
        self._client = client
        self._entrypoint = enter_jma(client)
        self._jobs_root = self._entrypoint.job_root_link.navigate()
        self._processing_step_root = (
            self._entrypoint.processing_step_root_link.navigate()
        )

    def create(self, name: str) -> Self:
        """
        Creates a new job by name.

        Args:
            name: Name of the job to be created

        Returns:
            The newly created job as `Job` object
        """
        job_link = self._jobs_root.create_job_action.execute(
            CreateJobParameters(name=name)
        )
        self._get_by_link(job_link)
        return self

    def _get_by_link(self, job_link: JobLink):
        self._job = job_link.navigate()

    @classmethod
    def from_url(cls, client: httpx.Client, job_url: URL) -> Self:
        """Initializes a `Job` object from an existing job given by its link as URL.

        Args:
            client: An httpx.Client instance initialized with the api-host-url as `base_url`
            job_url:

        Returns:
            The newly created job as `Job` object
        """
        link = Link.from_url(
            job_url,
            [str(Relations.CREATED_RESSOURCE)],
            "Created sub-job",
            MediaTypes.SIREN,
        )
        job_instance = cls(client)
        job_instance._get_by_link(JobLink.from_link(client, link))
        return job_instance

    def create_sub_job(self, name: str) -> "Job":
        """Create a new job by name as a sub-job of the current one.

        Args:
            name:
                Name of the job to be created
        Returns:
            The newly created job as `Job` object
        """
        parent_job_url = self._job.self_link.get_url()
        sub_job_link = self._jobs_root.create_subjob_action.execute(
            CreateSubJobParameters(name=name, parent_job_url=str(parent_job_url))
        )
        sub_job = Job(self._client)
        sub_job._get_by_link(sub_job_link)
        return sub_job

    def refresh(self) -> Self:
        """Updates the job from the server

        Returns:
            This `Job` object, but with updated properties.
        """
        self._job = self._job.self_link.navigate()
        return self

    def get_state(self) -> JobStates:
        """Returns the current state of this job from the server

        Returns:
            The current state of this `Job` from JobStates
        """
        self.refresh()
        return self._job.state

    def select_processing(self, processing_step: str) -> Self:
        """Set the processing step for this job given by name. This will query all
        processing steps of this name from the server and select the first result.

        Args:
            processing_step: Name of the processing step as string

        Returns:
            This `Job` object
        """
        # ToDo: provide more parameters to query a processing step
        query_param = ProcessingStepQueryParameters(
            filter=ProcessingStepFilterParameter(
                function_name_contains=processing_step,
            )
        )
        query_result = self._processing_step_root.query_action.execute(query_param)
        if len(query_result.processing_steps) == 0:
            raise AttributeError(f"No processing step with the name '{processing_step}' registered!")
        if len(query_result.processing_steps) > 1:
            raise AttributeError(f"Multiple results querying processing step '{processing_step}'!")
        assert len(query_result.processing_steps) == 1
        # Todo: For now we choose the first and only result. Make this more flexible?
        processing_url = query_result.processing_steps[0].self_link.get_url()

        self._job.select_processing_action.execute(
            SelectProcessingParameters(processing_step_url=str(processing_url))
        )

        self.refresh()

        return self

    def configure_parameters(self, **parameters: Any) -> Self:
        """Set the parameters to run the processing step with.

        Args:
            **parameters: Any keyword parameters provided will be forwarded as parameters
                to the processing step function.

        Returns:
            This `Job` object
        """
        self._job.configure_processing_action.execute(
            GenericProcessingConfigureParameters.model_validate(parameters)
        )

        self.refresh()
        return self

    def start(self) -> Self:
        """Start processing this job.

        Returns:
            This `Job` object
        """
        self._job.start_processing_action.execute()
        self.refresh()
        return self

    def get_result(self) -> Any:
        """Get the return value of the processing step after its completion.

        This value is not defined before completion, so check the state first or
        wait explicitly for it to complete.

        Returns:
            The result of the processing step
        """
        # TODO: return Sentinel or Exception on 'NotDoneYet'
        # TODO: handle return value equivalent to asyncio's Future objects
        self.refresh()
        result = self._job.result
        return json_.loads(result) if result else None


    def wait_for_state(self, state: JobStates, timeout_ms: int = 5000) -> Self:
        """Wait for this job to reach a state.

        Args:
            state: The state to wait for. After the job enters this state this function returns.
            timeout_ms: Time span in milliseconds to wait for reaching the state before
                raising an exception.

        Returns:
            This `Job` object
        """
        try:
            wait_until(
                condition=lambda: self.get_state() == state,
                timeout_ms=timeout_ms,
                timeout_message="Waiting for job completion",
                error_condition=lambda: self._job.state == JobStates.error,
            )
        except TimeoutError as timeout:
            raise Exception(
                f"Job did not reach state: '{state.value}' "
                f"current state: '{self.get_state().value}'. Error:{str(timeout)}"
            )
        except PollingException:
            if self._job.state == JobStates.error:
                error_reason = self._job.error_description
                raise Exception(f"Job failed'. Error:{error_reason}")
            raise Exception("Job failed")

        return self

    def assign_input_dataslot(self, index: int, workdata_link: WorkDataLink) -> Self:
        """Assign WorkData to DataSlots.

        Args:
            index: The numerical index of the dataslot.
            workdata_link:  WorkData given by its URL

        Returns:
            This `Job` object
        """
        dataslot = self._job.input_dataslots[index]
        dataslot.select_workdata_action.execute(
            parameters=SelectWorkDataForDataSlotParameters(
                work_data_url=str(workdata_link.get_url())
            )
        )
        self.refresh()

        return self

    def clear_input_dataslot(self, index: int) -> Self:
        """Clear the selected WorkData for a dataslot.

        Args:
            index: he numerical index of the dataslot.

        Returns:
            This `Job` object
        """
        dataslot = self._job.input_dataslots[index]

        # already cleared
        if not dataslot.clear_workdata_action:
            return

        dataslot.clear_workdata_action.execute()
        self.refresh()

        return self

    def _get_sub_jobs(
        self,
        sort_by: JobSortPropertiesSortParameter | None = None,
        state: JobStates | None = None,
        name: str | None = None,
        show_deleted: bool | None = None,
        processing_step_url: str | None = None,
    ) -> JobQueryResultHco:
        filter_param = JobFilterParameter(
            is_sub_job=True,
            parent_job_url=str(self._job.self_link.get_url()),
            state=state,
            name=name,
            show_deleted=show_deleted,
            processing_step_url=processing_step_url,
        )
        query_param = JobQueryParameters(sort_by=sort_by, filter=filter_param)
        job_query_result = self._jobs_root.job_query_action.execute(query_param)
        return job_query_result

    def get_sub_jobs(self, **tbd):
        # todo: Query result iterator to go through paginated result
        raise NotImplementedError

    def sub_jobs_in_state(self, state: JobStates) -> int:
        """Query how many sub-job are in a specific state.

        Args:
            state: Job state as `JobStates` enum.

        Returns:
            The number of sub-jobs in the requested state.
        """
        query_result = self._get_sub_jobs(state=state)
        return query_result.total_entities

    def wait_for_sub_jobs_complete(self, timeout_ms: int = 0) -> Self:
        """Wait for all sub-jobs to reach the state 'completed'.

        This function will block execution until the state is reached or raise an exception
        if the operation timed out or a sub-job returned an error.

        Args:
            timeout_ms: Timeout to wait for the sub-jobs to reach the next state.

        Returns:
            This `Job` object
        """
        wait_until(
            condition=lambda: self.sub_jobs_in_state(JobStates.pending) == 0,
            timeout_ms=timeout_ms,
            timeout_message=f"Timeout while waiting for sub-jobs to complete! [timeout: {timeout_ms}ms]",
        )
        wait_until(
            condition=lambda: self.sub_jobs_in_state(JobStates.processing) == 0,
            timeout_ms=timeout_ms,
            timeout_message=f"Timeout while waiting for sub-jobs to complete! [timeout: {timeout_ms}ms]",
        )
        wait_until(
            condition=lambda: self.sub_jobs_in_state(JobStates.completed) >= 0,
            error_condition=lambda: self.sub_jobs_in_state(JobStates.error) >= 0,
            error_condition_message="One or more sub-jobs returned an error!",
            timeout_ms=timeout_ms,
            timeout_message=f"Timeout while waiting for sub-jobs to complete! [timeout: {timeout_ms}ms]",
        )
        return self

    def hide(self) -> Self:
        """Mark this job as hidden.

        Returns:
            This `Job` object
        """
        self._job.hide_action.execute()
        self.refresh()
        return self

    def unhide(self):
        """Reveal this job again.

        Returns:
            This `Job` object"""
        self._job.unhide_action.execute()
        self.refresh()
        return self

    def allow_output_data_deletion(self):
        """Mark all output workdata from this job as "deletable".

        Returns:
            This `Job` object"""
        self._job.allow_output_data_deletion_action.execute()
        self.refresh()
        return self

    def disallow_output_data_deletion(self):
        """Mark all output workdata from this job as "not deletable".

        Returns:
            This `Job` object"""
        self._job.disallow_output_data_deletion_action.execute()
        self.refresh()
        return self