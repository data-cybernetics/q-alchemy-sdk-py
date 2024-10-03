import logging

import httpx
import numpy as np
from pinexq_client.job_management import Job
from sklearn.datasets import fetch_openml
from sklearn.utils import Bunch

from q_alchemy.initialize import OptParams, create_client, upload_statevector

LOG = logging.getLogger(__name__)


def embed_mnist(mnist: Bunch):
    qubits = int(np.ceil(np.log2(mnist.data.shape[1])))
    filler = np.empty((mnist.data.shape[0], 2 ** qubits - mnist.data.shape[1]))
    filler.fill(0)
    embedded = np.hstack([mnist.data, filler])
    embedded = np.einsum("ij,i -> ij", embedded, 1 / np.linalg.norm(embedded, axis=1))
    all(np.linalg.norm(embedded, axis=1).round(13) == 1.0)
    return embedded, mnist.target


def full_cleanup(client: httpx.Client):
    from pinexq_client.job_management import enter_jma, Job
    from pinexq_client.job_management.model import JobQueryParameters, Pagination, WorkDataQueryParameters, \
        WorkDataFilterParameter, WorkDataKind
    from tqdm import tqdm

    workdata_list = (
        enter_jma(client)
        .work_data_root_link.navigate()
        .query_action.execute(
            WorkDataQueryParameters(
                filter=WorkDataFilterParameter(
                    is_kind=WorkDataKind.processing_artefact,
                    is_deletable=False
                ),
                pagination=Pagination(page_size=1000)
            )
        )
        .workdatas
    )
    for wd in tqdm(workdata_list):
        wd.allow_deletion_action.execute()

    jobs = (
        enter_jma(client)
        .job_root_link
        .navigate()
        .job_query_action
        .execute(JobQueryParameters(pagination=Pagination(page_size=1000)))
        .jobs
     )
    for job in tqdm(jobs):
        (
            Job.from_hco(client, job)
            .delete_with_associated(delete_input_workdata=True)
        )


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    opt_params = OptParams()
    client = create_client(opt_params)
    LOG.info("Loading MNIST...")
    X, y = embed_mnist(fetch_openml('mnist_784', version=1, parser="auto"))
    targets = set(y)
    for target in targets:
        data = X[y == target]
        internal_opt_params = opt_params.clone()
        internal_opt_params.job_tags += [f"Target={target}", "Dataset=MNIST784", f"Shape={data.shape}"]
        LOG.info(f"Uploading {internal_opt_params.job_tags}")
        wd_link = upload_statevector(
            client=client,
            state_vector=data,
            opt_params=internal_opt_params,
            override_filename=f"mnist784_target-{target}.bin"
        )
        for fl in np.linspace(0.0, 1.0, 20):
            LOG.info(f"Creating Tranformation Job for MNIST784: y={target}, fl={fl:.2f}, #{len(data)}...")
            job = (
                Job(client)
                .create(name=f"MNIST768-Target{target}-FL{fl:.2f}")
                .select_processing(function_name="qs_batch_to_qasm")
                .configure_parameters(basis_gates=["u", "cx"], min_fidelity=1 - fl)
                .set_tags([f"Target={target}", "Dataset=MNIST768", f"FL={fl:.2f}"])
                .assign_input_dataslot(0, work_data_link=wd_link)
            )
            LOG.info(f"Created Job")
