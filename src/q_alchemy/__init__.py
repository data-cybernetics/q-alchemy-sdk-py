import logging
import os
from pkgutil import extend_path
import warnings

# The SDK keeps convenience exports in this package while other Q-Alchemy
# distributions contribute sibling subpackages (for example
# ``q_alchemy.visualization``).  Extending ``__path__`` preserves those
# separately installed subpackages in editable/source-tree environments, where
# they are not physically merged into the SDK package directory.
__path__ = extend_path(__path__, __name__)

LOG = logging.getLogger(__name__)

#: Set to a truthy value to let the pinexq-client / JobManagementAPI version
#: mismatch warning through. The acceptance-tests repo does exactly that.
SHOW_API_VERSION_WARNING_ENV = "Q_ALCHEMY_API_VERSION_WARNING"


def _silence_api_version_warning() -> None:
    """Stop pinexq-client nagging end users about *our* deployment's version skew.

    pinexq-client warns on every entry-point navigation when its protocol
    version differs from the deployed JobManagementAPI's. It fires several
    times per SDK call, and it describes a fact about our platform that the
    caller neither caused nor can fix — so the shipped package hides it.

    This is a process-wide filter, which a library normally has no business
    installing. It is done here deliberately, because the warning is raised
    inside pinexq-client's own internals (ProcessingStep, Job and WorkData all
    navigate the entry point themselves), so there is no call site of ours to
    wrap. The filter is kept as narrow as a filter can be — exact message
    prefix, exact category — and it is opt-out, not permanent: set
    ``Q_ALCHEMY_API_VERSION_WARNING=1`` and the warning comes back, which is
    how the acceptance tests keep the skew visible where it is actionable.
    """
    if os.getenv(SHOW_API_VERSION_WARNING_ENV, "").strip().lower() in ("1", "true", "yes"):
        return
    warnings.filterwarnings(
        "ignore",
        message=r"Version mismatch between 'pinexq_client'",
        category=UserWarning,
    )


_silence_api_version_warning()
from .initialize import q_alchemy_as_qasm
from .simulator import (
    SparseSimulator,
    SimulatorParams,
    CountsResult,
    SparseStatevectorResult,
    TomographyResult,
    simulate_counts,
    simulate_sparse_statevector,
    simulate_tomography,
)


from .quantum_io_contract import (
    SCHEMA_VERSION,
    BasisDistribution,
    BasisMeasurement,
    Circuit,
    CircuitMetrics,
    DistributionMetrics,
    ErrorMetrics,
    ExecutionPlan,
    ExecutionResult,
    ExperimentCircuitSummary,
    ExperimentReport,
    ExperimentSummary,
    MeasurementPlan,
    MeasurementPlanSummary,
    ObservablePlanSummary,
    Observation,
    ObservationSet,
    PauliObservable,
    PortablePauliSum,
    PreparationPreflightSummary,
    PreparationSummary,
    QuantumExperiment,
    ReferenceSummary,
    Runtime,
    SimulationSummary,
    State,
    StateEstimateSummary,
)

from .quantum_io import (
    LOCAL_SIMULATOR_RESOURCE,
    NOISY_BACKEND_SIMULATOR_RESOURCE,
    QUANTUM_BACKEND_RESOURCE,
    QuantumIOService,
    QuantumIOParams,
    QuantumIOJob,
    QuantumBackend,
    IBMQuantumCredentials,
    local_simulator_execution_plan,
    noisy_backend_execution_plan,
    quantum_backend_execution_plan,
)
from .feasibility_contract import (
    ClassicalResources,
    Criterion,
    EvidenceCollectionConfig,
    FeasibilityPolicy,
    FeasibilityRecommendation,
    FeasibilityStatus,
    FeasibilityReport,
    FeasibilityRequest,
    QuantumExecutionPolicy,
    RecommendedCompute,
    Relation,
    SolutionCriteria,
)
from .feasibility import FeasibilityJob, FeasibilityParams, FeasibilityService

try: # should fail silently if user has not installed optional dependencies
    from .qiskit_integration import QAlchemyInitialize
except ImportError:
    LOG.info("qiskit_integration module not available")
try:  # needs the qiskit extra
    from .qalchemy_backend import QAlchemyBackend, QAlchemyProvider, QAlchemyJob
except ImportError:
    LOG.info("qalchemy_backend module not available (needs qiskit)")
try:
    from .pennylane_integration import QAlchemyStatePreparation
except ImportError:
    LOG.info("pennylane_integration module not available")