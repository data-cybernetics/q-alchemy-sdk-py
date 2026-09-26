import logging
from pkgutil import extend_path

# The SDK keeps convenience exports in this package while other Q-Alchemy
# distributions contribute sibling subpackages (for example
# ``q_alchemy.visualization``).  Extending ``__path__`` preserves those
# separately installed subpackages in editable/source-tree environments, where
# they are not physically merged into the SDK package directory.
__path__ = extend_path(__path__, __name__)

LOG = logging.getLogger(__name__)

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
    CircuitCompressionConfig,
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
from .feasibility import (
    FeasibilityExecutionError,
    FeasibilityJob,
    FeasibilityParams,
    FeasibilityService,
)
from .circuit_compression import (
    CircuitCompressionExecutionError,
    CircuitCompressionJob,
    CircuitCompressionParams,
    CircuitCompressionReport,
    CircuitCompressionRequest,
    CircuitCompressionService,
)

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
