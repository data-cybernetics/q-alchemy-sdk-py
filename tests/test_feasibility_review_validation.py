"""SDK wire validation does not require the private Feasibility package."""
from dataclasses import replace
import pytest
from q_alchemy.feasibility_contract import (
    ClassicalResources, EvidenceCollectionConfig, FeasibilityPolicy,
    FeasibilityRequest, FeasibilityReport, SolutionCriteria, Criterion,
)


@pytest.mark.parametrize('version', [True, 1.0, 1.9, '1', None])
def test_schema_versions_are_strict(version):
    request = FeasibilityRequest(criteria=SolutionCriteria.common(max_observable_rmse=.1))
    payload = request.to_dict(); payload['schema_version'] = version
    with pytest.raises(ValueError): FeasibilityRequest.from_dict(payload)
    with pytest.raises(ValueError): replace(request, schema_version=version)
    with pytest.raises(ValueError): FeasibilityReport.from_dict({'kind': 'feasibility-report', 'schema_version': version})


@pytest.mark.parametrize('value', ['false', 0, 1, None])
def test_policy_and_criterion_booleans_are_strict(value):
    with pytest.raises(ValueError): EvidenceCollectionConfig.from_dict({'least_busy': value})
    with pytest.raises(ValueError): FeasibilityPolicy.from_dict({'classical_first': value})
    with pytest.raises(ValueError): Criterion.from_dict({'metric': 'quality.test', 'threshold': 1, 'relation': '<=', 'required': value})


@pytest.mark.parametrize('field', ['available_memory_bytes', 'total_memory_bytes', 'cpu_cores', 'gpu_memory_bytes'])
@pytest.mark.parametrize('value', [True, -1, 1.5, '100'])
def test_resource_quantities_are_strict(field, value):
    with pytest.raises(ValueError): ClassicalResources.from_dict({field: value})
