"""Opt-in deployed-host replay using only public SDK calls.

Creates real jobs only with --run. Targets guide calibration; no runtime SLA is
asserted. The API key selects the organization and subscription. Failed/running
jobs follow the SDK's normal preservation policy.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter

from q_alchemy import CircuitCompressionParams, CircuitCompressionRequest, CircuitCompressionService
from q_alchemy.simulator import SimulatorParams, SparseSimulator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', help='Authorize real hosted jobs')
    parser.add_argument('--adapter', choices=('simulator', 'compressor', 'all'), default='all')
    parser.add_argument('--tier', choices=('standard', 'enterprise'), default='standard')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--simulator-version')
    parser.add_argument('--compressor-version')
    args = parser.parse_args()
    if not args.run:
        parser.error('--run is required to create real hosted jobs')
    if args.simulator_version:
        # Simulator has no public step-version option; use the SDK's ordinary
        # explicit-version lookup for this command only.
        import q_alchemy.simulator as module
        from q_alchemy.initialize import from_name
        module.find_processing_step = lambda client, name: from_name(client, name, version=args.simulator_version)
    from qiskit import QuantumCircuit
    results = []
    with SparseSimulator(SimulatorParams(tier=args.tier, job_completion_timeout_sec=1800)) as simulator:
        grants = simulator.user_grants()
        if 'plan:enterprise' in grants:
            plan = 'enterprise'
        elif 'plan:free' in grants:
            plan = 'free'
        else:
            parser.error('A recognized Free or Enterprise subscription is required')
        if args.tier == 'enterprise' and plan != 'enterprise':
            parser.error('The enterprise preset requires an Enterprise account')
        if args.adapter in ('simulator', 'all'):
            score = 25000 if plan == 'enterprise' else 2500
            circuit = QuantumCircuit(24, 1)
            circuit.h(range(20))
            for _ in range(score - 21):
                circuit.ry(.113, 20)
            circuit.measure(0, 0)
            started = perf_counter()
            result = simulator.counts(circuit, shots=1, optimization_level=0, input_form='qpy')
            metadata = result.raw['simulation_metadata']
            results.append({'adapter': 'simulator', 'plan': plan, 'tier': args.tier,
                            'score': result.raw['circuit_complexity'],
                            'rpc_seconds': perf_counter()-started,
                            'core_execution_seconds': metadata.get('adapter_execution_seconds'),
                            'max_nnz': metadata.get('max_nnz'), 'peak_nnz': metadata.get('peak_nnz'),
                            'capping_events': metadata.get('amplitude_capping_events'),
                            'reference_profile_max_nnz': 1048576,
                            'note': 'Counts uses core automatic memory sizing; the local reference used an explicit '
                                    '1,048,576-amplitude cap. One terminal measurement replaces one RY. '
                                    'Compare effective caps before comparing timings.'})
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(results, indent=2)+'\n')
        if args.adapter in ('compressor', 'all'):
            score = 100000 if plan == 'enterprise' else 10000
            circuit = QuantumCircuit(24)
            circuit.h(range(4))
            for index in range((score-4)//3):
                q = index % 4
                circuit.ccx(q, (q+1)%24, (q+2)%24)
            options = CircuitCompressionParams(api_key=simulator.params.api_key, host=simulator.params.host,
                                               step_version=args.compressor_version,
                                               job_completion_timeout_sec=1800)
            with CircuitCompressionService(options) as service:
                started = perf_counter()
                result = service.compress(circuit, CircuitCompressionRequest(
                    options={'collect_report': True})).result()
                results.append({'adapter': 'compressor', 'plan': plan,
                                'score': result.raw['circuit_complexity'],
                                'rpc_seconds': perf_counter()-started,
                                'core_report_seconds': (result.report or {}).get('total_seconds')})
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(results, indent=2)+'\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
