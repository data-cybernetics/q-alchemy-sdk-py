"""Compress a complete circuit remotely; requires Q_ALCHEMY_API_KEY.

Install q-alchemy-sdk-py[qiskit]. For arbitrary-input subroutines set equivalence
to 'operator'. The default assumes that all circuit inputs start in |0...0>.
"""

from qiskit import QuantumCircuit

from q_alchemy import CircuitCompressionRequest, CircuitCompressionService


def main() -> None:
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.ry(0.37, 1)
    circuit.cx(0, 1)
    circuit.rz(0.19, 0)

    request = CircuitCompressionRequest(options={"collect_report": True})
    with CircuitCompressionService() as service:
        report = service.compress(circuit, request).result()
    print(report.format_summary())
    print(report.to_qiskit().draw(output="text"))


if __name__ == "__main__":
    main()
