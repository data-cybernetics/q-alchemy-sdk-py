import io
from enum import Enum
from io import TextIOBase
from typing import List, Tuple

import numpy as np
import openqasm3
from openqasm3 import properties
from openqasm3.ast import QuantumGate, Identifier, Program, Statement, QubitDeclaration, IntegerLiteral, FloatLiteral, IndexedIdentifier, Expression, UnaryExpression

import pennylane as qml

from pennylane.typing import TensorLike
from pennylane.wires import Wires
from pennylane.operation import Operation
from typing import Optional, Any, Sequence, Dict
from openqasm3.visitor import QASMVisitor
import openqasm3.ast as ast
import dataclasses


class ParsingState(Enum):
    DEFAULT = 0,
    PARAMS = 1,
    WIRES = 2,
    QUBIT_DECLARATION = 3,
    EXPRESSION = 4


@dataclasses.dataclass
class OperationState:
    name: str
    params: List[float|int] = dataclasses.field(default_factory=list)
    wires: List[Tuple[str, Wires]] = dataclasses.field(default_factory=list)
    id: Optional[str] = dataclasses.field(default=None)
    current_qubit: str = dataclasses.field(default=None)

    def to_operation(self, registers: Dict[str, int]):
        num_qubits = sum(registers.values())

        register_2_idx = []
        for r, s in registers.items():
            indices = range(len(register_2_idx), len(register_2_idx) + s)
            register_2_idx.append((r, list(indices)))
        register_2_idx = dict(register_2_idx)

        operation_wires = Wires([num_qubits - (register_2_idx[qb_name][qb_idx] + 1) for qb_name, qb_wires in self.wires for qb_idx in qb_wires])

        operation_parameters = self.params
        if self.name == 'rz':
            return qml.RZ(*operation_parameters, wires=operation_wires)
        elif self.name == 'cx':
            return qml.CNOT(wires=operation_wires)
        elif self.name == 'ry':
            return qml.RY(*operation_parameters, wires=operation_wires)
        elif self.name == 'rx':
            return qml.RX(*operation_parameters, wires=operation_wires)
        elif self.name == 'u1':
            return qml.U1(*operation_parameters, wires=operation_wires)
        elif self.name == 'u2':
            return qml.U2(*operation_parameters, wires=operation_wires)
        elif self.name == 'u3':
            return qml.U3(*operation_parameters, wires=operation_wires)
        else:
            raise ValueError(
                "Currently the parser can only do rx, ry, rz, u1, u2, u3, cx. Please reach out to the Q-Alchemy team."
            )

        pass


@dataclasses.dataclass
class PennyLaneState:
    registers: Dict[str, int] = dataclasses.field(default_factory=dict)
    op_list: List[Operation] = dataclasses.field(default_factory=list)


class PennyLaneVisitor(QASMVisitor[PennyLaneState]):
    stream: TextIOBase = None
    parsing_state: ParsingState = ParsingState.DEFAULT
    current_operation: OperationState | None = None

    def _start_recording(self):
        self.stream: io.TextIOBase = io.StringIO()

    def _end_recording(self):
        self.stream.seek(0)
        expression = self.stream.read()
        value = eval(expression)
        return value

    def _new_operation(self, name: Identifier):
        self.current_operation = OperationState(name.name)

    def _complete_operation(self, context: PennyLaneState):

        # Check if the

        context.op_list.append(self.current_operation.to_operation(context.registers))
        self.current_operation = None

    def _get_current_qubit_declaration(self, context: PennyLaneState) -> str | None:
        candidates = [r for r in context.registers if context.registers[r] is None]
        if len(candidates) == 0:
            return None
        elif len(candidates) > 1:
            raise AssertionError("Not expected more thanone candidate during qubit declaration!")
        else:
            return candidates[0]

    def visit(self, node: ast.QASMNode, context: Optional[PennyLaneState] = None) -> None:
        if context is None:
            context = PennyLaneState()
        return super().visit(node, context)

    def _visit_sequence(
            self,
            nodes: Sequence[ast.QASMNode],
            context: PennyLaneState
    ) -> None:
        for node in nodes:
            if self.parsing_state == ParsingState.PARAMS:
                self._start_recording()

            self.visit(node, context)

            if self.parsing_state == ParsingState.PARAMS:
                value = self._end_recording()
                self.current_operation.params.append(value)

    def visit_QuantumGate(self, node: ast.QuantumGate, context: PennyLaneState) -> None:
        self._new_operation(node.name)

        if node.arguments:
            self.parsing_state = ParsingState.PARAMS
            self._visit_sequence(node.arguments, context)

        self.parsing_state = ParsingState.WIRES
        self._visit_sequence(node.qubits, context)

        self._complete_operation(context)
        self.parsing_state = ParsingState.DEFAULT

    def visit_QubitDeclaration(self, node: ast.QubitDeclaration, context: PennyLaneState) -> None:
        self.parsing_state = ParsingState.QUBIT_DECLARATION
        context.registers[node.qubit.name] = None
        self.visit(node.size, context)

    def visit_Identifier(self, node: ast.Identifier, context: PennyLaneState) -> None:
        if self.parsing_state == ParsingState.PARAMS:
            if node.name == "pi":
                self.stream.write(str(np.pi))
            else:
                raise ValueError(f"We don't handle {node.name}. Please implement!")

    def visit_IndexedIdentifier(self, node: ast.IndexedIdentifier, context: PennyLaneState) -> None:
        if self.parsing_state == ParsingState.WIRES:
            self.current_operation.current_qubit = node.name.name
            self.current_operation.wires.append((node.name.name, Wires([])))
            for index in node.indices:
                if isinstance(index, ast.DiscreteSet):
                    self.visit(index, context)
                else:
                    self._visit_sequence(index, context)

    def visit_FloatLiteral(self, node: ast.FloatLiteral, context: PennyLaneState) -> None:
        if self.parsing_state == ParsingState.PARAMS:
            self.stream.write(str(node.value))

    def visit_IntegerLiteral(self, node: ast.IntegerLiteral, context: PennyLaneState) -> None:
        if self.parsing_state == ParsingState.PARAMS:
            self.stream.write(str(node.value))
        if self.parsing_state == ParsingState.WIRES:
            # Get the latest and then append to wire (second index)
            qubit, current_wires = self.current_operation.wires[-1]
            current_wires += [node.value]
            self.current_operation.wires[-1] = (qubit, current_wires)
        if self.parsing_state == ParsingState.QUBIT_DECLARATION:
            context.registers[self._get_current_qubit_declaration(context)] = node.value

    def visit_UnaryExpression(self, node: ast.UnaryExpression, context: PennyLaneState) -> None:
        self.stream.write(node.op.name)
        if properties.precedence(node) >= properties.precedence(node.expression):
            self.stream.write("(")
            self.visit(node.expression, context)
            self.stream.write(")")
        else:
            self.visit(node.expression, context)

    def visit_BinaryExpression(self, node: ast.BinaryExpression, context: PennyLaneState) -> None:
        our_precedence = properties.precedence(node)
        # All AST nodes that are built into BinaryExpression are currently left associative.
        if properties.precedence(node.lhs) < our_precedence:
            self.stream.write("(")
            self.visit(node.lhs, context)
            self.stream.write(")")
        else:
            self.visit(node.lhs, context)
        self.stream.write(f" {node.op.name} ")
        if properties.precedence(node.rhs) <= our_precedence:
            self.stream.write("(")
            self.visit(node.rhs, context)
            self.stream.write(")")
        else:
            self.visit(node.rhs, context)


def from_qasm(qasm: str) -> List[Operation]:
    program: Program = openqasm3.parse(qasm)
    ctx = PennyLaneState()
    v = PennyLaneVisitor()
    v.visit(program, ctx)
    return ctx.op_list
