import unittest

import numpy as np
import openqasm3
from openqasm3.ast import Program

from q_alchemy.parser.qasm_pennylane import PennyLaneState
from q_alchemy.parser.qasm_pennylane import PennyLaneVisitor


def reverse_qubits(size, idx):
    return size - (idx + 1)


class TestPennyLaneQasmParser(unittest.TestCase):

    def setUp(self):
        # This method will be called before each test
        pass

    def tearDown(self):
        # This method will be called after each test
        pass

    def test_qasm_file(self):

        with open("data/test.qasm", "r") as f:
            qasm = f.read()
            program: Program = openqasm3.parse(qasm)

        self.assertTrue(program.version == "2.0", "Only OpenQASM 2.0 is supported currently.")

        ctx = PennyLaneState()
        v = PennyLaneVisitor()
        v.visit(program, ctx)

        expected_qubit_size = 10
        self.assertDictEqual(ctx.registers, {"q": expected_qubit_size})
        self.assertEquals(len(ctx.op_list), 216)

        # line 4 of file: u3(0.03086834008633827,0,0) q[0];
        line_of_file = 4
        self.assertListEqual(
            ctx.op_list[line_of_file - 4].wires.tolist(),
            [reverse_qubits(expected_qubit_size, 0)]
        )
        self.assertEqual(ctx.op_list[line_of_file - 4].name, "U3")
        self.assertListEqual(ctx.op_list[line_of_file - 4].parameters, [0.03086834008633827, 0, 0])

        # line 12 of file: cx q[4],q[1];
        line_of_file = 12
        self.assertListEqual(
            ctx.op_list[line_of_file - 4].wires.tolist(),
            [reverse_qubits(expected_qubit_size, 4), reverse_qubits(expected_qubit_size, 1)]
        )
        self.assertEqual(ctx.op_list[line_of_file - 4].name, "CNOT")
        self.assertListEqual(ctx.op_list[line_of_file - 4].parameters, [])

        # line 33 of file: u3(1.5432484759579168,-pi/2,-pi/2) q[6];
        line_of_file = 33
        self.assertListEqual(
            ctx.op_list[line_of_file - 4].wires.tolist(),
            [reverse_qubits(expected_qubit_size, 6)]
        )
        self.assertEqual(ctx.op_list[line_of_file - 4].name, "U3")
        self.assertListEqual(
            ctx.op_list[line_of_file - 4].parameters,
            [1.5432484759579168, np.round(-np.pi/2, 16), np.round(-np.pi/2, 16)]
        )

        # line 48 of file: cx q[1],q[2];
        line_of_file = 48
        self.assertListEqual(
            ctx.op_list[line_of_file - 4].wires.tolist(),
            [reverse_qubits(expected_qubit_size, 1), reverse_qubits(expected_qubit_size, 2)]
        )
        self.assertEqual(ctx.op_list[line_of_file - 4].name, "CNOT")
        self.assertListEqual(ctx.op_list[line_of_file - 4].parameters, [])

        # line 65 of file: u3(0.024139535223237255,-pi,0) q[8];
        line_of_file = 65
        self.assertListEqual(
            ctx.op_list[line_of_file - 4].wires.tolist(),
            [reverse_qubits(expected_qubit_size, 8)]
        )
        self.assertEqual(ctx.op_list[line_of_file - 4].name, "U3")
        self.assertListEqual(
            ctx.op_list[line_of_file - 4].parameters,
            [0.024139535223237255, np.round(-np.pi, 16), 0]
        )

        # line 88 of file: u3(1.453284517332715,0.05045692280109648,2.734876252368144) q[6];
        line_of_file = 88
        self.assertListEqual(
            ctx.op_list[line_of_file - 4].wires.tolist(),
            [reverse_qubits(expected_qubit_size, 6)]
        )
        self.assertEqual(ctx.op_list[line_of_file - 4].name, "U3")
        self.assertListEqual(
            ctx.op_list[line_of_file - 4].parameters,
            [1.453284517332715,0.05045692280109648,2.734876252368144]
        )

        # line 110 of file: cx q[9],q[7];
        line_of_file = 110
        self.assertListEqual(
            ctx.op_list[line_of_file - 4].wires.tolist(),
            [reverse_qubits(expected_qubit_size, 9), reverse_qubits(expected_qubit_size, 7)]
        )
        self.assertEqual(ctx.op_list[line_of_file - 4].name, "CNOT")
        self.assertListEqual(ctx.op_list[line_of_file - 4].parameters, [])

        # line 219 of file: u3(2.389499827477419,-pi,pi/2) q[9];
        line_of_file = 219
        self.assertListEqual(
            ctx.op_list[line_of_file - 4].wires.tolist(),
            [reverse_qubits(expected_qubit_size, 9)]
        )
        self.assertEqual(ctx.op_list[line_of_file - 4].name, "U3")
        self.assertListEqual(
            ctx.op_list[line_of_file - 4].parameters,
            [2.389499827477419, np.round(-np.pi, 16), np.round(np.pi/2, 16)]
        )


if __name__ == '__main__':
    unittest.main()
