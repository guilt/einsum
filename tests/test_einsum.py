import unittest

import numpy as np

from einsum import Einsum, GreedyOrderOptimizer


class TestEinsum(unittest.TestCase):
    """Comprehensive test suite for the Einsum implementation."""

    # Basic Operations Tests
    def testMatrixMultiplication(self):
        """Test basic matrix multiplication ij,jk->ik."""
        a = np.ones((2, 3))
        b = np.ones((3, 4))
        einsumOp = Einsum("ij,jk->ik", a, b)
        result = einsumOp.execute()
        expected = np.einsum("ij,jk->ik", a, b)
        self.assertTrue(np.allclose(result, expected))
        self.assertEqual(result.shape, (2, 4))

    def testChainMatmul(self):
        """Test standard chain multiplication ij,jk,kl->il."""
        a = np.ones((2, 3))
        b = np.ones((3, 4))
        c = np.ones((4, 5))
        einsumOp = Einsum("ij,jk,kl->il", a, b, c)
        result = einsumOp.execute()
        expected = np.einsum("ij,jk,kl->il", a, b, c)
        self.assertTrue(np.allclose(result, expected))
        self.assertEqual(result.shape, (2, 5))

    def testBatchMatmul(self):
        """Test batch matrix multiplication bij,bjk->bik."""
        a = np.random.rand(5, 2, 3)
        b = np.random.rand(5, 3, 4)
        einsumOp = Einsum("bij,bjk->bik", a, b)
        result = einsumOp.execute()
        expected = np.einsum("bij,bjk->bik", a, b)
        self.assertTrue(np.allclose(result, expected))
        self.assertEqual(result.shape, (5, 2, 4))

    def testOuterProduct(self):
        """Test outer product i,j->ij."""
        a = np.arange(2)
        b = np.arange(3)
        einsumOp = Einsum("i,j->ij", a, b)
        result = einsumOp.execute()
        expected = np.einsum("i,j->ij", a, b)
        self.assertTrue(np.allclose(result, expected))
        self.assertEqual(result.shape, (2, 3))

    def testTranspose(self):
        """Test transpose operation ij->ji."""
        a = np.arange(6).reshape(2, 3)
        einsumOp = Einsum("ij->ji", a)
        result = einsumOp.execute()
        expected = np.einsum("ij->ji", a)
        self.assertTrue(np.allclose(result, expected))
        self.assertEqual(result.shape, (3, 2))

    def testSumAllElements(self):
        """Test summing all elements ij->."""
        a = np.arange(6).reshape(2, 3)
        einsumOp = Einsum("ij->", a)
        result = einsumOp.execute()
        expected = np.einsum("ij->", a)
        self.assertTrue(np.allclose(result, expected))
        self.assertEqual(result.shape, ())

    def testTrace(self):
        """Test trace operation ii->."""
        a = np.random.rand(3, 3)
        einsumOp = Einsum("ii->", a)
        result = einsumOp.execute()
        expected = np.einsum("ii->", a)
        self.assertTrue(np.allclose(result, expected))

    # Implicit Notation Tests
    def testImplicitNotation(self):
        """Test implicit notation to ensure it matches numpy's behavior."""
        a = np.random.rand(2, 3)
        b = np.random.rand(3, 4)
        # Numpy's implicit output for 'ij,jk' is 'ik' (alphabetical)
        einsumOp = Einsum("ij,jk", a, b)
        result = einsumOp.execute()
        expected = np.einsum("ij,jk", a, b)
        self.assertTrue(np.allclose(result, expected))
        self.assertEqual(result.shape, expected.shape)
        # Also check against explicit notation
        expectedExplicit = np.einsum("ij,jk->ik", a, b)
        self.assertTrue(np.allclose(result, expectedExplicit))

    # Nested Einsum Tests
    def testSingleNested(self):
        """Test single-nested einsum (ab,(bd,dc->bc)->ac)."""
        tensorAb = np.ones((2, 3))  # ab
        tensorBd = np.ones((3, 4))  # bd
        tensorDc = np.ones((4, 5))  # dc
        einsumOp = Einsum("(ab,(bd,dc->bc)->ac)", tensorAb, tensorBd, tensorDc)
        result = einsumOp.execute()
        expected = np.einsum("ab,bd,dc->ac", tensorAb, tensorBd, tensorDc)
        self.assertTrue(np.allclose(result, expected))
        self.assertEqual(result.shape, (2, 5))

    def testDoubleNested(self):
        """Test double-nested einsum (ab,(bc,(cd,de->ce)->be)->ae)."""
        tensorAb = np.ones((2, 3))  # ab
        tensorBc = np.ones((3, 4))  # bc
        tensorCd = np.ones((4, 5))  # cd
        tensorDe = np.ones((5, 6))  # de
        einsumOp = Einsum("(ab,(bc,(cd,de->ce)->be)->ae)", tensorAb, tensorBc, tensorCd, tensorDe)
        result = einsumOp.execute()
        expected = np.einsum("ab,bc,cd,de->ae", tensorAb, tensorBc, tensorCd, tensorDe)
        self.assertTrue(np.allclose(result, expected))
        self.assertEqual(result.shape, (2, 6))

    # Shape and Metadata Tests
    def testGetShapes(self):
        """Test the getShapes method."""
        a = np.ones((2, 3))
        b = np.ones((3, 4))
        einsumOp = Einsum("ij,jk->ik", a, b)
        einsumOp.execute()
        inputShapes, outputShape = einsumOp.getShapes()
        self.assertEqual(inputShapes, [(2, 3), (3, 4)])
        self.assertEqual(outputShape, (2, 4))

    # Error Handling Tests
    def testShapeMismatchError(self):
        """Test ValueError for tensor shape not matching notation."""
        with self.assertRaisesRegex(ValueError, "Shape mismatch"):
            Einsum("ij", np.ones((2, 3, 4))).execute()

    def testDimensionMismatchError(self):
        """Test ValueError for inconsistent dimension sizes."""
        with self.assertRaisesRegex(ValueError, "Inconsistent dimension"):
            Einsum("ij,jk", np.ones((2, 3)), np.ones((4, 5))).execute()

    def testInputCountMismatchError(self):
        """Test ValueError for wrong number of tensors."""
        with self.assertRaisesRegex(ValueError, "Input count mismatch"):
            Einsum("ij,jk", np.ones((2, 3))).execute()

    # Optimizer Tests
    def testOptimizerPathChoice(self):
        """Test if the optimizer chooses the correct contraction path."""
        # Case 1: Cheaper to contract first two tensors
        a = np.random.rand(10, 20)  # i,j
        b = np.random.rand(20, 30)  # j,k
        c = np.random.rand(30, 40)  # k,l
        # Cost ((ij,jk),kl): 10*20*30 + 10*30*40 = 6000 + 12000 = 18000
        # Cost (ij,(jk,kl)): 20*30*40 + 10*20*40 = 24000 + 8000 = 32000
        einsumOp = Einsum("ij,jk,kl->il", a, b, c)
        einsumOp.execute()
        # Path should be [(0, 1), (0, 1)]
        self.assertEqual(einsumOp.path, [(0, 1), (0, 1)])

        # Case 2: Cheaper to contract last two tensors
        a = np.random.rand(40, 30)  # i,j
        b = np.random.rand(30, 20)  # j,k
        c = np.random.rand(20, 10)  # k,l
        # Cost ((ij,jk),kl): 40*30*20 + 40*20*10 = 24000 + 8000 = 32000
        # Cost (ij,(jk,kl)): 30*20*10 + 40*30*10 = 6000 + 12000 = 18000
        einsumOp = Einsum("ij,jk,kl->il", a, b, c)
        einsumOp.execute()
        # Path should be [(1, 2), (0, 1)]
        self.assertEqual(einsumOp.path, [(1, 2), (0, 1)])

    def testCustomOptimizer(self):
        """Test using a custom optimizer."""
        a = np.ones((2, 3))
        b = np.ones((3, 4))
        customOptimizer = GreedyOrderOptimizer()
        einsumOp = Einsum("ij,jk->ik", a, b, optimizer=customOptimizer)
        result = einsumOp.execute()
        expected = np.einsum("ij,jk->ik", a, b)
        self.assertTrue(np.allclose(result, expected))
        self.assertIs(einsumOp.optimizer, customOptimizer)


if __name__ == "__main__":
    unittest.main()
