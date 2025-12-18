import os
import sys
import unittest

# Add root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestOrchardStructure(unittest.TestCase):
    def test_kernels_exist(self):
        """Check if kernel files were moved correctly."""
        import orchard
        orchard_path = os.path.dirname(orchard.__file__)
        kernels_path = os.path.join(orchard_path, "kernels")
        self.assertTrue(os.path.exists(kernels_path))
        self.assertTrue(os.path.exists(os.path.join(kernels_path, "matmul.metal")))
        self.assertTrue(os.path.exists(os.path.join(kernels_path, "gemm_int4.metal")))

    def test_cli_import(self):
        """Check if CLI module is importable."""
        from orchard import cli
        self.assertTrue(hasattr(cli, "main"))

if __name__ == "__main__":
    unittest.main()
