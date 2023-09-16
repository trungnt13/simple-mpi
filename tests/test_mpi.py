import unittest

from simpi.mpi import MPI


class TestMPI(unittest.TestCase):
    def test_mpi_lambda(self):
        worker = lambda x: x**2
        jobs = [1, 2, 3, 4, 5]

        for i in MPI(jobs=jobs, func=worker, ncpu=4):
            print(i)


if __name__ == "__main__":
    unittest.main()
