from .estimate_references import *

test_matrix = np.array([[0,0,0,0,1,0,2,3],[1,0,2,4,0,0,0,0]])

def test_csr():
    assert estimate_csr(test_matrix) == 6 + 2*6 + 2*2
    assert estimate_csr(test_matrix, row_ptr_bytes=1) == 6 + 2*6 + 2
    assert estimate_csr(test_matrix, col_idx_bytes=1) == 6 + 6 + 2*2

def test_bcsr():
    assert estimate_bcsr(test_matrix) == 16 + 4*2 + 2*2

    test_matrix2 = np.array([[1,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0]])
    assert estimate_bcsr(test_matrix2) == 4 + 1*2 + 2*2
