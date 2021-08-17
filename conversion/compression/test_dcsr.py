from .dcsr import *

def test_remove_zeros():
    test_arr = np.array([0,0,5,0,3,0,7])
    np.testing.assert_array_equal(
        remove_zeros(test_arr)[0],
        np.array([2,4,6])
    )
    np.testing.assert_array_equal(
        remove_zeros(test_arr)[1],
        np.array([5,3,7])
    )

def test_idx_overflow():
    test_arr = np.array([[0, 256], [3, 4]])
    assert idx_overflow(test_arr, 1)

def test_pad_gap():
    test_indices = np.array([0,16,31])
    test_values = np.array([1,2,3])
    np.testing.assert_array_equal(
        pad_largest_gap(test_indices, test_values, 32)[0],
        np.array([0,8,16,31])
    )
    np.testing.assert_array_equal(
        pad_largest_gap(test_indices, test_values, 32)[1],
        np.array([1,0,2,3])
    )

    test_indices = np.array([15])
    test_values = np.array([1])
    np.testing.assert_array_equal(
        pad_largest_gap(test_indices, test_values, 20)[0],
        np.array([7,15])
    )
    np.testing.assert_array_equal(
        pad_largest_gap(test_indices, test_values, 20)[1],
        np.array([0,1])
    )
    test_indices = np.array([75,113,151])
    test_values = np.array([1,2,3])
    np.testing.assert_array_equal(
        pad_largest_gap(test_indices, test_values, 156)[0],
        np.array([37,75,113,151])
    )
    np.testing.assert_array_equal(
        pad_largest_gap(test_indices, test_values, 156)[1],
        np.array([0,1,2,3])
    )
    np.testing.assert_array_equal(
        pad_largest_gap(test_indices, test_values, 256)[0],
        np.array([75,113,151,203])
    )
    np.testing.assert_array_equal(
        pad_largest_gap(test_indices, test_values, 256)[1],
        np.array([1,2,3,0])
    )

def test_row_offsets():
    test_arr = np.array([3,4,5])
    np.testing.assert_array_equal(
            get_row_offsets(test_arr),
            np.array([-1, 0, 1])
    )

def test_pad_values():
    test_arr = np.array([1,2,3])
    np.testing.assert_array_equal(
            pad_values(test_arr, 4),
            np.array([1,2,3,0])
    )

def test_groups():
    test_arr = np.array([1,2,3,4,5])
    np.testing.assert_array_equal(
            simd_groups(test_arr, 3),
            np.array([[1,2,3],[4,5,0]])
    )

def test_remove_slope():
    test_arr = np.array([[0,1,2,3]])
    np.testing.assert_array_equal(
            remove_slope(test_arr,1),
            np.zeros((1,4))
    )
    np.testing.assert_array_equal(
            remove_slope(test_arr,2),
            np.array([[0,-1,-2,-3]])
    )

def test_remove_mins():
    test_arr = np.array([[0,3], [-1,2], [3,4]])
    np.testing.assert_array_equal(
        remove_minimums(test_arr)[0],
            np.array([[0,3],[0,3],[0,1]])
    )
    np.testing.assert_array_equal(
        remove_minimums(test_arr)[1],
        np.array([0,-1,3])
    )

def test_min_offsets():
    test_arr = np.array([3,10,20,29])
    np.testing.assert_array_equal(
        minimum_offsets(test_arr, slope=2, group_size=4),
        np.array([3,-1,2,1])
    )
    test_arr = np.array([3,10,20,29])
    np.testing.assert_array_equal(
        minimum_offsets(test_arr, slope=2, group_size=4, absolute=True),
        np.array([3, 2, 4, 5])
    )


def test_merged_bitmaps():
    test_arr = np.array([[0,0], [0,48], [32,0]])
    np.testing.assert_array_equal(
        get_merged_bitmaps(test_arr, [5,6])[0],
        np.array([0, 0x3, 0x2])
    )
    np.testing.assert_array_equal(
        get_merged_bitmaps(test_arr, [5,6])[1],
        np.array([[False, True], [False, True], [True, False]])
    )

def test_compress_row():
    test_arr = np.array([0,1,0,2,0,3])
    row_data = compress_row(test_arr, group_size=4)
    np.testing.assert_array_equal(
        row_data['values'],
        np.array([1,2,3])
    )
    np.testing.assert_array_equal(
        row_data['delta_indices'],
        np.array([[0,0,0,0]])
    )
    np.testing.assert_array_equal(
        row_data['minimums'],
        np.array([1])
    )
    assert row_data['groups_count'] == 1

def test_bit_position():
    test_arr = np.array([[0,0], [0,16], [0,32]])
    np.testing.assert_array_equal(
        get_bit_position(test_arr)[0],
        np.array([False, True, False])
    )
    np.testing.assert_array_equal(
        get_bit_position(test_arr)[1],
        np.array([[False,True]])
    )
    np.testing.assert_array_equal(
        get_bit_position(test_arr)[2],
        np.array([[0,0], [0,0], [0,32]])
    )

def test_pack():
    test_arr = np.array([[0xf,0x0], [0x1,0x2], [0x3,0x4]])
    np.testing.assert_array_equal(
        pack_nibbles(test_arr),
        np.array([[0xf1,0x02], [0x30,0x40]])
    )

def test_bit_widths():
    np.testing.assert_array_equal(
        get_bit_widths(np.array([[0,1], [2,0], [16,0]])),
        np.array([1,2,5])
    )

def test_mask_padding():
    np.testing.assert_array_equal(
        compensate_padding(np.array([[1,2,3],[3,4,5]]),4),
        np.array([[1,2,3],[3,3,3]])
    )

def test_offset_differences():
    np.testing.assert_array_equal(
        get_offset_differences(np.array([1,3,5,6,9])),
        np.array([1,2,2,1,3])
    )

def test_convert_map():
    assert convert_map(np.array([True, True, False, False, True, False, True, False])) == 83

def test_convert_mask():
    test_arr = np.array([[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]])
    np.testing.assert_array_equal(
        convert_mask(test_arr),
        np.array([1, 0x8000])
    )

def test_c_def():
    test_arr = np.array([1,2,3])
    assert to_c_def(test_arr, "test") == "#define TEST {1, 2, 3}\r\n"
    assert to_c_def(123456, "test") == "#define TEST 123456\r\n"
