import numpy as np

from ramanpl.dataImporter import DataImporter


def _write_txt_map(filepath, rows):
    header = "X Y xaxis intensity"
    np.savetxt(filepath, np.asarray(rows, dtype=float), header=header, comments="")


def test_mask_by_xrange_selects_same_physical_window_for_ascending_and_descending_x():
    x_asc = np.array([100.0, 125.0, 150.0, 175.0, 200.0])
    x_desc = x_asc[::-1].copy()

    mask_asc = DataImporter.mask_by_xrange(x_asc, (120.0, 180.0))
    mask_desc = DataImporter.mask_by_xrange(x_desc, (120.0, 180.0))

    assert np.array_equal(mask_asc, np.array([False, True, True, True, False]))
    assert np.array_equal(mask_desc, np.array([False, True, True, True, False]))

    x_sel_asc, _ = DataImporter.select_by_xrange(
        x_asc,
        np.arange(x_asc.size, dtype=float),
        (120.0, 180.0),
        axis="wavenumber",
    )
    x_sel_desc, _ = DataImporter.select_by_xrange(
        x_desc,
        np.arange(x_desc.size, dtype=float),
        (120.0, 180.0),
        axis="wavenumber",
    )

    assert np.allclose(x_sel_asc, np.array([125.0, 150.0, 175.0]))
    assert np.allclose(x_sel_desc, np.array([175.0, 150.0, 125.0]))


def test_mask_by_xrange_accepts_reversed_user_range():
    x = np.array([100.0, 125.0, 150.0, 175.0, 200.0])

    mask = DataImporter.mask_by_xrange(x, (180.0, 120.0))

    assert np.array_equal(mask, np.array([False, True, True, True, False]))


def test_read_txt_map_assigns_correct_pixels_when_coordinate_blocks_are_shuffled(tmp_path):
    filepath = tmp_path / "shuffled_map.txt"

    # Each pixel contributes one contiguous block of 3 spectral points.
    # Blocks are intentionally written in a scrambled coordinate order.
    rows = [
        # (x=1, y=0)
        [1, 0, 100, 11],
        [1, 0, 110, 12],
        [1, 0, 120, 13],

        # (x=0, y=1)
        [0, 1, 100, 21],
        [0, 1, 110, 22],
        [0, 1, 120, 23],

        # (x=1, y=1)
        [1, 1, 100, 31],
        [1, 1, 110, 32],
        [1, 1, 120, 33],

        # (x=0, y=0)
        [0, 0, 100, 41],
        [0, 0, 110, 42],
        [0, 0, 120, 43],
    ]
    _write_txt_map(filepath, rows)

    spectra_cube, xdata, X, Y = DataImporter._read_txt_map(str(filepath), txt_skiprows=1)

    assert X == 2
    assert Y == 2
    assert spectra_cube.shape == (2, 2, 3)
    assert np.allclose(xdata, np.array([100.0, 110.0, 120.0]))

    # np.unique sorts coordinates ascending, so:
    # x index 0 -> x=0, x index 1 -> x=1
    # y index 0 -> y=0, y index 1 -> y=1
    assert np.allclose(spectra_cube[0, 0, :], np.array([41.0, 42.0, 43.0]))  # (0,0)
    assert np.allclose(spectra_cube[0, 1, :], np.array([11.0, 12.0, 13.0]))  # (1,0)
    assert np.allclose(spectra_cube[1, 0, :], np.array([21.0, 22.0, 23.0]))  # (0,1)
    assert np.allclose(spectra_cube[1, 1, :], np.array([31.0, 32.0, 33.0]))  # (1,1)


def test_map_import_txt_applies_xrange_trim_after_coordinate_assignment(tmp_path):
    filepath = tmp_path / "trimmed_map.txt"

    rows = [
        # (x=0, y=0)
        [0, 0, 100, 1],
        [0, 0, 110, 2],
        [0, 0, 120, 3],

        # (x=1, y=0)
        [1, 0, 100, 4],
        [1, 0, 110, 5],
        [1, 0, 120, 6],

        # (x=0, y=1)
        [0, 1, 100, 7],
        [0, 1, 110, 8],
        [0, 1, 120, 9],

        # (x=1, y=1)
        [1, 1, 100, 10],
        [1, 1, 110, 11],
        [1, 1, 120, 12],
    ]
    _write_txt_map(filepath, rows)

    spectra_cube, xdata, X, Y = DataImporter.map_import(
        str(filepath),
        x_range=(105.0, 115.0),
        axis="wavenumber",
        txt_skiprows=1,
    )

    assert X == 2
    assert Y == 2
    assert spectra_cube.shape == (2, 2, 1)
    assert np.allclose(xdata, np.array([110.0]))

    assert np.allclose(spectra_cube[0, 0, :], np.array([2.0]))
    assert np.allclose(spectra_cube[0, 1, :], np.array([5.0]))
    assert np.allclose(spectra_cube[1, 0, :], np.array([8.0]))
    assert np.allclose(spectra_cube[1, 1, :], np.array([11.0]))