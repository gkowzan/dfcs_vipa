"""Prepare reference grid for online spectrum generation by data
acquisition software.

Average camera frames with a CW marker, broadband light source and save
the resultant grid to a file.  Prepare frequency axis based on FSR of
the etalon and wavelength of the CW marker.
"""
import sys
from pathlib import Path
from argparse import ArgumentParser
import numpy as np
import dfcs_vipa
import dfcs_vipa.grid as grid
import dfcs_vipa.collect as collect
import dfcs_vipa.experiment as ex

def main():
    parser = ArgumentParser()
    parser.add_argument('cw_marker', help="HDF5 file with CW marker.")
    parser.add_argument('cw_marker_dc', help="HDF5 file with CW marker dark frames.")
    parser.add_argument('broadband', help="HDF5 file with broadband spectrum.")
    parser.add_argument('broadband_dc', help="HDF5 file with broadband spectrum dark frames.")
    parser.add_argument('col_min', help="Left limiting column.", type=int)
    parser.add_argument('col_max', help="Right limiting column.", type=int)
    parser.add_argument('fsr', help="Free spectral range of the VIPA etalon.", type=float)
    parser.add_argument('cw_wavelength', help="Wavelength of the CW marker (in nm).", type=float)
    parser.add_argument('cols', help="Camera width", type=int)
    parser.add_argument('rows', help="Camera height", type=int)
    args = parser.parse_args(sys.argv[1:])

    dfcs_vipa.ROWS = args.rows
    dfcs_vipa.COLS = args.cols
    # determine the grid
    cw_arr = collect.average_h5(
        args.cw_marker, args.cw_marker_dc)
    cw_pos = grid.get_rio_pos(cw_arr)

    grid_arr = collect.average_h5(
        args.broadband, args.broadband_dc)
    grid_points = grid.limit_grid(
        grid.make_grid(grid_arr, cw_pos[0]),
        cw_pos[0], (args.col_min, args.col_max))
    grid_fancy = grid.grid2fancy(grid_points)
    freq_axis_wn = grid.naive_axis(
        grid_fancy, cw_pos, args.fsr, args.cw_wavelength*1e-9)

    # save averaged arrray
    ex.write2dlv(cw_arr, str(Path(args.cw_marker).with_suffix('.dat')))
    ex.write2dlv(grid_arr, str(Path(args.broadband).with_suffix('.dat')))

    # save LabView data
    np.savetxt(Path(args.broadband).with_name('grid.csv'),
               np.array(grid_points), fmt='%d', delimiter=',')
    print("Grid saved to: {!s}".format(
        Path(args.broadband).with_name('grid.csv')))
    np.savetxt(Path(args.broadband).with_name('freqs_wn.csv'), freq_axis_wn)
    print("Frequency axis in wavenumbers saved to: {!s}".format(
        Path(args.broadband).with_name('freqs_wn.csv')))

    # Make a dummy image for viewing
    dummy = np.zeros((args.rows, args.cols), dtype=np.int)
    dummy[grid_fancy] = 1.0
    ex.write2dlv(dummy, str(Path(args.broadband).with_name('dummy_grid.dat')))
