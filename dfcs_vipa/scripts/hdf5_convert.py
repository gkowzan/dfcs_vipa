"""Convert old style HDF5 files to the new style."""
import sys
from argparse import ArgumentParser
import dfcs_vipa
from dfcs_vipa.cmws import dir_hdf5old2new_copy

def main():
    parser = ArgumentParser()
    parser.add_argument('cols', help="Camera width.", type=int)
    parser.add_argument('rows', help="Camera height.", type=int)
    parser.add_argument('directory', help="Directory with old style files.")
    args = parser.parse_args(sys.argv[1:])

    dfcs_vipa.COLS = args.cols
    dfcs_vipa.ROWS = args.rows
    dir_hdf5old2new_copy(args.directory)
