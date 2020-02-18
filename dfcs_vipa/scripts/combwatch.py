"""Get frequency counter measurements from FiberCombControl server
communicating with Menlo FC1500.

The amount of data is limited by the internal buffer size of the server
to the last hour.
"""
import logging
import sys
from xmlrpc.client import ServerProxy
import numpy as np
from argparse import ArgumentParser

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s: %(message)s', 
    datefmt='%d.%m.%Y %H:%M:%S'
)
log = logging.getLogger(__name__)

COMB_URI_FMT = 'http://{:s}:8123'
HEADER = 'timestamp, reprate.freq, offset.freq'

def main():
    parser = ArgumentParser()
    parser.add_argument('n', help='Save last n values (last n seconds)', type=int)
    parser.add_argument('output', help='Path to output file')
    parser.add_argument('ip', help='IP address of the comb PC')
    args = parser.parse_args(sys.argv[1:])

    COMB_URI = COMB_URI_FMT.format(args.ip.strip())
    server = ServerProxy(COMB_URI)
    data = [(float(k), v['reprate.freq'], v['offset.freq'], v['counter2.freq']) for k, v in
            server.data.query(-float(args.n),
                              ['reprate.freq', 'offset.freq', 'counter2.freq']).items()]
    data.sort(key=lambda x: x[0])
    log.debug(data)
    np.savetxt(args.output, np.array(data), delimiter=',', header=HEADER)
