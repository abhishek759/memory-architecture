"""Compatibility entrypoint; all new experiments use the shared runner.
Original September 15 implementation is archived in notes/ for provenance.
"""
import sys
from run_experiment import main

if __name__ == '__main__':
    sys.argv.extend(['--approach', 'raw_history'])
    if '--output' not in sys.argv:
        sys.argv.extend(['--output', 'results/pilot/raw_history'])
    main()
