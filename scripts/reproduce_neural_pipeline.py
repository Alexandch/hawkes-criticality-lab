"""Regenerate simulation experiments without modifying published snapshots."""
from __future__ import annotations
import argparse
from pathlib import Path
from critical_processes.selected_training import run_selected_training
from critical_processes.sensitivity_diagnostics import run_sensitivity_diagnostics
from critical_processes.dynamic_experiment import run_dynamic_experiment
from critical_processes.extended_training import run_extended_training


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',required=True,choices=['selected','sensitivity','dynamic','extended','all'])
    parser.add_argument('--out',type=Path,default=Path('runs/reproduced'))
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    paths={name:args.out/name for name in ('selected_detectors','sensitivity_diagnostics','dynamic_estimation','extended_neural')}
    archived=Path('results').resolve()
    if args.out.resolve()==archived or args.out.resolve().is_relative_to(archived):
        parser.error('choose an output outside archived results/')
    stages=('selected','sensitivity','dynamic','extended') if args.stage=='all' else (args.stage,)
    for stage in stages:
        if stage=='selected':run_selected_training(out=paths['selected_detectors'],workers=args.workers)
        elif stage=='sensitivity':run_sensitivity_diagnostics(source=paths['selected_detectors'],out=paths['sensitivity_diagnostics'],workers=args.workers)
        elif stage=='dynamic':run_dynamic_experiment(out=paths['dynamic_estimation'],sources=[paths['selected_detectors'],paths['sensitivity_diagnostics']],workers=args.workers)
        else:run_extended_training(out=paths['extended_neural'],sources=[paths['selected_detectors'],paths['sensitivity_diagnostics'],paths['dynamic_estimation']],workers=args.workers)


if __name__=='__main__':main()
