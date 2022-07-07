#!/usr/bin/env python


import json
import os
from enum import Enum
from typing import Optional

import typer
from coffea import processor
from coffea.nanoevents import NanoAODSchema
from tqdm import tqdm

from hzupsilonphoton.analyzer_trigg import Analyzer_Trigg
#from hzupsilonphoton.gen_analyzer import GenAnalyzer
#from hzupsilonphoton.output_merger import output_merger
from hzupsilonphoton.utils import file_tester
from samples.samples_details import mc_samples_files, samples, samples_files

# create typer app
help_str = """
H/Z \n\n\n--> Y(nS) + gamma trigger analysis code (NanoAOD - Run2) \n\n

Usual workflow:\n
# clear output buffers\n
./run_analysis_Trigg.py clear \n

# main analysis code for signal selection\n
./run_analysis_Trigg.py main \n

"""
app = typer.Typer(help=typer.style(help_str, fg=typer.colors.BRIGHT_BLUE, bold=True))


@app.command() #mudar samples para samples_Trigg
def test_files() -> None:
    """Test uproot.open each sample file."""
    files = []
    for s in samples:
        for f in samples[s]["files"]:
            files.append(f)
    for f in tqdm(files):
        file_tester(f)


@app.command()
def clear() -> None:
    """Clear outputs."""

    os.system("rm -rf outputs_Trigg/*")
    os.system("mkdir -p outputs_Trigg/buffer")


class CoffeaExecutors(str, Enum):
    futures = "futures"
    iterative = "iterative"


@app.command()
def main(
    maxchunks: Optional[int] = -1,  # default -1
    executor: CoffeaExecutors = CoffeaExecutors.futures,
    workers: int = 60,  # default 60
) -> None:
    """Run Trigg analysis and saves outputs."""

    executor_args = {"schema": NanoAODSchema, "workers": workers}
    if executor.value == "interative":
        executor_args = {"schema": NanoAODSchema}

    executor = getattr(processor, f"{executor.value}_executor")

    if maxchunks == -1:
        maxchunks = None

    # clear buffers
    os.system("rm -rf outputs_Trigg/buffer")
    os.system("mkdir -p outputs_Trigg/buffer")

    # run analysis
    print("\n\n\n--> Running Trigg level analysis...")
    output = processor.run_uproot_job(
        fileset=samples_files,
        treename="Events",
        processor_instance=Analyzer_Trigg(),
        # executor=processor.futures_executor,
        # executor = processor.iterative_executor,
        executor=executor,
        executor_args=executor_args,
        # executor_args = {"schema": NanoAODSchema},
        # chunksize =
        maxchunks=maxchunks,
    )

    # save outputs
    print("\n\n\n--> saving output...")
    output_filename = "outputs_Trigg/cutflow.json"
    os.system(f"rm -rf {output_filename}")
    # pprint(output)
    with open(output_filename, "w") as f:
        f.write(json.dumps(output))


if __name__ == "__main__":
    app()  # start typer app
