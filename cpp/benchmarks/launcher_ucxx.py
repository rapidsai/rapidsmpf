# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

import subprocess
import re
import click
import time

@click.command()
@click.option('-e', '--executable', type=click.Path(exists=True), required=True, help='Path to the executable (e.g., ./cpp/build/benchmarks/bench_shuffle)')
@click.option('-k', '--ranks', type=int, default=1, help='Number of ranks to launch')
@click.option('-r', '--runs', type=int, default=1, help='Number of runs to perform')
@click.option('-w', '--warmup-runs', type=int, default=1, help='Number of warmup runs to perform')
def launch(executable, ranks, runs, warmup_runs):
    """
    Launcher for External Executable.

    This script launches a main instance of the specified executable, parses its output for an IP address and port,
    and then launches additional ranks configured to connect to the main instance.
    """
    common_args = ['-k', str(ranks), '-r', str(runs), '-w', str(warmup_runs)]
    root_rank_cmd = [executable, '-z'] + common_args
    additional_rank_base_cmd = [executable] + common_args

    # Run the main process and capture its output
    print("Running main process...")
    process = subprocess.Popen(root_rank_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)

    # Pattern to match IP address and port from the output
    pattern = r'Root running at address ([^:]+):(\d+)'

    # Initialize variables to hold the IP and port
    ip, port = None, None

    # Iterate over the output to find the match before the process potentially finishes
    for line in iter(process.stdout.readline, ''):
        print(line, end='')  # Print the output live
        match = re.search(pattern, line)
        if match:
            ip, port = match.groups()
            ip = "localhost"
            print(f"Found IP: {ip}, Port: {port}")
            break
    else:
        # If no match found and the process has finished, print a message and exit
        if process.poll() is not None:
            print("Failed to retrieve IP and Port from the main process output.")
            return

    if ip and port:
        # Launch additional ranks
        for rank in range(ranks-1):
            # Customize the command for this instance
            additional_rank_cmd = additional_rank_base_cmd + ['-H', ip, '-P', str(port)]
            print(f"Launching rank {rank+1} with command: {' '.join(additional_rank_cmd)}")
            subprocess.Popen(additional_rank_cmd)
    else:
        print("Unable to proceed without IP and Port.")

    # Wait for the main process to finish
    process.wait()

if __name__ == "__main__":
    launch()
