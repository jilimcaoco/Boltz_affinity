#!/usr/bin/env python3
"""Queue Boltz affinity rescoring jobs on SLURM with configurable concurrency.

This script discovers receptor/pose pairs and submits each job through
`submit_boltz.py` while optionally chaining dependencies.

With `--max-concurrent 1` (default):
    job_2 depends on job_1 finishing,
    job_3 depends on job_2 finishing, etc.

With `--max-concurrent N`:
    up to N jobs can run at once.

With `--max-concurrent 0`:
    no dependency throttling is applied (SLURM uses as many GPUs as available).

Optional `--wait-and-gather` mode waits for all submitted jobs to leave the queue
and then aggregates per-receptor CSV outputs into one master CSV.

example usages:
python queue_boltz_rescoring_serial.py --help
python queue_boltz_rescoring_serial.py --time 24:00:00 --max-concurrent 3 --wait-and-gather
python queue_boltz_rescoring_serial.py --time 24:00:00 --max-concurrent 3

"""

from __future__ import annotations

import argparse
import csv
import time
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ReceptorJob:
    receptor_id: str
    receptor_path: Path
    poses_path: Path
    output_path: Path
    log_path: Path


def default_paths() -> tuple[Path, Path, Path, Path]:
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[2]
    experiment_root = script_path.parents[1]
    submit_script = repo_root / "submit_boltz.py"
    poses_dir = experiment_root / "DOCK3.8_poses"
    receptors_dir = experiment_root / "receptors"
    output_dir = experiment_root / "results" / "rescoring"
    return submit_script, poses_dir, receptors_dir, output_dir


def parse_args() -> argparse.Namespace:
    submit_script, poses_dir, receptors_dir, output_dir = default_paths()

    parser = argparse.ArgumentParser(
        description="Submit Boltz rescoring jobs with configurable SLURM concurrency.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--submit-script", type=Path, default=submit_script)
    parser.add_argument("--poses-dir", type=Path, default=poses_dir)
    parser.add_argument("--receptors-dir", type=Path, default=receptors_dir)
    parser.add_argument("--output-dir", type=Path, default=output_dir)
    parser.add_argument("--manifest", type=Path, default=None, help="Optional manifest CSV path")
    parser.add_argument("--time", default="01:00:00")
    parser.add_argument("--account", default="maom")
    parser.add_argument("--partition", default="maom-h200")
    parser.add_argument("--cpus", type=int, default=1)
    parser.add_argument("--mem", default="48000")
    parser.add_argument("--conda-env", default="boltz_affinity")
    parser.add_argument("--sort-by", default="affinity_score")
    parser.add_argument("--output-format", choices=["csv", "json"], default="csv")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="DEBUG")
    parser.add_argument(
        "--no-msa-server",
        action="store_true",
        help="Disable MSA server for all submitted jobs",
    )
    parser.add_argument("--start-after-jobid", default=None, help="Chain first job after this existing SLURM job ID")
    parser.add_argument("--limit", type=int, default=None, help="Submit only first N receptors")
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=1,
        help="Max jobs allowed to run concurrently (1=serial, 0=no cap)",
    )
    parser.add_argument(
        "--wait-and-gather",
        action="store_true",
        help="Wait for all submitted jobs to finish, then aggregate CSV outputs",
    )
    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=120,
        help="Polling interval while waiting for jobs to finish",
    )
    parser.add_argument(
        "--aggregate-output",
        type=Path,
        default=None,
        help="Path for combined aggregate CSV (default: <output-dir>/all_rescored_results.csv)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Build scripts and print plan, but do not submit")
    return parser.parse_args()


def discover_jobs(poses_dir: Path, receptors_dir: Path, output_dir: Path) -> tuple[list[ReceptorJob], list[str]]:
    jobs: list[ReceptorJob] = []
    missing: list[str] = []

    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    for poses_path in sorted(poses_dir.glob("*_poses.mol2")):
        receptor_id = poses_path.stem.removesuffix("_poses")
        receptor_path = receptors_dir / f"{receptor_id}_receptor.pdb"

        if not receptor_path.exists():
            missing.append(receptor_id)
            continue

        output_path = output_dir / f"{receptor_id}_rescored.csv"
        log_path = logs_dir / f"boltz_{receptor_id}.log"
        jobs.append(
            ReceptorJob(
                receptor_id=receptor_id,
                receptor_path=receptor_path,
                poses_path=poses_path,
                output_path=output_path,
                log_path=log_path,
            )
        )

    return jobs, missing


def build_slurm_script(args: argparse.Namespace, job: ReceptorJob, job_output_path: Path) -> str:
    cmd = [
        sys.executable,
        str(args.submit_script),
        "--receptor",
        str(job.receptor_path),
        "--ligands",
        str(job.poses_path),
        "--output",
        str(job_output_path),
        "--output-format",
        args.output_format,
        "--sort-by",
        args.sort_by,
        "--log-level",
        args.log_level,
        "--job-name",
        f"boltz_{job.receptor_id}",
        "--log-file",
        str(job.log_path),
        "--time",
        args.time,
        "--account",
        args.account,
        "--partition",
        args.partition,
        "--cpus",
        str(args.cpus),
        "--mem",
        str(args.mem),
        "--conda-env",
        args.conda_env,
        "--dry-run",
    ]

    if args.no_msa_server:
        cmd.append("--no-msa-server")

    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    text = result.stdout
    idx = text.find("#!/bin/bash")
    if idx == -1:
        raise RuntimeError(f"Could not parse dry-run SLURM script for receptor {job.receptor_id}.")
    return text[idx:]


def submit_serial_chain(
    jobs: list[ReceptorJob],
    args: argparse.Namespace,
) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    submitted_jobids: list[str] = []

    for idx, job in enumerate(jobs, start=1):
        job_output_path = job.output_path.with_suffix(f".{args.output_format}")
        script_text = build_slurm_script(args, job, job_output_path)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".slurm", prefix=f"boltz_{job.receptor_id}_", delete=False) as tf:
            tf.write(script_text)
            temp_script_path = Path(tf.name)

        dependency_ids: list[str] = []
        if args.start_after_jobid:
            dependency_ids.append(args.start_after_jobid)

        if args.max_concurrent < 0:
            raise SystemExit("--max-concurrent must be >= 0")

        if args.max_concurrent > 0 and len(submitted_jobids) >= args.max_concurrent:
            dependency_ids.append(submitted_jobids[-args.max_concurrent])

        sbatch_cmd = ["sbatch"]
        if dependency_ids:
            sbatch_cmd.append(f"--dependency=afterany:{':'.join(dependency_ids)}")
        sbatch_cmd.append(str(temp_script_path))

        if args.dry_run:
            print(f"[DRY-RUN] Would submit {job.receptor_id}: {' '.join(sbatch_cmd)}")
            pseudo_jobid = f"DRY_RUN_{idx}"
            records.append(
                {
                    "receptor_id": job.receptor_id,
                    "slurm_job_id": pseudo_jobid,
                    "dependency": ":".join(dependency_ids),
                    "receptor": str(job.receptor_path),
                    "poses": str(job.poses_path),
                    "output": str(job_output_path),
                    "log": str(job.log_path),
                }
            )
            submitted_jobids.append(pseudo_jobid)
            temp_script_path.unlink(missing_ok=True)
            continue

        submit = subprocess.run(sbatch_cmd, check=True, capture_output=True, text=True)
        stdout = submit.stdout.strip()
        match = re.search(r"Submitted batch job (\d+)", stdout)
        if not match:
            temp_script_path.unlink(missing_ok=True)
            raise RuntimeError(f"Unexpected sbatch output for {job.receptor_id}: {stdout}")

        jobid = match.group(1)
        print(
            f"Queued {job.receptor_id}: job {jobid}"
            + (f" (after {':'.join(dependency_ids)})" if dependency_ids else "")
        )

        records.append(
            {
                "receptor_id": job.receptor_id,
                "slurm_job_id": jobid,
                    "dependency": ":".join(dependency_ids),
                "receptor": str(job.receptor_path),
                "poses": str(job.poses_path),
                    "output": str(job_output_path),
                "log": str(job.log_path),
            }
        )
        submitted_jobids.append(jobid)
        temp_script_path.unlink(missing_ok=True)

    return records


def write_manifest(path: Path, records: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["receptor_id", "slurm_job_id", "dependency", "receptor", "poses", "output", "log"],
        )
        writer.writeheader()
        writer.writerows(records)


def wait_for_jobs_to_finish(job_ids: list[str], poll_seconds: int) -> None:
    if not job_ids:
        return

    print(f"Waiting for {len(job_ids)} jobs to finish (poll every {poll_seconds}s)...")
    last_status = ""

    while True:
        check = subprocess.run(
            ["squeue", "-h", "-j", ",".join(job_ids), "-o", "%i %T"],
            capture_output=True,
            text=True,
        )
        if check.returncode != 0:
            raise RuntimeError(f"squeue failed while waiting for jobs: {check.stderr.strip()}")

        active_lines = [line.strip() for line in check.stdout.splitlines() if line.strip()]
        if not active_lines:
            print("All submitted jobs have completed and left the queue.")
            return

        status_msg = f"Active jobs: {len(active_lines)}"
        if status_msg != last_status:
            print(status_msg)
            last_status = status_msg

        time.sleep(max(poll_seconds, 5))


def aggregate_csv_outputs(records: list[dict[str, str]], aggregate_output: Path) -> None:
    rows: list[dict[str, str]] = []
    fieldnames: list[str] = ["receptor_id", "source_file"]
    missing_outputs: list[str] = []

    for record in records:
        receptor_id = record["receptor_id"]
        output_path = Path(record["output"])
        if not output_path.exists():
            missing_outputs.append(receptor_id)
            continue

        with output_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                for field in reader.fieldnames:
                    if field not in fieldnames:
                        fieldnames.append(field)

            for row in reader:
                row_out = {
                    "receptor_id": receptor_id,
                    "source_file": str(output_path),
                }
                row_out.update(row)
                rows.append(row_out)

    aggregate_output.parent.mkdir(parents=True, exist_ok=True)
    with aggregate_output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote aggregated results: {aggregate_output} (rows: {len(rows)})")
    if missing_outputs:
        print(f"Missing output CSVs for {len(missing_outputs)} receptors.")


def main() -> None:
    args = parse_args()

    for p in (args.submit_script, args.poses_dir, args.receptors_dir):
        if not p.exists():
            raise SystemExit(f"Missing required path: {p}")

    jobs, missing = discover_jobs(args.poses_dir, args.receptors_dir, args.output_dir)

    if args.limit is not None:
        jobs = jobs[: args.limit]

    if not jobs:
        raise SystemExit("No receptor/poses pairs found to submit.")

    print(f"Discovered {len(jobs)} jobs.")
    if missing:
        print(f"Skipping {len(missing)} receptor IDs with missing receptor PDB files.")

    records = submit_serial_chain(jobs, args)

    manifest = args.manifest or (args.output_dir / "submission_manifest.csv")
    write_manifest(manifest, records)
    print(f"Wrote manifest: {manifest}")

    if args.wait_and_gather:
        if args.dry_run:
            print("Skipping wait-and-gather in --dry-run mode.")
            return
        if args.output_format != "csv":
            raise SystemExit("--wait-and-gather currently supports only --output-format csv")

        submitted_job_ids = [record["slurm_job_id"] for record in records]
        wait_for_jobs_to_finish(submitted_job_ids, args.poll_seconds)

        aggregate_output = args.aggregate_output or (args.output_dir / "all_rescored_results.csv")
        aggregate_csv_outputs(records, aggregate_output)


if __name__ == "__main__":
    main()
