
import logging
import os
import random
import tarfile
import time
from typing import Optional, Union, Dict

import requests
from requests.auth import HTTPBasicAuth
from tqdm import tqdm

logger = logging.getLogger(__name__)

TQDM_BAR_FORMAT = (
    "{l_bar}{bar}| {n_fmt}/{total_fmt} [elapsed: {elapsed} remaining: {remaining}]"
)


def run_mmseqs2(  # noqa: PLR0912, D103, C901, PLR0915
    x: Union[str, list[str]],
    prefix: str = "tmp",
    use_env: bool = True,
    use_filter: bool = True,
    use_pairing: bool = False,
    pairing_strategy: str = "greedy",
    host_url: str = "https://api.colabfold.com",
    msa_server_username: Optional[str] = None,
    msa_server_password: Optional[str] = None,
    auth_headers: Optional[Dict[str, str]] = None,
) -> tuple[list[str], list[str]]:
    submission_endpoint = "ticket/pair" if use_pairing else "ticket/msa"

    # Validate mutually exclusive authentication methods
    has_basic_auth = msa_server_username and msa_server_password
    has_header_auth = auth_headers is not None
    if has_basic_auth and (has_header_auth or auth_headers):
        raise ValueError(
            "Cannot use both basic authentication (username/password) and header/API key authentication. "
            "Please use only one authentication method."
        )

    # Set header agent as boltz
    headers = {}
    headers["User-Agent"] = "boltz"

    # Set up authentication
    auth = None
    if has_basic_auth:
        auth = HTTPBasicAuth(msa_server_username, msa_server_password)
        logger.debug(f"MMSeqs2 server authentication: using basic auth for user '{msa_server_username}'")
    elif has_header_auth:
        headers.update(auth_headers)
        logger.debug("MMSeqs2 server authentication: using header-based authentication")
    else:
        logger.debug("MMSeqs2 server authentication: no credentials provided")
    
    logger.debug(f"Connecting to MMSeqs2 server at: {host_url}")
    logger.debug(f"Using endpoint: {submission_endpoint}")
    logger.debug(f"Pairing strategy: {pairing_strategy}")
    logger.debug(f"Use environment databases: {use_env}")
    logger.debug(f"Use filtering: {use_filter}")

    def submit(seqs, mode, N=101):
        n, query = N, ""
        for seq in seqs:
            query += f">{n}\n{seq}\n"
            n += 1

        error_count = 0
        while True:
            try:
                # https://requests.readthedocs.io/en/latest/user/advanced/#advanced
                # "good practice to set connect timeouts to slightly larger than a multiple of 3"
                logger.debug(f"Submitting MSA request to {host_url}/{submission_endpoint}")
                res = requests.post(
                    f"{host_url}/{submission_endpoint}",
                    data={"q": query, "mode": mode},
                    timeout=6.02,
                    headers=headers,
                    auth=auth,
                )
                logger.debug(f"MSA submission response status: {res.status_code}")
            except Exception as e:
                error_count += 1
                logger.warning(
                    f"Error while fetching result from MSA server. Retrying... ({error_count}/5)"
                )
                logger.warning(f"Error: {e}")
                if error_count > 5:
                    raise Exception(
                        "Too many failed attempts for the MSA generation request."
                    )
                time.sleep(5)
            else:
                break

        try:
            out = res.json()
        except ValueError:
            logger.error(f"Server didn't reply with json: {res.text}")
            out = {"status": "ERROR"}
        return out

    def status(ID):
        error_count = 0
        while True:
            try:
                logger.debug(f"Checking MSA job status for ID: {ID}")
                res = requests.get(
                    f"{host_url}/ticket/{ID}", timeout=6.02, headers=headers, auth=auth
                )
                logger.debug(f"MSA status check response status: {res.status_code}")
            except Exception as e:
                error_count += 1
                logger.warning(
                    f"Error while fetching result from MSA server. Retrying... ({error_count}/5)"
                )
                logger.warning(f"Error: {e}")
                if error_count > 5:
                    raise Exception(
                        "Too many failed attempts for the MSA generation request."
                    )
                time.sleep(5)
            else:
                break
        try:
            out = res.json()
        except ValueError:
            logger.error(f"Server didn't reply with json: {res.text}")
            out = {"status": "ERROR"}
        return out

    def download(ID, path):
        error_count = 0
        while True:
            try:
                logger.debug(f"Downloading MSA results for ID: {ID}")
                res = requests.get(
                    f"{host_url}/result/download/{ID}", timeout=6.02, headers=headers, auth=auth
                )
                logger.debug(f"MSA download response status: {res.status_code}")
            except Exception as e:
                error_count += 1
                logger.warning(
                    f"Error while fetching result from MSA server. Retrying... ({error_count}/5)"
                )
                logger.warning(f"Error: {e}")
                if error_count > 5:
                    raise Exception(
                        "Too many failed attempts for the MSA generation request."
                    )
                time.sleep(5)
            else:
                break
        with open(path, "wb") as out:
            out.write(res.content)

    # process input x
    seqs = [x] if isinstance(x, str) else x

    # setup mode
    if use_filter:
        mode = "env" if use_env else "all"
    else:
        mode = "env-nofilter" if use_env else "nofilter"

    if use_pairing:
        mode = ""
        # greedy is default, complete was the previous behavior
        if pairing_strategy == "greedy":
            mode = "pairgreedy"
        elif pairing_strategy == "complete":
            mode = "paircomplete"
        if use_env:
            mode = mode + "-env"

    # define path
    path = f"{prefix}_{mode}"
    if not os.path.isdir(path):
        os.mkdir(path)

    # call mmseqs2 api
    tar_gz_file = f"{path}/out.tar.gz"
    N, REDO = 101, True

    # deduplicate and keep track of order
    seqs_unique = []
    # TODO this might be slow for large sets
    [seqs_unique.append(x) for x in seqs if x not in seqs_unique]
    Ms = [N + seqs_unique.index(seq) for seq in seqs]
    # lets do it!
    if not os.path.isfile(tar_gz_file):
        TIME_ESTIMATE = 150 * len(seqs_unique)
        with tqdm(total=TIME_ESTIMATE, bar_format=TQDM_BAR_FORMAT) as pbar:
            while REDO:
                pbar.set_description("SUBMIT")

                # Resubmit job until it goes through
                out = submit(seqs_unique, mode, N)
                while out["status"] in ["UNKNOWN", "RATELIMIT"]:
                    sleep_time = 5 + random.randint(0, 5)
                    logger.error(f"Sleeping for {sleep_time}s. Reason: {out['status']}")
                    # resubmit
                    time.sleep(sleep_time)
                    out = submit(seqs_unique, mode, N)

                if out["status"] == "ERROR":
                    msg = (
                        "MMseqs2 API is giving errors. Please confirm your "
                        " input is a valid protein sequence. If error persists, "
                        "please try again an hour later."
                    )
                    raise Exception(msg)

                if out["status"] == "MAINTENANCE":
                    msg = (
                        "MMseqs2 API is undergoing maintenance. "
                        "Please try again in a few minutes."
                    )
                    raise Exception(msg)

                # wait for job to finish
                ID, TIME = out["id"], 0
                logger.debug(f"MSA job submitted successfully with ID: {ID}")
                pbar.set_description(out["status"])
                while out["status"] in ["UNKNOWN", "RUNNING", "PENDING"]:
                    t = 5 + random.randint(0, 5)
                    logger.error(f"Sleeping for {t}s. Reason: {out['status']}")
                    time.sleep(t)
                    out = status(ID)
                    pbar.set_description(out["status"])
                    if out["status"] == "RUNNING":
                        TIME += t
                        pbar.update(n=t)

                if out["status"] == "COMPLETE":
                    logger.debug(f"MSA job completed successfully for ID: {ID}")
                    if TIME < TIME_ESTIMATE:
                        pbar.update(n=(TIME_ESTIMATE - TIME))
                    REDO = False

                if out["status"] == "ERROR":
                    REDO = False
                    msg = (
                        "MMseqs2 API is giving errors. Please confirm your "
                        " input is a valid protein sequence. If error persists, "
                        "please try again an hour later."
                    )
                    raise Exception(msg)

            # Download results
            download(ID, tar_gz_file)

    # prep list of a3m files
    if use_pairing:
        a3m_files = [f"{path}/pair.a3m"]
    else:
        a3m_files = [f"{path}/uniref.a3m"]
        if use_env:
            a3m_files.append(f"{path}/bfd.mgnify30.metaeuk30.smag30.a3m")

    # extract a3m files
    if any(not os.path.isfile(a3m_file) for a3m_file in a3m_files):
        with tarfile.open(tar_gz_file) as tar_gz:
            tar_gz.extractall(path)

    # gather a3m lines
    a3m_lines = {}
    for a3m_file in a3m_files:
        update_M, M = True, None
        for line in open(a3m_file, "r"):
            if len(line) > 0:
                if "\x00" in line:
                    line = line.replace("\x00", "")
                    update_M = True
                if line.startswith(">") and update_M:
                    M = int(line[1:].rstrip())
                    update_M = False
                    if M not in a3m_lines:
                        a3m_lines[M] = []
                a3m_lines[M].append(line)

    a3m_lines = ["".join(a3m_lines[n]) for n in Ms]
    return a3m_lines


def precompute_msa(
    sequence: str,
    out_path: Union[str, "os.PathLike[str]"],
    host_url: str = "https://api.colabfold.com",
    msa_server_username: Optional[str] = None,
    msa_server_password: Optional[str] = None,
) -> "str":
    """Run MMseqs2 for *sequence* and write the resulting MSA to *out_path*.

    This is the preferred entry-point for generating MSAs within the
    affinity-rescoring pipeline.  It calls :func:`run_mmseqs2` directly
    (no ``--use_msa_server`` subprocess flag), writes the raw a3m content
    to disk, and returns the absolute path so callers can inject it into
    Boltz YAML files.

    Parameters
    ----------
    sequence : str
        Amino-acid sequence to search.
    out_path : str or Path
        Destination file.  Should end in ``.a3m``.
    host_url : str
        ColabFold API endpoint (default: ``https://api.colabfold.com``).
    msa_server_username, msa_server_password : str, optional
        Credentials for servers that require HTTP basic auth.

    Returns
    -------
    str
        Absolute path to the written MSA file (same as *out_path* resolved).
    """
    import shutil
    from pathlib import Path as _Path

    out_path = _Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Temp prefix for intermediate MMseqs2 download directories
    prefix = str(out_path.parent / f"_mmseqs2_tmp_{out_path.stem}")

    logger.info(
        "Generating MSA via ColabFold MMseqs2 server "
        "(sequence length=%d) → %s",
        len(sequence),
        out_path,
    )

    a3m_lines = run_mmseqs2(
        x=[sequence],
        prefix=prefix,
        use_env=True,
        use_filter=True,
        use_pairing=False,
        host_url=host_url,
        msa_server_username=msa_server_username,
        msa_server_password=msa_server_password,
    )

    # run_mmseqs2 returns list[str] — one a3m block per input sequence
    a3m_content = a3m_lines[0] if a3m_lines else ""
    out_path.write_text(a3m_content)
    logger.info("MSA written to %s (%d bytes)", out_path, out_path.stat().st_size)

    # Remove intermediate MMseqs2 temp directories
    for suffix in ("_env", "_all", "_env-nofilter", "_nofilter",
                   "_pairgreedy-env", "_pairgreedy", "_paircomplete-env", "_paircomplete"):
        tmp_dir = f"{prefix}{suffix}"
        if os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)

    return str(out_path.resolve())


# ─── Standalone CLI ─────────────────────────────────────────────────────────
#
# Usage:
#     python -m boltz.affinity_rescoring.mmseqs2 \
#         --sequence MKTL... --out /shared/msa_cache/<name>.a3m
#
# Pre-compute exactly one MSA per unique sequence and stash it in your
# shared cache. This is the *only* sanctioned way to call the ColabFold
# MMseqs2 endpoint from this fork — every other pipeline reads the
# resulting .a3m via boltz.affinity_rescoring.msa_cache.find_msa.

def _cli_main() -> int:  # pragma: no cover - thin wrapper
    import argparse

    p = argparse.ArgumentParser(
        description=(
            "Pre-compute a Boltz MSA for a single sequence via the "
            "ColabFold MMseqs2 server. Writes a canonical "
            "<sha256[:16]>.a3m by default and symlinks any user-provided "
            "filename so downstream code paths discover it."
        ),
    )
    p.add_argument("--sequence", "-s",
                   help="Amino-acid sequence to search (mutually exclusive "
                        "with --sequence-file).")
    p.add_argument("--sequence-file",
                   help="Path to a file whose contents are the sequence.")
    p.add_argument("--out", "-o", required=False,
                   help="Output .a3m path. If omitted, defaults to "
                        "<cache-dir>/<sha256[:16]>.a3m.")
    p.add_argument("--cache-dir", "-d", default=None,
                   help="Directory for the canonical hash-named MSA file. "
                        "Defaults to the parent of --out (or the first "
                        "entry of $BOLTZ_MSA_CACHE_DIR).")
    p.add_argument("--host-url", default="https://api.colabfold.com",
                   help="ColabFold API endpoint.")
    p.add_argument("--username", default=None)
    p.add_argument("--password", default=None)
    p.add_argument("--chain-id", default=None,
                   help="Optional chain id used for legacy-name symlinks.")
    p.add_argument("--target", default=None,
                   help="Optional target name used for legacy "
                        "<target>_<chain>.a3m symlinks.")
    p.add_argument("--force", action="store_true", default=False,
                   help="Re-fetch even if a cached file already exists.")
    args = p.parse_args()

    if not (args.sequence or args.sequence_file):
        p.error("--sequence or --sequence-file is required")
    if args.sequence and args.sequence_file:
        p.error("--sequence and --sequence-file are mutually exclusive")

    sequence = args.sequence
    if args.sequence_file:
        from pathlib import Path as _P
        sequence = _P(args.sequence_file).read_text().strip()

    from boltz.affinity_rescoring.msa_cache import (
        canonical_msa_path,
        default_cache_dirs,
        find_msa,
        write_legacy_symlinks,
    )
    from pathlib import Path as _P

    cache_dir = None
    if args.cache_dir:
        cache_dir = _P(args.cache_dir)
    elif args.out:
        cache_dir = _P(args.out).expanduser().resolve().parent
    else:
        dirs = default_cache_dirs()
        if not dirs:
            p.error(
                "No --cache-dir, no --out, and $BOLTZ_MSA_CACHE_DIR is "
                "unset — nowhere to write the MSA."
            )
        cache_dir = dirs[0]
    cache_dir.mkdir(parents=True, exist_ok=True)

    canonical = canonical_msa_path(sequence, cache_dir)

    if not args.force:
        hit = find_msa(
            sequence=sequence,
            msa_dirs=[cache_dir],
            chain_id=args.chain_id,
            target=args.target,
        )
        if hit is not None:
            print(f"[cached] {hit}")
            if args.out:
                out = _P(args.out).expanduser()
                if not out.exists():
                    out.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        out.symlink_to(hit)
                    except OSError:
                        out.write_bytes(hit.read_bytes())
            return 0

    precompute_msa(
        sequence=sequence,
        out_path=canonical,
        host_url=args.host_url,
        msa_server_username=args.username,
        msa_server_password=args.password,
    )
    write_legacy_symlinks(canonical, chain_id=args.chain_id, target=args.target)
    if args.out:
        out = _P(args.out).expanduser()
        if out.resolve() != canonical.resolve() and not out.exists():
            out.parent.mkdir(parents=True, exist_ok=True)
            try:
                out.symlink_to(canonical)
            except OSError:
                out.write_bytes(canonical.read_bytes())
    print(str(canonical.resolve()))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_cli_main())