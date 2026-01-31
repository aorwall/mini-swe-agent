"""Utilities for processing command output."""

import base64
import logging
import time

logger = logging.getLogger("minisweagent.output_utils")

# Max bytes of base64 per chunk to stay well under ARG_MAX (~2MB on Linux)
_B64_CHUNK_SIZE = 500_000

# Number of tail lines to keep in output after truncation
_TAIL_LINES = 50

# Max characters to keep in the tail (guards against very long lines)
_MAX_TAIL_CHARS = 2_000

# Character threshold: also trigger truncation if output exceeds this many chars,
# even if line count is under threshold (guards against few but very long lines)
_CHAR_THRESHOLD = 10_000


def process_large_output(output: dict, threshold: int, environment=None) -> dict:
    """If threshold > 0 and output exceeds that many lines, save to file via the environment and add file path.

    After saving, the output["output"] field is replaced with only the last _TAIL_LINES lines
    so the full string never enters the message pipeline.

    Args:
        output: Dictionary containing command output with 'output' key.
        threshold: Line threshold. If > 0 and output exceeds this many lines, save to file.
        environment: Environment to write the file into (e.g., DockerEnvironment).
            Must have an execute(action) method. If None, file writing is skipped
            but metadata is still added.

    Returns:
        Modified output dict with output_file_path, output_line_count, output_char_count
        if output was saved to file, otherwise returns original output unchanged.
    """
    if threshold <= 0:
        return output

    raw_output = output.get("output", "")
    lines = raw_output.split("\n")
    line_count = len(lines)

    if line_count <= threshold and len(raw_output) <= _CHAR_THRESHOLD:
        return output

    output = output.copy()
    output["output_line_count"] = line_count
    output["output_char_count"] = len(raw_output)

    if environment is not None:
        _save_output_to_file(raw_output, output, environment)

    # Truncate output to last _TAIL_LINES lines, then cap by characters
    tail = "\n".join(lines[-_TAIL_LINES:])
    if len(tail) > _MAX_TAIL_CHARS:
        tail = tail[-_MAX_TAIL_CHARS:]
    output["output"] = tail

    return output


def _save_output_to_file(raw_output: str, output: dict, environment) -> None:
    """Save raw_output to a file in the environment, set output["output_file_path"] on success."""
    timestamp = int(time.time() * 1000)
    filepath = f"/tmp/bash_output_{timestamp}.txt"
    b64_path = f"{filepath}.b64"

    try:
        encoded = base64.b64encode(raw_output.encode("utf-8")).decode("ascii")

        # Write base64 data in chunks to avoid exceeding ARG_MAX
        for i in range(0, len(encoded), _B64_CHUNK_SIZE):
            chunk = encoded[i : i + _B64_CHUNK_SIZE]
            op = ">" if i == 0 else ">>"
            result = environment.execute({"command": f"printf '%s' '{chunk}' {op} {b64_path}"})
            if result.get("returncode") != 0:
                logger.warning("Failed to write base64 chunk to file: %s", result.get("output", ""))
                return

        # Decode and clean up
        result = environment.execute({"command": f"base64 -d {b64_path} > {filepath} && rm -f {b64_path}"})
        if result.get("returncode") == 0:
            output["output_file_path"] = filepath
        else:
            logger.warning("Failed to decode base64 output file: %s", result.get("output", ""))
            environment.execute({"command": f"rm -f {filepath} {b64_path}"})
    except Exception:
        logger.warning("Failed to write large output to file in environment", exc_info=True)
