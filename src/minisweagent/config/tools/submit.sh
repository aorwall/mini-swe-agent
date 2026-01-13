#!/bin/bash
# Validates and submits a git patch
# Usage: /tools/submit.sh <patch_file>
#        /tools/submit.sh --abort

# Handle abort case - empty submission
if [ "$1" = "--abort" ]; then
    echo "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
    exit 0
fi

PATCH_FILE="${1:-patch.txt}"

# Check file exists
if [ ! -f "$PATCH_FILE" ]; then
    echo "ERROR: Patch file '$PATCH_FILE' does not exist"
    exit 1
fi

# Check file is non-empty
if [ ! -s "$PATCH_FILE" ]; then
    echo "ERROR: Patch file '$PATCH_FILE' is empty"
    exit 1
fi

# Validate git diff format
if ! head -1 "$PATCH_FILE" | grep -qE '^diff --git'; then
    echo "ERROR: '$PATCH_FILE' is not a valid git diff"
    echo "Expected 'diff --git ...' but got: $(head -1 "$PATCH_FILE")"
    echo "Use 'git diff > patch.txt' to create the patch"
    exit 1
fi

# Check patch matches current changes (reverse apply = can unapply from current state)
git apply --check --reverse "$PATCH_FILE" 2>&1 || exit 1

# All validations passed - output patch and signal completion
cat "$PATCH_FILE"
echo "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
