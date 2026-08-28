#!/usr/bin/env bash
set -euo pipefail

# Capture committed, dirty, and untracked state before anyone edits/restarts the
# old checkout. Review untracked files for credentials before transferring them.

repo="${K3_REPO:?set K3_REPO to the old checkout}"
out="${K3_SOURCE_EXPORT:?set K3_SOURCE_EXPORT to a new output directory}"
mkdir -p "${out}"

head="$(git -C "${repo}" rev-parse HEAD)"
printf '%s\n' "${head}" >"${out}/HEAD.txt"
git -C "${repo}" status --short --untracked-files=all >"${out}/status-short.txt"
git -C "${repo}" diff --binary >"${out}/working-tree.patch"
git -C "${repo}" diff --cached --binary >"${out}/index.patch"
git -C "${repo}" submodule status --recursive >"${out}/submodules.txt" || true
git -C "${repo}" bundle create "${out}/head.bundle" HEAD
git -C "${repo}" archive --format=tar.gz \
    --prefix=tracked-tree/ -o "${out}/tracked-tree.tar.gz" HEAD

git -C "${repo}" ls-files --others --exclude-standard -z \
    >"${out}/untracked-files.nul"

# GNU tar accepts a NUL-separated file list. The resulting archive can contain
# secrets, large logs, cores, or model fragments; inspect before sharing.
if [[ -s "${out}/untracked-files.nul" ]]; then
    tar -C "${repo}" --null --files-from="${out}/untracked-files.nul" \
        -czf "${out}/untracked-files.REVIEW-BEFORE-SHARING.tar.gz"
fi

echo "Source state exported to ${out}. Review untracked archive before sharing."

