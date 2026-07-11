#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_dir="${repo_root}/data/p6_real_samples"
output_path="${output_dir}/000000039769.jpg"
temporary_path="${output_path}.tmp"
source_url="http://images.cocodataset.org/val2017/000000039769.jpg"
expected_sha256="dea9e7ef97386345f7cff32f9055da4982da5471c48d575146c796ab4563b04e"

mkdir -p "${output_dir}"
rm -f "${temporary_path}"
curl --fail --location --retry 3 --output "${temporary_path}" "${source_url}"
printf '%s  %s\n' "${expected_sha256}" "${temporary_path}" | sha256sum --check --status
mv "${temporary_path}" "${output_path}"
printf 'verified P6 real sample: %s\n' "${output_path}"
