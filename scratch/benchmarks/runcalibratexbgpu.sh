#!/bin/bash
# Copyright (c) 2026 National Research Foundation (SARAO)
#
# Licensed under the BSD 3-Clause License (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy
# of the License at
#
#   https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

set -e -u

runs=$1
n=$2
low_freq=$3
high_freq=$4

max_ssh_retries=${MAX_SSH_RETRIES:-3}
ssh_retry_delay=${SSH_RETRY_DELAY:-30}

EMAIL_ADDRESS=${EMAIL_ADDRESS:-"$(whoami)@sarao.ac.za"}
send_email() {
  local subject=$1
  local body=$2
  ssmtp ${EMAIL_ADDRESS}<<EOF
Subject: ${subject}
${body}
EOF
}

is_ssh_failure() {
  local logfile=$1
  grep -qiE 'asyncssh|SSH.*(fail|error|closed|lost|reset|disconnect)|ConnectionLost|DisconnectError|ChannelOpenError' "$logfile"
}

run_benchmark() {
  local outfile=$1
  ./benchmark_xbgpu.py -vvv -n "$n" --substreams 1024 --channels 32768 --array-size 680 --calibrate-repeat 2 --calibrate --low "$low_freq" --high "$high_freq" --multicast-groups 239.192.160.0/19 >"$outfile" 2>&1
}

for i in $(seq 1 "$runs"); do
  outfile="xbgpu_calibration/calibration-n${n}-${i}.txt"
  attempt=1
  while true; do
    if run_benchmark "$outfile"; then
      send_email "The test is progressing" \
        "$(hostname -s) has completed the test for calibration-n${n}-${i}.txt"
      break
    fi
    if is_ssh_failure "$outfile" && [ "$attempt" -lt "$max_ssh_retries" ]; then
      send_email "SSH failure, retrying calibration run" \
        "$(hostname -s): SSH connection failed on calibration-n${n}-${i}.txt (attempt ${attempt}/${max_ssh_retries}). Retrying in ${ssh_retry_delay}s."
      attempt=$((attempt + 1))
      sleep "$ssh_retry_delay"
    else
      if is_ssh_failure "$outfile"; then
        send_email "SSH failure, calibration run abandoned" \
          "$(hostname -s): SSH connection failed on calibration-n${n}-${i}.txt after ${max_ssh_retries} attempts. See ${outfile} for details."
      fi
      exit 1
    fi
  done
done

send_email "The test is complete" "The test is complete for ${n} antennas with low frequency ${low_freq} and high frequency ${high_freq} on $(hostname -s)"
