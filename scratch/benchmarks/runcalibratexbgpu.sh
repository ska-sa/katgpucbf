#!/bin/bash

set -e -u

runs=$1
n=$2
max_ssh_retries=${MAX_SSH_RETRIES:-3}
ssh_retry_delay=${SSH_RETRY_DELAY:-30}

send_email() {
  local subject=$1
  local body=$2
  ssmtp eemmerich@sarao.ac.za <<EOF
Subject: ${subject}
${body}
EOF
}

is_ssh_failure() {
  local logfile=$1
  grep -qiE 'asyncssh|SSH.*(fail|error|closed|lost|reset|disconnect)|ConnectionLost|DisconnectError|ChannelOpenError' "$logfile"
}

# n = 2: --high 5850000000 --low 5600000000
# n = 4: --high 3350000000 --low 3100000000
run_benchmark() {
  local outfile=$1
  ./benchmark_xbgpu.py -vvv -n "$n" --substreams 1024 --channels 32768 --array-size 680 --calibrate-repeat 2 --calibrate --high 5850000000 --low 5600000000 --multicast-groups 239.192.160.0/19 >"$outfile" 2>&1
}

for i in $(seq 1 "$runs"); do
  outfile="xbgpu_calibration/calibration-n${n}-${i}.txt"
  attempt=1
  while true; do
    if run_benchmark "$outfile"; then
      send_email "The test is progressing" \
        "cbfgpu004 has completed the test for calibration-n${n}-${i}.txt"
      break
    fi
    if is_ssh_failure "$outfile" && [ "$attempt" -lt "$max_ssh_retries" ]; then
      send_email "SSH failure, retrying calibration run" \
        "cbfgpu004: SSH connection failed on calibration-n${n}-${i}.txt (attempt ${attempt}/${max_ssh_retries}). Retrying in ${ssh_retry_delay}s."
      attempt=$((attempt + 1))
      sleep "$ssh_retry_delay"
    else
      if is_ssh_failure "$outfile"; then
        send_email "SSH failure, calibration run abandoned" \
          "cbfgpu004: SSH connection failed on calibration-n${n}-${i}.txt after ${max_ssh_retries} attempts. See ${outfile} for details."
      fi
      exit 1
    fi
  done
done

echo "The test is complete" | ssmtp eemmerich@sarao.ac.za
