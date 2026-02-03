#!/usr/bin/env bash
set -euo pipefail

# Raspberry Pi System Probe
# Usage: ./scripts/rpi_sys_probe.sh [output_file]
# Default output: debug_dumps/rpi_sys_profile.txt

OUT=${1:-debug_dumps/rpi_sys_profile.txt}
mkdir -p "$(dirname "$OUT")"

echo "# Raspberry Pi System Profile - $(date)" >"$OUT"

log() { echo "---- $* ----" | tee -a "$OUT"; }
run() { echo "\n$ $*" | tee -a "$OUT"; bash -lc "$*" 2>&1 | tee -a "$OUT"; }

log "Host and OS"
run "date"
run "uname -a"
[ -f /etc/os-release ] && run "cat /etc/os-release"

log "CPU (lscpu, cpuinfo)"
if command -v lscpu >/dev/null 2>&1; then run "lscpu"; else echo "lscpu not found" | tee -a "$OUT"; fi
run "grep -m1 '^model name' /proc/cpuinfo || true"
run "grep -m1 '^Hardware' /proc/cpuinfo || true"
run "grep -m1 '^Revision' /proc/cpuinfo || true"
run "grep -m1 '^Features' /proc/cpuinfo || true"

log "Caches and cache line size"
run "getconf LEVEL1_DCACHE_LINESIZE || true"
run "getconf -a | grep -E 'CACHE|LEVEL' || true"

log "Memory"
run "free -h"
run "grep -E 'MemTotal|MemFree|SwapTotal|SwapFree' /proc/meminfo"

log "Storage"
run "df -h"

log "CPU frequency and governor"
for f in /sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_governor; do
  [ -f "$f" ] && { echo -n "$f: " | tee -a "$OUT"; cat "$f" | tee -a "$OUT"; } || true
done
for f in /sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_cur_freq; do
  [ -f "$f" ] && { echo -n "$f: " | tee -a "$OUT"; cat "$f" | tee -a "$OUT"; } || true
done
run "cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_governors || true"

log "Thermals (vcgencmd)"
if command -v vcgencmd >/dev/null 2>&1; then
  run "vcgencmd measure_temp"
  run "vcgencmd get_throttled"
else
  echo "vcgencmd not installed (install: sudo apt-get install -y libraspberrypi-bin)" | tee -a "$OUT"
fi

log "Toolchain"
run "which gcc || true"
run "gcc --version || true"
run "ldd --version || true"
run "ld --version | head -n1 || true"
run "which make || true"

log "OpenMP availability"
cat >/tmp/omp_test.c <<'EOF'
#include <stdio.h>
#ifdef _OPENMP
#include <omp.h>
#endif
int main(){
#ifdef _OPENMP
    printf("OpenMP: yes, max_threads=%d\n", omp_get_max_threads());
#else
    printf("OpenMP: no\n");
#endif
    return 0;
}
EOF
if command -v gcc >/dev/null 2>&1; then
  if gcc -fopenmp /tmp/omp_test.c -o /tmp/omp_test 2>>"$OUT"; then
    /tmp/omp_test | tee -a "$OUT"
  else
    echo "OpenMP: compile failed (-fopenmp)" | tee -a "$OUT"
  fi
else
  echo "gcc not found" | tee -a "$OUT"
fi
rm -f /tmp/omp_test /tmp/omp_test.c

log "BLAS presence"
run "ldconfig -p | grep -E 'openblas|blas' || true"
run "dpkg -l | grep -E 'openblas|blas|atlas' || true"
run "pkg-config --modversion openblas || true"

log "Python (optional)"
run "python3 --version || true"
run "pip3 --version || true"

echo "\nProfile saved to: $OUT" | tee -a "$OUT"
