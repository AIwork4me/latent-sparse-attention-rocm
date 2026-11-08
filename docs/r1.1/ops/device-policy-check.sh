#!/usr/bin/env bash
# device-policy-check.sh — pure, unit-testable core of migrate-runner.sh §7's
# device-access validation (PR #6 final correction).
#
# Why this exists (defects found in the 7f1c971 owner review):
#   1. The transient systemd-run validation context transcribed DevicePolicy=closed
#      from the installed unit but NONE of its DeviceAllow= entries — under
#      `closed` an empty allowlist is a zero-device sandbox, so the in-context
#      GPU/Triton probe could never pass (guaranteed false abort).
#   2. The old check compared `ls /dev/dri` filenames against the allowlist.
#      DevicePolicy is enforced by the cgroup eBPF filter on open(2), not by
#      hiding nodes — visibility comparisons prove nothing about ACCESS.
#
# Subcommands (file-in/file-out, no systemctl calls — CI-testable, see
# tests/test_r11_device_policy.py):
#   parity <unit-file> <show-file>      exit 0 iff the device access policy
#                                       (DevicePolicy + full DeviceAllow set,
#                                       paths AND modes) in the unit file equals
#                                       what `systemctl show -p DevicePolicy
#                                       -p DeviceAllow` reports for the loaded
#                                       unit; catches drop-ins / reload lag /
#                                       transcription loss. exit 1 with a named
#                                       diff, exit 2 on malformed input.
#   emit-entries <show-file>            print one canonical "<path> <mode>" line
#                                       per DeviceAllow entry (modes normalized
#                                       to r/w/m char order; omitted mode =>
#                                       rwm). migrate-runner.sh feeds each line
#                                       to systemd-run --property=DeviceAllow=.
#   nodes <unit-file> [--dev-root DIR]  exit 0 iff every /dev/-path DeviceAllow
#                                       entry exists under DIR (default /dev) —
#                                       the "unit vs actual GPU device nodes"
#                                       compatibility check (host renumbered
#                                       card0/card1 after reboot = loud abort,
#                                       not a mystery EPERM). char-*/block-*
#                                       group specifiers are skipped (they name
#                                       device groups, not nodes).
#
# Exit codes: 0 = check passed · 1 = policy/node mismatch (ABORT) · 2 = malformed
# input (SYNTAX-ABORT). List-property parsing handles BOTH serializations systemd
# produces: space-joined on one line and one line per entry.
set -euo pipefail
export LC_ALL=C

die()  { echo "ABORT: $*" >&2; exit 1; }
die2() { echo "SYNTAX-ABORT: $*" >&2; exit 2; }

# parse_entries: unit-file/show capture on stdin -> sorted canonical "<path> <mode>"
# lines on stdout. All validation and canonicalization live inside the awk process
# so failures propagate (pipefail) and stderr reaches the caller unchanged.
parse_entries() {
  awk '
    function canon(m) {
      out = ""
      if (index(m, "r")) out = out "r"
      if (index(m, "w")) out = out "w"
      if (index(m, "m")) out = out "m"
      return out
    }
    /^DeviceAllow[=:]/ {
      line = $0
      sub(/^DeviceAllow[=:]/, "", line)
      n = split(line, t, /[ \t]+/)
      for (i = 1; i <= n; i++) {
        tok = t[i]
        if (tok == "") continue
        if (tok ~ /^\/dev\/|^char-|^block-/) {
          if (path != "") printf "%s %s\n", path, mode   # previous entry had no mode
          path = tok; mode = "rwm"                       # omitted mode = rwm
        } else {
          if (path == "") { print "SYNTAX-ABORT: dangling DeviceAllow mode token \x27" tok "\x27 (no device specifier precedes it)" > "/dev/stderr"; bad = 1; exit 2 }
          if (tok !~ /^[rwm]+$/) { print "SYNTAX-ABORT: unknown DeviceAllow mode characters in \x27" tok "\x27 (only r/w/m allowed)" > "/dev/stderr"; bad = 1; exit 2 }
          mode = canon(tok)
        }
      }
    }
    END { if (!bad && path != "") printf "%s %s\n", path, mode }
  ' | sort -u
}

# parse_file <file> [what] -> entries on stdout, empty allowed (parity compares
# allowlists; emptiness semantics are the CALLER's policy decision).
parse_file() {
  local out rc=0
  [ -f "$1" ] || die2 "${2:-file} not found: $1"
  out=$(parse_entries < "$1") || rc=$?   # SYNTAX-ABORT already printed to stderr
  [ "$rc" -eq 0 ] || exit 2
  [ -n "$out" ] || return 0
  printf '%s\n' "$out"
}

policy_from_file() {  # last DevicePolicy= assignment wins (systemd semantics)
  # Tolerant pipeline: a missing file / missing DevicePolicy= line must reach the
  # CALLER's [ -n ] check and die loudly (rc 2) — grep's own rc under pipefail
  # would otherwise kill the script silently here (review finding #1).
  [ -f "$1" ] || die2 "${2:-file} not found: $1"
  grep -E '^[[:space:]]*DevicePolicy=' "$1" 2>/dev/null | tail -1 | cut -d= -f2- | tr -d '[:space:]' || true
}

cmd_parity() {
  [ $# -eq 2 ] || die2 "usage: parity <unit-file> <show-file>"
  local unit_file=$1 show_file=$2 up sp uent sent only_unit only_live
  up=$(policy_from_file "$unit_file" "unit file"); [ -n "$up" ] || die2 "no DevicePolicy= in unit file $unit_file"
  sp=$(policy_from_file "$show_file" "show capture"); [ -n "$sp" ] || die2 "no DevicePolicy= line in show capture $show_file (capture with: systemctl show <unit> -p DevicePolicy -p DeviceAllow)"
  uent=$(parse_file "$unit_file" "unit file")
  sent=$(parse_file "$show_file" "show capture")
  if [ "$up" != "$sp" ]; then
    die "DevicePolicy drift: unit file '$up' vs live unit '$sp' — reconcile (drop-in? daemon-reload lag?) before validating"
  fi
  case "$up" in closed|strict)
    [ -n "$sent" ] || die "DevicePolicy=$up with an EMPTY live DeviceAllow list = zero-device sandbox (the defect-1 class)" ;;
  esac
  only_unit=$(comm -23 <(echo "$uent") <(echo "$sent"))
  only_live=$(comm -13 <(echo "$uent") <(echo "$sent"))
  [ -z "$only_unit" ] || die "DeviceAllow entries in the unit file but NOT in the live unit:
$only_unit"
  [ -z "$only_live" ] || die "DeviceAllow entries in the live unit but NOT in the unit file:
$only_live"
  echo "device-policy parity OK: DevicePolicy=$sp, $(echo "$sent" | wc -l) DeviceAllow entries identical (unit file == live unit)"
}

cmd_emit() {
  [ $# -eq 1 ] || die2 "usage: emit-entries <show-file>"
  local out
  out=$(parse_file "$1" "show capture")
  [ -n "$out" ] || die2 "no DeviceAllow entries in show capture $1 — refusing to emit an empty transcription (would recreate the defect-1 zero-device context)"
  printf '%s\n' "$out"
}

cmd_nodes() {
  local unit_file="" dev_root=/dev entries missing="" path _mode
  while [ $# -gt 0 ]; do
    case "$1" in
      --dev-root)   [ $# -ge 2 ] || die2 "--dev-root needs a value"; dev_root=$2; shift 2 ;;
      --dev-root=*) dev_root=${1#*=}; shift ;;
      *)
        if [ -z "$unit_file" ]; then unit_file=$1; else die2 "unexpected argument: $1"; fi
        shift ;;
    esac
  done
  [ -n "$unit_file" ] || die2 "usage: nodes <unit-file> [--dev-root DIR]"
  entries=$(parse_file "$unit_file" "unit file")
  [ -n "$entries" ] || die2 "no DeviceAllow entries in unit file $unit_file — DevicePolicy=closed with an empty allowlist is a zero-device sandbox (defect 1 class)"
  while read -r path _mode; do
    case "$path" in
      char-*|block-*) continue ;;   # device-group globs: no node to stat
      /dev/*) ;;
      *) die2 "DeviceAllow specifier '$path' is neither a /dev/ path nor a char-/block- group" ;;
    esac
    [ -e "$dev_root${path#/dev}" ] || missing+="$path "
  done < <(echo "$entries")
  [ -z "$missing" ] || die "DeviceAllow nodes missing under $dev_root (host GPU nodes changed? renumbered after reboot?): $missing— fix the unit, then re-run with --resume"
  echo "device nodes OK: every /dev-path DeviceAllow entry exists under $dev_root"
}

case "${1:-}" in
  parity)       shift; cmd_parity "$@" ;;
  emit-entries) shift; cmd_emit "$@" ;;
  nodes)        shift; cmd_nodes "$@" ;;
  *) die2 "usage: $0 {parity <unit-file> <show-file> | emit-entries <show-file> | nodes <unit-file> [--dev-root DIR]}" ;;
esac
