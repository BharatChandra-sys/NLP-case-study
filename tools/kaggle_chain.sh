#!/bin/sh
for n in "$@"; do
  py tools/kaggle_sync.py push "$n" 2>&1 | tail -1
  ping -n 41 127.0.0.1 >/dev/null
  while :; do
    s=$(py -m kaggle kernels status "bharatchandra1234/$(py -c "import sys;sys.path.insert(0,'tools');import kaggle_sync as k;print(next(v for a,v in k.KERNELS.items() if a.startswith('$n')))")" 2>&1 | tail -1)
    case "$s" in *COMPLETE*|*ERROR*|*CANCEL*) echo "$n: $s"; break;; esac
    ping -n 46 127.0.0.1 >/dev/null
  done
  py tools/kaggle_sync.py pull "$n" >/dev/null 2>&1
done
