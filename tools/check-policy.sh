#!/usr/bin/env bash
# Gagal bila ada berkas teks yang tracked berisi em-dash (U+2014) atau
# karakter pengganti (U+FFFD). Aturan repo: tanpa em-dash, encoding valid.
set -u
status=0
files=$(git ls-files -- . ':!:*.png' ':!:*.webp' ':!:*.ico' ':!:*.pt' ':!:*.pdf' ':!:*.tif' ':!:*.lock' ':!:*.jar')
for f in $files; do
  [ -f "$f" ] || continue
  if grep -Iq . "$f" 2>/dev/null; then
    if grep -nP '\x{2014}|\x{FFFD}' "$f" >/dev/null 2>&1; then
      echo "Dilarang: em-dash atau karakter pengganti di $f"
      grep -nP '\x{2014}|\x{FFFD}' "$f" | head -3
      status=1
    fi
  fi
done
exit $status
