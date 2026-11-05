#!/bin/bash
# Run by a systemd timer: build the newest published run if it's new, then serve it. A failed build leaves the current store serving.
set -euo pipefail
exec 9>/run/pirate-nam-wms.lock; flock -n 9 || exit 0
STORE=$(runuser -u ubuntu -- /srv/wms/venv/bin/python /srv/wms/build_store.py /srv/wms | tail -1)
[ -n "$STORE" ] || exit 0
ln -sfn "$STORE" /srv/wms/current.new; mv -T /srv/wms/current.new /srv/wms/current
systemctl restart pirate-nam-wms
# Cached tiles belong to the old run. Requests without a time would otherwise keep showing it.
find /var/cache/nginx/wms -type f -delete
find /srv/wms/stores -mindepth 1 -maxdepth 1 ! -path "$STORE" -exec rm -rf {} +
echo "serving $STORE"
