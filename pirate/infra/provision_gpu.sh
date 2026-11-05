#!/bin/bash
# Copy the WOOF-NAM + GSI setup from nam-woof to a new GPU machine, directly between them. Usage: provision_gpu.sh TARGET_IP
# The hop key lives on nam-woof only for this copy; these machines hold no credentials.
set -uo pipefail
T=$1; K=${NAM_SSH_KEY:?}; SRC=${GPU_SOURCE_IP:?}; SSH="ssh -i $K -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15"
for a in $(seq 1 60); do $SSH ubuntu@$T 'cloud-init status --wait >/dev/null 2>&1; which docker rsync' >/dev/null 2>&1 && break; sleep 15; done
$SSH ubuntu@$SRC 'test -f ~/.ssh/hop || ssh-keygen -q -t ed25519 -N "" -C hop -f ~/.ssh/hop; cat ~/.ssh/hop.pub' 2>/dev/null | tail -1 | $SSH ubuntu@$T 'cat >> ~/.ssh/authorized_keys'
$SSH ubuntu@$T 'sudo mkdir -p /woof /nam && sudo chown ubuntu /woof /nam; sudo usermod -aG docker ubuntu; nvidia-smi -L; df -h / | tail -1' 2>&1 | grep -vE "Ubuntu 24|documentation"
T0=$(date +%s)
$SSH ubuntu@$SRC "H='ssh -i ~/.ssh/hop -o StrictHostKeyChecking=accept-new'; rsync -aH -e \"\$H\" --exclude tmpnwprd /nam2/ ubuntu@$T:/nam/; rsync -aH -e \"\$H\" --exclude work/runs --exclude work/scratch --exclude '*.log' --exclude 'inc.tm*' /woof/ ubuntu@$T:/woof/ && docker save namhar | gzip -1 | \$H ubuntu@$T 'gunzip | sudo docker load'" 2>&1 | grep -vE "Ubuntu 24|documentation" | tail -2
RC=$?
$SSH ubuntu@$T 'mkdir -p /nam/tmpnwprd /woof/work/runs; du -sh /nam /woof; docker images --format "{{.Repository}}" 2>/dev/null || sudo docker images --format "{{.Repository}}"; /woof/venv/bin/python -c "import cupy; print(cupy.cuda.runtime.getDeviceCount(), cupy.ones(3).sum())"' 2>&1 | grep -vE "Ubuntu 24|documentation"
echo "provisioned $T rc=$RC in $(( $(date +%s)-T0 )) s"
