#!/usr/bin/env bash
#
# cluster-launch.sh -- fan out cado-nfs-client-rs across a cluster (Track 3.4).
#
# CADO-NFS distributes sieving by running many work-unit clients against one
# server. This helper starts the static Rust client (cado-nfs-client-rs) on a
# list of hosts (over SSH) or through Slurm (srun/sbatch), all pointed at the
# same server URL with the same TLS pinning -- so you do not hand-start clients
# one machine at a time.
#
# It assumes, like CADO's own distributed mode, SSH public-key auth to the hosts
# and that the client binary is reachable on each host (a shared filesystem, or
# pre-deployed to the same path -- see --client-bin). The server must already be
# running (cado-nfs.py prints "server.address=... server.port=..."; or run the
# Rust server directly). Cert pinning: pass the server's --certsha1 (cado-nfs.py
# prints the certificate SHA1) so clients trust it without a CA file.
#
# Examples:
#   # 4 clients each on two hosts, over SSH, with cert pinning:
#   scripts/cluster-launch.sh --server https://head:4242 --certsha1 AB12.. \
#       --hosts node01,node02 --clients-per-host 4
#
#   # one client per Slurm task across an allocation:
#   scripts/cluster-launch.sh --server https://head:4242 --certsha1 AB12.. \
#       --slurm --ntasks 16
#
#   # GPU-aware: one client pinned per GPU on each host (GPU-prefactor/cofactor):
#   scripts/cluster-launch.sh --server https://head:4242 --hosts gpu01,gpu02 \
#       --gpus-per-node 4
#
#   # batch cluster: submit an sbatch JOB ARRAY, one task per node, 4 GPU clients each:
#   scripts/cluster-launch.sh --server https://head:4242 --sbatch --nodes 8 \
#       --gpus-per-node 4 --partition gpu --time 24:00:00
#
#   # PBS/Torque cluster: the same as a qsub job array (--partition is the queue):
#   scripts/cluster-launch.sh --server https://head:4242 --pbs --nodes 8 \
#       --gpus-per-node 4 --partition gpu --time 24:00:00
#
#   # stop everything started over SSH:
#   scripts/cluster-launch.sh --hosts node01,node02 --stop
#
set -euo pipefail

PROG=$(basename "$0")
SERVER=""
CERTSHA1=""
HOSTS=""
HOSTFILE=""
SLURM=0
SBATCH=0
PBS=0
NTASKS=0
NODES=0
GPUS_PER_NODE=0
PARTITION=""
TIMELIMIT=""
JOBNAME="cado-sieve"
CLIENTS_PER_HOST=1
CLIENT_BIN=""
ARCH=""
SSH_OPTS="-o BatchMode=yes -o StrictHostKeyChecking=accept-new"
STOP=0
DRYRUN=0
EXTRA=()

usage() {
    sed -n '3,37p' "$0" | sed 's/^# \{0,1\}//'
    cat <<EOF

Options:
  --server URL          work-unit server URL (required unless --stop)
  --certsha1 HEX        pin the server cert by SHA1 (recommended over TLS)
  --hosts h1,h2,...     comma-separated SSH hosts
  --hostfile FILE       file with one host per line (# comments allowed)
  --clients-per-host N  SSH clients to start per host (default 1)
  --gpus-per-node N     start one GPU-pinned client per GPU (CUDA_VISIBLE_DEVICES
                        0..N-1); applies to SSH, --slurm and --sbatch. One client
                        per GPU is the right placement for GPU-prefactor/cofactor
                        clients. Overrides --clients-per-host when set.
  --slurm               launch via Slurm srun (interactive, holds the allocation)
  --sbatch              generate + submit a Slurm sbatch JOB ARRAY (batch; one
                        array task per node, each starting the per-node clients).
                        Use --nodes for the array size; with --dry-run it prints
                        the generated script instead of submitting.
  --pbs                 generate + submit a PBS/Torque qsub JOB ARRAY (batch;
                        same shape as --sbatch). --partition is the queue (-q),
                        --time is the walltime. --dry-run prints the script.
  --ntasks N            Slurm srun: number of client tasks (srun -n N)
  --nodes N             sbatch/--pbs: number of nodes = job-array size (default 1)
  --partition P         sbatch: --partition / PBS: queue (-q)
  --time T              sbatch/--pbs: walltime (e.g. 24:00:00)
  --job-name NAME       sbatch/--pbs: job name (default cado-sieve)
  --client-bin PATH     client binary path valid on the hosts
                        (default: this checkout's build .../cado-nfs-client-rs
                         or rust/target/release/cado-nfs-client-rs)
  --arch S              pass --arch S to the client
  --ssh-opts "..."      override ssh options
  --stop                kill cado-nfs-client-rs on the SSH hosts and exit
  --dry-run             print the commands without running them
  -h, --help            this help
  Anything after -- is passed verbatim to each client.
EOF
}

die() { echo "$PROG: $*" >&2; exit 1; }

while [ $# -gt 0 ]; do
    case "$1" in
        --server) SERVER="$2"; shift 2;;
        --certsha1) CERTSHA1="$2"; shift 2;;
        --hosts) HOSTS="$2"; shift 2;;
        --hostfile) HOSTFILE="$2"; shift 2;;
        --clients-per-host) CLIENTS_PER_HOST="$2"; shift 2;;
        --slurm) SLURM=1; shift;;
        --sbatch) SBATCH=1; shift;;
        --pbs) PBS=1; shift;;
        --ntasks) NTASKS="$2"; shift 2;;
        --nodes) NODES="$2"; shift 2;;
        --gpus-per-node) GPUS_PER_NODE="$2"; shift 2;;
        --partition) PARTITION="$2"; shift 2;;
        --time) TIMELIMIT="$2"; shift 2;;
        --job-name) JOBNAME="$2"; shift 2;;
        --client-bin) CLIENT_BIN="$2"; shift 2;;
        --arch) ARCH="$2"; shift 2;;
        --ssh-opts) SSH_OPTS="$2"; shift 2;;
        --stop) STOP=1; shift;;
        --dry-run) DRYRUN=1; shift;;
        -h|--help) usage; exit 0;;
        --) shift; EXTRA=("$@"); break;;
        *) die "unknown argument $1 (try --help)";;
    esac
done

# Resolve the host list (from --hosts and/or --hostfile).
declare -a HOSTLIST=()
if [ -n "$HOSTS" ]; then
    IFS=',' read -r -a _h <<< "$HOSTS"
    HOSTLIST+=("${_h[@]}")
fi
if [ -n "$HOSTFILE" ]; then
    [ -r "$HOSTFILE" ] || die "cannot read hostfile $HOSTFILE"
    while IFS= read -r line; do
        line="${line%%#*}"; line="$(echo "$line" | tr -d '[:space:]')"
        [ -n "$line" ] && HOSTLIST+=("$line")
    done < "$HOSTFILE"
fi

run() {  # echo + (maybe) execute
    echo "+ $*"
    [ "$DRYRUN" -eq 1 ] || eval "$@"
}

# --stop: kill the client on each SSH host.
if [ "$STOP" -eq 1 ]; then
    [ ${#HOSTLIST[@]} -gt 0 ] || die "--stop needs --hosts/--hostfile"
    for h in "${HOSTLIST[@]}"; do
        run "ssh $SSH_OPTS $h 'pkill -f cado-nfs-client-rs || true'"
    done
    echo "$PROG: stop signal sent to ${#HOSTLIST[@]} host(s)."
    exit 0
fi

[ -n "$SERVER" ] || die "--server URL is required (try --help)"

# Default client binary: prefer this checkout's hostname build dir, else the
# Rust workspace release dir.
if [ -z "$CLIENT_BIN" ]; then
    here=$(cd "$(dirname "$0")/.." && pwd)
    for cand in \
        "$here/build/$(hostname)/rust/cado-nfs-client-rs" \
        "$here/rust/target/release/cado-nfs-client-rs"; do
        [ -x "$cand" ] && CLIENT_BIN="$cand" && break
    done
    [ -n "$CLIENT_BIN" ] || die "could not find cado-nfs-client-rs; pass --client-bin"
fi

# Assemble the common client argument string.
cargs="--server $SERVER"
[ -n "$CERTSHA1" ] && cargs="$cargs --certsha1 $CERTSHA1"
[ -n "$ARCH" ] && cargs="$cargs --arch $ARCH"
if [ ${#EXTRA[@]} -gt 0 ]; then
    cargs="$cargs ${EXTRA[*]}"
fi

echo "$PROG: server=$SERVER  client=$CLIENT_BIN"

# How many clients per node, and how to pin each one. With --gpus-per-node N we
# start one client per GPU, each pinned via CUDA_VISIBLE_DEVICES so a GPU-prefactor
# / GPU-cofactor client gets a distinct device (one rank per GPU). Otherwise
# --clients-per-host plain clients.
per_node=$CLIENTS_PER_HOST
[ "$GPUS_PER_NODE" -gt 0 ] && per_node=$GPUS_PER_NODE

# emit the shell to start the j-th (0-based) client on a node, with GPU pinning
# when requested. $1 = a clientid prefix unique to the node/task.
client_cmd() {  # client_cmd <cidprefix> <j>
    local j="$2" pin="" cid="$1.$2"
    if [ "$GPUS_PER_NODE" -gt 0 ]; then pin="CUDA_VISIBLE_DEVICES=$j "; cid="$1.gpu$j"; fi
    echo "${pin}nohup $CLIENT_BIN $cargs --clientid $cid >/tmp/cado-client-$cid.log 2>&1 &"
}

# emit the per-task client line for a job array; $1 = the scheduler's array-index
# env var name (SLURM_ARRAY_TASK_ID or PBS_ARRAY_INDEX). $j and the index var are
# expanded at job runtime, GPU-pinned via CUDA_VISIBLE_DEVICES when requested.
array_body() {  # array_body <index-var-name>
    local idx="$1"
    if [ "$GPUS_PER_NODE" -gt 0 ]; then
        echo '    CUDA_VISIBLE_DEVICES=$j nohup '"$CLIENT_BIN $cargs"' --clientid n${'"$idx"'}.gpu$j >/tmp/cado-client-n${'"$idx"'}.gpu$j.log 2>&1 &'
    else
        echo '    nohup '"$CLIENT_BIN $cargs"' --clientid n${'"$idx"'}.$j >/tmp/cado-client-n${'"$idx"'}.$j.log 2>&1 &'
    fi
}

# ---- Slurm sbatch JOB ARRAY (batch; one array task per node) ----
if [ "$SBATCH" -eq 1 ]; then
    [ "$NODES" -gt 0 ] || NODES=1
    arr_hi=$((NODES - 1))
    body=$(array_body SLURM_ARRAY_TASK_ID)
    script=$(cat <<SB
#!/usr/bin/env bash
#SBATCH --job-name=$JOBNAME
#SBATCH --array=0-$arr_hi
#SBATCH --ntasks=1
${PARTITION:+#SBATCH --partition=$PARTITION}
${TIMELIMIT:+#SBATCH --time=$TIMELIMIT}
${GPUS_PER_NODE:+#SBATCH --gres=gpu:$GPUS_PER_NODE}
# one array task per node; each starts $per_node client(s), GPU-pinned if requested.
set -e
for j in \$(seq 0 $((per_node - 1))); do
$body
done
wait
SB
)
    if [ "$DRYRUN" -eq 1 ]; then
        echo "--- generated sbatch script (--dry-run; not submitted) ---"
        printf '%s\n' "$script"
        echo "--- (submit with: sbatch <this script>) ---"
        exit 0
    fi
    command -v sbatch >/dev/null || die "--sbatch given but sbatch not found"
    tmp=$(mktemp /tmp/cado-sbatch.XXXXXX.sh)
    printf '%s\n' "$script" > "$tmp"
    run "sbatch $tmp"
    echo "$PROG: submitted job array ($NODES node(s) x $per_node client(s)); script $tmp"
    exit 0
fi

# ---- PBS/Torque qsub JOB ARRAY (batch; one array task per node) ----
if [ "$PBS" -eq 1 ]; then
    [ "$NODES" -gt 0 ] || NODES=1
    arr_hi=$((NODES - 1))
    body=$(array_body PBS_ARRAY_INDEX)
    # PBS chunk request: one chunk per node; ncpus is advisory here (the clients
    # are multi-threaded). --partition doubles as the PBS queue (-q).
    sel="select=1:ncpus=8"
    [ "$GPUS_PER_NODE" -gt 0 ] && sel="select=1:ncpus=8:ngpus=$GPUS_PER_NODE"
    script=$(cat <<PB
#!/usr/bin/env bash
#PBS -N $JOBNAME
#PBS -J 0-$arr_hi
#PBS -l $sel
${TIMELIMIT:+#PBS -l walltime=$TIMELIMIT}
${PARTITION:+#PBS -q $PARTITION}
# one array task per node; each starts $per_node client(s), GPU-pinned if requested.
cd "\${PBS_O_WORKDIR:-.}"
set -e
for j in \$(seq 0 $((per_node - 1))); do
$body
done
wait
PB
)
    if [ "$DRYRUN" -eq 1 ]; then
        echo "--- generated PBS script (--dry-run; not submitted) ---"
        printf '%s\n' "$script"
        echo "--- (submit with: qsub <this script>) ---"
        exit 0
    fi
    command -v qsub >/dev/null || die "--pbs given but qsub not found"
    tmp=$(mktemp /tmp/cado-pbs.XXXXXX.sh)
    printf '%s\n' "$script" > "$tmp"
    run "qsub $tmp"
    echo "$PROG: submitted PBS job array ($NODES node(s) x $per_node client(s)); script $tmp"
    exit 0
fi

# ---- Slurm srun (interactive) ----
if [ "$SLURM" -eq 1 ]; then
    [ "$DRYRUN" -eq 1 ] || command -v srun >/dev/null || die "--slurm given but srun not found"
    [ "$NTASKS" -gt 0 ] || die "--slurm needs --ntasks N"
    gres=""
    [ "$GPUS_PER_NODE" -gt 0 ] && gres="--gpus-per-task=1 "
    # one client per task; srun fans across the allocation's nodes. With GPUs,
    # --gpus-per-task=1 gives each task its own device (the client sees it as dev 0).
    run "srun ${gres}--ntasks=$NTASKS --kill-on-bad-exit=0 $CLIENT_BIN $cargs"
    exit 0
fi

# ---- SSH fan-out ----
[ ${#HOSTLIST[@]} -gt 0 ] || die "no hosts (use --hosts/--hostfile, --slurm or --sbatch)"
total=0
for h in "${HOSTLIST[@]}"; do
    for j in $(seq 0 $((per_node - 1))); do
        # nohup + background so the SSH session returns immediately; clientid is
        # unique per (host, slot|gpu) so the server distinguishes them.
        remote="$(client_cmd "$h" "$j")"
        run "ssh $SSH_OPTS $h '$remote'"
        total=$((total + 1))
    done
done
echo "$PROG: launched $total client(s) across ${#HOSTLIST[@]} host(s)$([ "$GPUS_PER_NODE" -gt 0 ] && echo ' (one per GPU)')."
echo "  logs: /tmp/cado-client-<id>.log on each host"
echo "  stop: $PROG --hosts <same> --stop"
