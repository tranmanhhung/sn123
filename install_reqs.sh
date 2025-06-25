#!/usr/bin/env bash
###############################################################################
# Timelock one-shot installer (current dir only) + pip3 -r requirements.txt
###############################################################################
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

ROOT="$(pwd)"                   # directory you launched from
VENV="$ROOT/.venv"              # local virtual-env
SRC="$ROOT/timelock-src"        # Timelock source tree
PY_BIN="${PY_BIN:-python3}"     # interpreter to create the venv

echo "▶ venv : $VENV"
echo "▶ src  : $SRC"
echo "────────────────────────────────────────────────────────────"

if command -v apt-get >/dev/null 2>&1; then
  sudo rm -rf /var/lib/apt/lists/* || true
  sudo apt-get clean -qq           || true
fi

if [[ ! -d "$VENV" ]]; then
  "$PY_BIN" -m venv "$VENV"
  echo "✅ created .venv"
fi
# shellcheck disable=SC1090
source "$VENV/bin/activate"
python -m pip install -qU pip setuptools wheel

if ! command -v rustup >/dev/null 2>&1; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
fi
source "$HOME/.cargo/env"
rustup toolchain install stable          # no “-q” flag (avoids CLI error)
rustup default stable
cargo install wasm-pack --force --locked

python -m pip install -qU maturin
if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update -qq || true
  sudo apt-get install -y --no-install-recommends pkg-config libssl-dev
elif command -v yum >/dev/null 2>&1;   then sudo yum install -y pkgconfig openssl-devel
elif command -v brew >/dev/null 2>&1;  then brew install pkg-config openssl@3
fi

if [[ ! -d "$SRC/.git" ]]; then
  git clone --depth 1 https://github.com/ideal-lab5/timelock.git "$SRC"
else
  git -C "$SRC" pull --ff-only
fi

sed -i.bak 's|ark_std::rand::rng::OsRng|ark_std::rand::rngs::OsRng|g' \
  "$SRC/wasm/src/"{py,js}.rs || true

pushd "$SRC/wasm" >/dev/null
maturin build --release --features python
WHEEL="$(realpath ../target/wheels/timelock_wasm_wrapper-*.whl)"   # absolute path
popd >/dev/null

pip install -U "$WHEEL"
pip install -U "$SRC/py"

cd "$ROOT"
if [[ -f requirements.txt ]]; then
  echo "▶ Installing project requirements with system pip3…"
  pip3 install -r requirements.txt
else
  echo "ℹ No requirements.txt found – skipping pip3 install."
fi

echo
echo "�� Timelock ready in .venv"
echo "   Activate with:  source .venv/bin/activate"

