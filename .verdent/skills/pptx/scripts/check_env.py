#!/usr/bin/env python3
"""Check pptx skill environment dependencies and report status."""

import os
import shutil
import subprocess
import sys
import platform

COMMON_BIN_PATHS = [
    "/opt/homebrew/bin",
    "/opt/homebrew/sbin",
    "/usr/local/bin",
    "/usr/local/sbin",
    os.path.expanduser("~/.nvm/current/bin"),
    os.path.expanduser("~/.local/bin"),
]


def _enrich_path():
    """Add common install locations to PATH so we can find binaries
    even when the shell session doesn't source .zshrc/.bashrc."""
    current = os.environ.get("PATH", "")
    dirs = current.split(os.pathsep)
    added = []
    for p in COMMON_BIN_PATHS:
        if os.path.isdir(p) and p not in dirs:
            dirs.insert(0, p)
            added.append(p)
    if added:
        os.environ["PATH"] = os.pathsep.join(dirs)
    return added


def _find_binary(name):
    """Find binary on PATH, and also scan common locations."""
    found = shutil.which(name)
    if found:
        return found
    for d in COMMON_BIN_PATHS:
        candidate = os.path.join(d, name)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def check(name, cmd, hint):
    binary = _find_binary(cmd[0]) if isinstance(cmd, list) else _find_binary(cmd)
    if binary:
        try:
            run_cmd = [binary] + (cmd[1:] if isinstance(cmd, list) else ["--version"])
            ver = subprocess.check_output(
                run_cmd, stderr=subprocess.STDOUT, timeout=10
            ).decode().strip().splitlines()[0]
        except Exception:
            ver = f"installed ({binary})"
        return True, ver, binary
    return False, hint, None


def check_python_pkg(pkg, import_name=None):
    import_name = import_name or pkg
    try:
        mod = __import__(import_name)
        ver = getattr(mod, "__version__", "installed")
        return True, ver
    except ImportError:
        return False, f"pip install {pkg}"


def _run_install(cmd_str):
    print(f"    Running: {cmd_str}")
    try:
        subprocess.check_call(cmd_str, shell=True, timeout=120)
        print(f"    Done.")
        return True
    except Exception as e:
        print(f"    Failed: {e}")
        return False


def main(install=False):
    added_paths = _enrich_path()

    print("=== pptx skill: Environment Check ===\n")
    if added_paths:
        print(f"  (Added to PATH: {', '.join(added_paths)})\n")

    all_ok = True
    missing_system = []
    missing_py = []
    missing_node = []

    is_mac = platform.system() == "Darwin"
    brew_path = _find_binary("brew")

    checks = [
        ("Python", ["python3", "--version"], "Install Python 3.8+"),
        ("Node.js", ["node", "--version"], "brew install node" if is_mac else "https://nodejs.org"),
        ("LibreOffice", ["soffice", "--version"], "brew install --cask libreoffice" if is_mac else "https://www.libreoffice.org"),
        ("pdftoppm (Poppler)", ["pdftoppm", "-v"], "brew install poppler" if is_mac else "apt install poppler-utils"),
    ]

    print("[System Tools]")
    for name, cmd, hint in checks:
        ok, info, binary = check(name, cmd, hint)
        symbol = "+" if ok else "-"
        print(f"  [{symbol}] {name}: {info if ok else 'NOT FOUND -> ' + hint}")
        if not ok:
            all_ok = False
            missing_system.append((name, hint))

    py_pkgs = [
        ("python-pptx", "pptx"),
        ("Pillow", "PIL"),
        ("defusedxml", "defusedxml"),
        ("lxml", "lxml"),
    ]

    print("\n[Python Packages]")
    for pkg, imp in py_pkgs:
        ok, info = check_python_pkg(pkg, imp)
        symbol = "+" if ok else "-"
        print(f"  [{symbol}] {pkg}: {info if ok else 'NOT FOUND -> ' + info}")
        if not ok:
            all_ok = False
            missing_py.append(pkg)

    node_bin = _find_binary("node")
    npm_bin = None
    if node_bin:
        node_dir = os.path.dirname(node_bin)
        candidate_npm = os.path.join(node_dir, "npm")
        if os.path.isfile(candidate_npm) and os.access(candidate_npm, os.X_OK):
            npm_bin = candidate_npm
    if not npm_bin:
        npm_bin = _find_binary("npm")
    node_pkgs = [("pptxgenjs", "pptxgenjs")]
    global_node_path = None
    if npm_bin:
        try:
            global_node_path = subprocess.check_output(
                [npm_bin, "root", "-g"], stderr=subprocess.STDOUT, timeout=10
            ).decode().strip()
        except Exception:
            pass

    print("\n[Node Packages (scratch builds)]")
    for pkg, _ in node_pkgs:
        if node_bin:
            env = os.environ.copy()
            if global_node_path:
                existing = env.get("NODE_PATH", "")
                env["NODE_PATH"] = f"{global_node_path}:{existing}" if existing else global_node_path
            try:
                subprocess.check_output(
                    [node_bin, "-e", f"require('{pkg}')"],
                    stderr=subprocess.STDOUT, timeout=10, env=env
                )
                ok, info = True, "installed"
            except Exception:
                ok, info = False, f"npm install -g {pkg}"
        else:
            ok, info = False, "requires Node.js first"
        symbol = "+" if ok else "-"
        print(f"  [{symbol}] {pkg}: {info if ok else 'NOT FOUND -> ' + info}")
        if not ok:
            all_ok = False
            missing_node.append(pkg)

    print()
    if all_ok:
        print("All dependencies satisfied.")
        return 0

    if not install:
        print("Some dependencies are missing. Install them with the commands above,")
        print("or re-run with --install to auto-install missing items.")
        return 1

    print("--- Auto-installing missing dependencies ---\n")
    if missing_py:
        _run_install(f"pip3 install {' '.join(missing_py)}")

    if missing_system and brew_path:
        for name, hint in missing_system:
            if hint.startswith("brew "):
                _run_install(f"{brew_path} {hint[5:]}")
    elif missing_system:
        print("  [!] brew not found, cannot auto-install system tools.")
        print("      Install Homebrew: https://brew.sh")

    if missing_node and npm_bin:
        for pkg in missing_node:
            _run_install(f"{npm_bin} install -g {pkg}")
    elif missing_node and _find_binary("npm"):
        npm = _find_binary("npm")
        for pkg in missing_node:
            _run_install(f"{npm} install -g {pkg}")

    print("\n--- Re-checking after install ---\n")
    return main(install=False)


if __name__ == "__main__":
    do_install = "--install" in sys.argv
    sys.exit(main(install=do_install))
