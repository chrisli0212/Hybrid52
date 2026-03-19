"""Format conversion utilities using LibreOffice.

Handles .doc -> .docx conversion and .docx -> .pdf export,
with automatic detection and workaround for sandboxed environments
where AF_UNIX sockets are blocked.
"""

import os
import socket
import subprocess
import tempfile
from pathlib import Path


def get_office_env() -> dict:
    env = os.environ.copy()
    env["SAL_USE_VCLPLUGIN"] = "svp"

    if _unix_sockets_blocked():
        shim = _build_socket_shim()
        env["LD_PRELOAD"] = str(shim)

    return env


def invoke_office(cli_args: list[str], **kwargs) -> subprocess.CompletedProcess:
    env = get_office_env()
    return subprocess.run(["soffice"] + cli_args, env=env, **kwargs)


def transform_format(source: str, target_format: str = "docx") -> Path:
    src = Path(source)
    if not src.exists():
        raise FileNotFoundError(f"{source} not found")

    out_dir = src.parent
    result = invoke_office(
        ["--headless", "--convert-to", target_format,
         "--outdir", str(out_dir), str(src)],
        capture_output=True,
        timeout=60,
    )

    expected = out_dir / f"{src.stem}.{target_format}"
    if expected.exists():
        print(f"Converted {source} -> {expected}")
        return expected

    raise RuntimeError(
        f"Conversion failed: {result.stderr.decode() if result.stderr else 'unknown error'}"
    )


_SHIM_BINARY = Path(tempfile.gettempdir()) / "office_socket_bridge.so"


def _unix_sockets_blocked() -> bool:
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.close()
        return False
    except OSError:
        return True


def _build_socket_shim() -> Path:
    if _SHIM_BINARY.exists():
        return _SHIM_BINARY

    shim_src = Path(tempfile.gettempdir()) / "office_socket_bridge.c"
    shim_src.write_text(_BRIDGE_SOURCE_CODE)
    subprocess.run(
        ["gcc", "-shared", "-fPIC", "-o", str(_SHIM_BINARY), str(shim_src), "-ldl"],
        check=True,
        capture_output=True,
    )
    shim_src.unlink()
    return _SHIM_BINARY


_BRIDGE_SOURCE_CODE = r"""
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <unistd.h>

static int (*orig_socket)(int, int, int);
static int (*orig_socketpair)(int, int, int, int[2]);
static int (*orig_listen)(int, int);
static int (*orig_accept)(int, struct sockaddr *, socklen_t *);
static int (*orig_close)(int);
static int (*orig_read)(int, void *, size_t);

static int bridged[1024];
static int paired[1024];
static int signal_rd[1024];
static int signal_wr[1024];
static int server_fd = -1;

__attribute__((constructor))
static void setup(void) {
    orig_socket     = dlsym(RTLD_NEXT, "socket");
    orig_socketpair = dlsym(RTLD_NEXT, "socketpair");
    orig_listen     = dlsym(RTLD_NEXT, "listen");
    orig_accept     = dlsym(RTLD_NEXT, "accept");
    orig_close      = dlsym(RTLD_NEXT, "close");
    orig_read       = dlsym(RTLD_NEXT, "read");
    for (int i = 0; i < 1024; i++) {
        paired[i] = -1;
        signal_rd[i] = -1;
        signal_wr[i] = -1;
    }
}

int socket(int domain, int type, int protocol) {
    if (domain == AF_UNIX) {
        int fd = orig_socket(domain, type, protocol);
        if (fd >= 0) return fd;
        int sv[2];
        if (orig_socketpair(domain, type, protocol, sv) == 0) {
            if (sv[0] >= 0 && sv[0] < 1024) {
                bridged[sv[0]] = 1;
                paired[sv[0]] = sv[1];
                int pp[2];
                if (pipe(pp) == 0) {
                    signal_rd[sv[0]] = pp[0];
                    signal_wr[sv[0]] = pp[1];
                }
            }
            return sv[0];
        }
        errno = EPERM;
        return -1;
    }
    return orig_socket(domain, type, protocol);
}

int listen(int sockfd, int backlog) {
    if (sockfd >= 0 && sockfd < 1024 && bridged[sockfd]) {
        server_fd = sockfd;
        return 0;
    }
    return orig_listen(sockfd, backlog);
}

int accept(int sockfd, struct sockaddr *addr, socklen_t *addrlen) {
    if (sockfd >= 0 && sockfd < 1024 && bridged[sockfd]) {
        if (signal_rd[sockfd] >= 0) {
            char b;
            orig_read(signal_rd[sockfd], &b, 1);
        }
        errno = ECONNABORTED;
        return -1;
    }
    return orig_accept(sockfd, addr, addrlen);
}

int close(int fd) {
    if (fd >= 0 && fd < 1024 && bridged[fd]) {
        int was_server = (fd == server_fd);
        bridged[fd] = 0;
        if (signal_wr[fd] >= 0) {
            char c = 0;
            write(signal_wr[fd], &c, 1);
            orig_close(signal_wr[fd]);
            signal_wr[fd] = -1;
        }
        if (signal_rd[fd] >= 0) { orig_close(signal_rd[fd]); signal_rd[fd] = -1; }
        if (paired[fd] >= 0) { orig_close(paired[fd]); paired[fd] = -1; }
        if (was_server) _exit(0);
    }
    return orig_close(fd);
}
"""


if __name__ == "__main__":
    import sys
    result = invoke_office(sys.argv[1:])
    sys.exit(result.returncode)
