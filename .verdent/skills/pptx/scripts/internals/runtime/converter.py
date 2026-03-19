"""LibreOffice process launcher with sandbox-aware socket shim.

In sandboxed environments where AF_UNIX sockets are blocked,
this module builds and injects a small C shim via LD_PRELOAD
that intercepts socket() calls so LibreOffice can start headless.
"""
from __future__ import annotations

import os
import socket
import subprocess
import tempfile
from pathlib import Path


def lo_env() -> dict:
    env = os.environ.copy()
    env["SAL_USE_VCLPLUGIN"] = "svp"

    if _unix_socket_unavailable():
        shim = _ensure_shim()
        env["LD_PRELOAD"] = str(shim)

    return env


def run_lo(args: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(["soffice"] + args, env=lo_env(), **kwargs)


_SHIM_LIB = Path(tempfile.gettempdir()) / "verdent_sockshim.so"


def _unix_socket_unavailable() -> bool:
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.close()
        return False
    except OSError:
        return True


def _ensure_shim() -> Path:
    if _SHIM_LIB.exists():
        return _SHIM_LIB

    src = Path(tempfile.gettempdir()) / "verdent_sockshim.c"
    src.write_text(_SHIM_SRC)
    subprocess.run(
        ["gcc", "-shared", "-fPIC", "-o", str(_SHIM_LIB), str(src), "-ldl"],
        check=True, capture_output=True,
    )
    src.unlink()
    return _SHIM_LIB


_SHIM_SRC = r"""
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <unistd.h>

static int (*real_socket)(int, int, int);
static int (*real_socketpair)(int, int, int, int[2]);
static int (*real_listen)(int, int);
static int (*real_accept)(int, struct sockaddr *, socklen_t *);
static int (*real_close)(int);
static int (*real_read)(int, void *, size_t);

static int patched[1024];
static int partner[1024];
static int sig_rd[1024];
static int sig_wr[1024];
static int listener = -1;

__attribute__((constructor))
static void init_shim(void) {
    real_socket     = dlsym(RTLD_NEXT, "socket");
    real_socketpair = dlsym(RTLD_NEXT, "socketpair");
    real_listen     = dlsym(RTLD_NEXT, "listen");
    real_accept     = dlsym(RTLD_NEXT, "accept");
    real_close      = dlsym(RTLD_NEXT, "close");
    real_read       = dlsym(RTLD_NEXT, "read");
    for (int i = 0; i < 1024; i++) {
        partner[i] = -1;
        sig_rd[i] = -1;
        sig_wr[i] = -1;
    }
}

int socket(int domain, int type, int protocol) {
    if (domain == AF_UNIX) {
        int fd = real_socket(domain, type, protocol);
        if (fd >= 0) return fd;
        int sv[2];
        if (real_socketpair(domain, type, protocol, sv) == 0) {
            if (sv[0] >= 0 && sv[0] < 1024) {
                patched[sv[0]] = 1;
                partner[sv[0]] = sv[1];
                int p[2];
                if (pipe(p) == 0) {
                    sig_rd[sv[0]] = p[0];
                    sig_wr[sv[0]] = p[1];
                }
            }
            return sv[0];
        }
        errno = EPERM;
        return -1;
    }
    return real_socket(domain, type, protocol);
}

int listen(int fd, int backlog) {
    if (fd >= 0 && fd < 1024 && patched[fd]) {
        listener = fd;
        return 0;
    }
    return real_listen(fd, backlog);
}

int accept(int fd, struct sockaddr *addr, socklen_t *len) {
    if (fd >= 0 && fd < 1024 && patched[fd]) {
        if (sig_rd[fd] >= 0) {
            char b;
            real_read(sig_rd[fd], &b, 1);
        }
        errno = ECONNABORTED;
        return -1;
    }
    return real_accept(fd, addr, len);
}

int close(int fd) {
    if (fd >= 0 && fd < 1024 && patched[fd]) {
        int was_listener = (fd == listener);
        patched[fd] = 0;
        if (sig_wr[fd] >= 0) {
            char c = 0;
            write(sig_wr[fd], &c, 1);
            real_close(sig_wr[fd]);
            sig_wr[fd] = -1;
        }
        if (sig_rd[fd] >= 0) { real_close(sig_rd[fd]); sig_rd[fd] = -1; }
        if (partner[fd] >= 0) { real_close(partner[fd]); partner[fd] = -1; }
        if (was_listener) _exit(0);
    }
    return real_close(fd);
}
"""
