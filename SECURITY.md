# Security Policy

## Scope

This repository is a **modernization fork** of upstream
[CADO-NFS](https://gitlab.inria.fr/cado-nfs/cado-nfs) 2.3.0. It carries
portability patches only; the cryptographic/number-theoretic code is upstream's.

- Vulnerabilities in the **core CADO-NFS code** should be reported **upstream**
  (<https://gitlab.inria.fr/cado-nfs/cado-nfs>), as fixes there benefit all
  users and downstreams.
- Issues in **this fork's modifications** — for example the TLS work-unit
  server/client changes in `scripts/cadofactor/wuserver.py` and
  `cado-nfs-client.py`, or the build configuration — may be reported here.

## Reporting

Please report suspected vulnerabilities **privately** via GitHub's
[private security advisories](https://github.com/doublegate/cado-nfs-2.3.1-modern/security/advisories/new)
rather than a public issue. Include a description, affected versions/toolchain,
and reproduction steps if possible.

## Operational note

CADO-NFS runs an internal HTTP/HTTPS work-unit server. The default binds with a
host whitelist (localhost by default) and a self-signed certificate pinned by
SHA-1 fingerprint. **Do not expose the work-unit server to untrusted networks.**
For multi-machine runs, follow the SSH/whitelist guidance in the upstream
[`README`](README).

## Supported versions

Only the latest tagged release of this fork (`2.3.1-modern`) receives fixes.
