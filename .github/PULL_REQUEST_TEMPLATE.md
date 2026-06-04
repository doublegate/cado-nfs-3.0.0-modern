<!--
Reminder: this fork only modernizes upstream CADO-NFS 2.3.0 for current
toolchains. Algorithm/feature changes should go upstream:
https://gitlab.inria.fr/cado-nfs/cado-nfs
-->

## Summary

<!-- What does this change and why? Which toolchain/OS/version does it target? -->

## Type of change

- [ ] Toolchain / compiler / OS compatibility fix
- [ ] Python stdlib deprecation/removal fix
- [ ] Build system / packaging
- [ ] CI
- [ ] Documentation

## Verification

- [ ] `make` builds cleanly
- [ ] `make check ARGS="-R '<subsystem>'"` passes for the affected area
- [ ] `./cado-nfs.py <N> -t 4` still factors end-to-end (if runtime was touched)

Environment tested on (OS, compiler, CMake/Python/GMP/hwloc/OpenSSL versions):

```
```

## Checklist

- [ ] Minimal, surgical change; no bulk reformatting of upstream source
- [ ] Compatibility shims are version-guarded where practical
- [ ] `CHANGELOG.md` updated
