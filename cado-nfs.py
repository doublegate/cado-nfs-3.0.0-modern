#!/usr/bin/env python3
import os
import sys
import subprocess
import re
import itertools

if sys.hexversion < 0x03060000:
    sys.exit("Python 3.6 or newer is required to run this program.")


# THIS PART MUST BE EXACTLY IDENTICAL IN cado-nfs.py and cado-nfs-client.py

# Three possible locations for this script
#
# The installed path. Then we should rely on the @-escaped paths, and
# determine the location of all our utility stuff based on that.
#
# The build directory. We rely on the presence of a magic file called
# source-location.txt, and we fetch the python source code from there
#
# The source tree. We call the ./scripts/build_environment.sh script to
# determine where the binaries are being put.

import locale

pathdict = dict()

one_pyfile_example_subpath = "scripts/cadofactor/workunit.py"


def detect_installed_tree(pathdict):
    mydir = os.path.normpath(os.path.dirname(sys.argv[0]))
    md = mydir.split(os.path.sep)
    install_tree = mydir
    bd = "@BINSUFFIX@".split(os.path.sep)
    while md and bd and md[-1] == bd[-1]:
        md.pop()
        bd.pop()
        install_tree = os.path.normpath(os.path.join(install_tree, ".."))
    if bd:
        if os.environ.get("CADO_NFS_DEBUG_PATHDETECT"):
            print("{} does not end in @BINSUFFIX@".format(mydir))
        return False
    example = os.path.join(install_tree,
                           "@LIBSUFFIX@",
                           one_pyfile_example_subpath)
    t = os.path.exists(example)
    if not t:
        if os.environ.get("CADO_NFS_DEBUG_PATHDETECT"):
            print("{} does not exist".format(example))
        return False

    # make all this relocatable, it doesn't cost us much.
    # (note though that the rpaths in the binaries are likely to still
    # contain absolute paths)
    pathdict["pylib"] = os.path.join(install_tree, "@LIBSUFFIX@/scripts")
    pathdict["data"] = os.path.join(install_tree, "@DATASUFFIX@")
    pathdict["lib"] = os.path.join(install_tree, "@LIBSUFFIX@")
    pathdict["bin"] = os.path.join(install_tree, "@BINSUFFIX@")

    if os.environ.get("CADO_NFS_DEBUG_PATHDETECT"):
        print("cado-nfs running in installed tree")
    return True


def detect_build_tree(pathdict):
    # source-location.txt is created by our build system, and can be used
    # *ONLY* when we call this script from the build directory. We don't
    # want this to perspire in any installed file, of course.
    mydir = os.path.normpath(os.path.dirname(sys.argv[0]))
    source_location_subpath = "source-location.txt"
    source_location_file = os.path.join(mydir, source_location_subpath)
    if not os.path.exists(source_location_file):
        if os.environ.get("CADO_NFS_DEBUG_PATHDETECT"):
            print("{} does not exist".format(source_location_file))
        return False

    # ok, we're in the build tree, apparently
    source_tree = open(source_location_file, "r").read().strip()

    pathdict["pylib"] = os.path.join(source_tree, "scripts")
    pathdict["data"] = os.path.join(source_tree, "parameters")
    pathdict["lib"] = mydir
    pathdict["bin"] = mydir

    if os.environ.get("CADO_NFS_DEBUG_PATHDETECT"):
        print("cado-nfs running in build tree")
    return True


def detect_source_tree(pathdict):
    mydir = os.path.normpath(os.path.dirname(sys.argv[0]))
    helper = os.path.join(mydir, "scripts/build_environment.sh")
    if not os.path.exists(helper):
        if os.environ.get("CADO_NFS_DEBUG_PATHDETECT"):
            print("{} does not exist".format(helper))
        return False
    pipe = subprocess.Popen([helper, "--show"], stdout=subprocess.PIPE)
    loc = locale.getlocale()[1]
    if not loc:
        loc = "ascii"
    output = pipe.communicate()[0].decode(loc)
    cado_bin_path = [x.split("=", 2)[1]
                     for x in output.split("\n")
                     if re.match("^build_tree", x)][0]
    cado_bin_path = re.sub("^\"(.*)\"$", "\\1", cado_bin_path)

    pathdict["pylib"] = os.path.join(mydir, "scripts")
    pathdict["data"] = os.path.join(mydir, "parameters")
    pathdict["lib"] = cado_bin_path
    pathdict["bin"] = cado_bin_path

    if os.environ.get("CADO_NFS_DEBUG_PATHDETECT"):
        print("cado-nfs running in source tree")
    return True


if detect_installed_tree(pathdict):
    pass
elif detect_build_tree(pathdict):
    pass
elif detect_source_tree(pathdict):
    pass
else:
    raise RuntimeError("We're unable to determine"
                       " the location of the cado-nfs binaries"
                       " and python files")

sys.path.append(pathdict["pylib"])

# END OF THE PART THAT MUST BE IDENTICAL IN cado-nfs.py and cado-nfs-client.py


from cadofactor import toplevel, cadologger, cadotask  # noqa: E402
from cadofactor.status import STATUS                    # noqa: E402
from cadofactor.cadocommand import shellquote          # noqa: E402
from cadofactor.cadoutils import Computation           # noqa: E402

if __name__ == '__main__':
    sys.set_int_max_str_digits(0)  # to be able to print very large integer
    # Parse command line arguments

    # Some command-line arguments are really parsed only here, while some
    # others are relevant to the whole hierarchy of cado-nfs programs.
    # The (hairy) logic which is used to form the definitive list of
    # parameters (in the cadoparams sense) from what we got here on the
    # command line is grouped in the Cado_NFS_toplevel class, down in
    # scripts/cadofactor/toplevel.py

    toplevel_params = toplevel.Cado_NFS_toplevel()
    for key, value in pathdict.items():
        toplevel_params.setpath(key, value)
    logger = toplevel_params.logger
    parameters, db = toplevel_params.get_cooked_parameters()

    # We have to remove the credential info from the database which gets
    # stored in the parameter snapshot. We must also make sure that we
    # store this info always at the root of the param tree, or we may end
    # up do bizarre things with duplicated keys if we resume from a
    # parameter snapshot file *and* we have something on the command
    # line.
    if parameters.get_or_set_default("database", None):
        parameters.replace("database", db.uri_without_credentials)
        # do this so that the parameter does not appear unused.
        parameters.get_or_set_default("database")

    # well, this *must* exist, right ?
    name = parameters.get_or_set_default("tasks.name")
    wdir = parameters.get_or_set_default("tasks.workdir")

    # Add a logger to capture the command lines of programs we run
    cmdfilename = os.path.join(wdir, name + ".cmd")
    logger.addHandler(cadologger.CmdFileHandler(cmdfilename))

    # Add a logger to write debugging information to a log file
    filelvl = getattr(cadologger, toplevel_params.args.filelog.upper())
    logfilename = os.path.join(wdir, name + ".log")
    filehandler = cadologger.FileHandler(filename=logfilename, lvl=filelvl)
    logger.addHandler(filehandler)

    cmdline = " ".join([shellquote(arg, idx == 0)
                        for idx, arg in enumerate(sys.argv)])
    cmdline = cmdline.replace(db.uri, db.uri_without_credentials)
    logger.info("Command line parameters: %s", cmdline)

    logger.debug("Root parameter dictionary:\n%s", parameters)

    # Write a snapshot of the parameters to a file
    for counter in itertools.count():
        snapshot_basename = name + ".parameters_snapshot.%d" % counter
        snapshot_filename = os.path.join(wdir, snapshot_basename)
        if not os.path.isfile(snapshot_filename):
            break
    with open(snapshot_filename, "w") as snapshot_file:
        logger.debug("Writing parameter snapshot to %s", snapshot_filename)
        snapshot_file.write(str(parameters))
        snapshot_file.write("\n")

    logger.info("If this computation gets interrupted,"
                " it can be resumed with %s %s",
                sys.argv[0], snapshot_filename)

    # Run-status reporting + terminal-event hooks (Tracks 3.1 / E9 / E10 / E11).
    # Configured *before* the GPU pre-factoring stage so the completion
    # notification, the structured event log and the run-history record all fire
    # even when the GPU stage fully factors N (it calls STATUS.finish and exits
    # without entering the NFS path below).
    _comp = parameters.myparams({"computation": ""}, "").get("computation")
    try:
        _Nval = int(parameters.get_or_set_default("N"))
        _digits = len(str(_Nval))
    except (ValueError, TypeError):
        _Nval, _digits = None, None
    try:
        _threads = int(parameters.get_or_set_default("tasks.threads"))
    except (KeyError, ValueError, TypeError):
        _threads = None

    # E9: build the notifier from --notify, drawing SMTP host/from and the ntfy
    # server from the [notifications] block (secrets via env override it).
    _notifier = None
    if getattr(toplevel_params.args, "notify", None):
        from cadofactor import notify as _notify
        _ncfg = parameters.myparams({
            "ntfy_server": "", "smtp_host": "", "smtp_port": "",
            "smtp_user": "", "smtp_pass": "", "smtp_from": "", "smtp_ssl": "",
        }, "notifications")
        _ncfg = {k: v for k, v in _ncfg.items() if v != ""}
        try:
            _notifier = _notify.Notifier.from_args(
                toplevel_params.args.notify, config=_ncfg, logger=logger)
        except ValueError as _e:
            logger.warning("--notify: %s", _e)

    STATUS.configure(json_path=toplevel_params.args.json_status,
                     progress=toplevel_params.args.progress,
                     json_log=toplevel_params.args.json_log,
                     name=name,
                     computation=str(_comp) if _comp is not None else None,
                     input_digits=_digits)
    if _notifier is not None:
        STATUS.add_finish_hook(_notifier.notify)
    # E11: record every terminal run in ~/.cado-nfs/runs.db (best-effort; a DB
    # error is swallowed by the finish-hook wrapper and never affects the run).
    from cadofactor import runs as _runs
    STATUS.add_finish_hook(
        lambda ev, _n=_Nval, _t=_threads: _runs.record(ev, n=_n, threads=_t))

    # GPU pre-NFS factoring stage (Track 2.1): strip small/medium factors on the
    # GPU before NFS. If it fully factors N, skip NFS; if a composite cofactor
    # remains, finish it with a fresh cado-nfs.py run (which picks parameters for
    # the cofactor's size). Falls through to a normal run if nothing is stripped
    # or the GPU binary is absent.
    if getattr(toplevel_params.args, "gpu_prefactor", False) and \
            parameters.myparams({"computation": ""}, "").get("computation") \
            in (Computation.FACT, "FACT", None):
        from cadofactor import gpu_prefactor as _gp
        _binary = _gp.find_binary(pathdict)
        if not _binary:
            logger.warning("--gpu-prefactor: gpu-prefactor binary not found "
                           "(build with -DENABLE_GPU=ON); running NFS normally")
        else:
            _N = int(parameters.get_or_set_default("N"))
            _primes, _cof = _gp.run(_binary, _N, toplevel_params.args.gpu_b1,
                                    toplevel_params.args.gpu_curves,
                                    toplevel_params.args.gpu_b2, logger)
            if _primes:
                logger.info("GPU pre-factoring stripped: %s",
                            " ".join(str(p) for p in _primes))
                if _cof == 1 or _gp.is_probable_prime(_cof):
                    _all = sorted(_primes + ([_cof] if _cof > 1 else []))
                    logger.info("Factored entirely by the GPU pre-stage; "
                                "skipping NFS")
                    STATUS.finish(factors=[str(x) for x in _all], state="done")
                    print(" ".join(str(x) for x in _all))
                    toplevel_params.purge_temp_files()
                    sys.exit(0)
                logger.info("Composite %d-digit cofactor remains; finishing it "
                            "with NFS", len(str(_cof)))
                _sub = [sys.executable, os.path.abspath(sys.argv[0])]
                _skip = False
                for _a in sys.argv[1:]:
                    if _skip:
                        _skip = False
                        continue
                    if _a == "--gpu-prefactor":
                        continue
                    if _a in ("--gpu-b1", "--gpu-b2", "--gpu-curves"):
                        _skip = True
                        continue
                    if _a.startswith("--gpu-") and "=" in _a:
                        continue
                    # the top-level run owns the user-facing terminal event, so
                    # don't let the cofactor sub-run notify or log again (E9/E10).
                    if _a in ("--notify", "--json-log"):
                        _skip = True
                        continue
                    if _a.startswith(("--notify=", "--json-log=")):
                        continue
                    _sub.append(str(_cof) if _a == str(_N) else _a)
                _r = subprocess.run(_sub, stdout=subprocess.PIPE, text=True)
                _subfac = _r.stdout.split() if _r.returncode == 0 else []
                _all = sorted(_primes + [int(x) for x in _subfac if x.isdigit()])
                STATUS.finish(factors=[str(x) for x in _all],
                              state="done" if _subfac else "error")
                print(" ".join(str(x) for x in _all))
                toplevel_params.purge_temp_files(nopurge=not _subfac)
                sys.exit(0 if _subfac else 1)

    # GPU polynomial-selection collision search (Track C2, EXPERIMENTAL). Setting the
    # env makes every spawned polyselect worker offload the collision search (the
    # memory-bound bulk of stage-1, which grows with N) to the device when worthwhile
    # (polyselect-gpu.cu, a -DENABLE_GPU=ON build); without that build, or below the
    # size gate, the CPU path runs -- bit-identical polynomial set either way. The
    # GPU root-finding path is a separate measured-negative opt-in
    # (CADO_GPU_POLYSELECT_ROOTS). See docs/gpu-polyselect.md.
    if getattr(toplevel_params.args, "gpu_polyselect", False):
        os.environ["CADO_GPU_POLYSELECT"] = "1"
        logger.info("--gpu-polyselect: enabling GPU collision search in polyselect "
                    "(experimental, size-gated; see docs/gpu-polyselect.md)")

    factorjob = cadotask.CompleteFactorization(db=db,
                                               parameters=parameters,
                                               path_prefix=[])

    if toplevel_params.args.verboseparam:
        logger.info("Summary of all recognized parameters\n" +
                    factorjob.parameter_help)

    try:
        factors = factorjob.run()
    except cadotask.EarlyStopException:
        STATUS.finish(state="done")
        sys.exit(0)

    if factors is None:
        STATUS.finish(state="error")
        toplevel_params.purge_temp_files(nopurge=True)
        sys.exit("Error occurred, terminating")
    else:
        STATUS.finish(factors=factors, state="done")
        toplevel_params.purge_temp_files()

    computation_param = parameters.myparams(keys=("computation",), path="")
    computation = computation_param["computation"]
    target_param = parameters.myparams({"target": ""}, "")
    target = target_param["target"]

    if computation == Computation.FACT:
        print(" ".join(factors))
    elif computation == Computation.DLP:
        logger.info("If you want to compute"
                    " one or several new target(s),"
                    " run %s %s target=<target>[,<target>,...]",
                    sys.argv[0], snapshot_filename)
        base = factors[0]
        if target != "":
            logtargets = factors[1:]
            targets = [int(ts) for ts in target.split(",")]
            logger.info("logbase = " + str(base))
            for i in range(min(len(targets), len(logtargets))):
                t = targets[i]
                logt = logtargets[i]
                logger.info("target = " + str(t))
                logger.info("log(target) = " + str(logt) + " mod ell")
            for i in range(len(logtargets), len(targets)):
                t = targets[i]
                logger.warning("NO LOG FOUND for target = " + str(t))
            print(",".join([str(x) for x in logtargets]))
    elif computation == Computation.CL:
        print(factors, end="")
    else:
        raise RuntimeError(f"unknown computation '{computation}'")
