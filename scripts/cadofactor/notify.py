"""
Completion / failure notifications for cado-nfs.py (v3.4.0-modern, Track E9).

A long factorization can run for hours or days; the operator should not have to
poll a terminal to learn it finished (or died). This module sends a short
notification through one or more user-chosen channels when the run reaches a
terminal state. It is dependency-free (urllib / smtplib / subprocess only),
import-safe, and a complete no-op unless ``--notify`` is given.

Channels (selected by ``--notify CHAN[,CHAN...]``; each ``kind`` or ``kind=target``):

- ``desktop``                 -- local pop-up (``notify-send`` on Linux, ``osascript``
                                 on macOS); no target.
- ``ntfy=TOPIC``              -- ntfy.sh push (server from ``notifications.ntfy_server``
                                 or the env ``CADO_NTFY_SERVER``, default https://ntfy.sh).
- ``slack=WEBHOOK_URL``       -- Slack incoming-webhook (``{"text": ...}``).
- ``discord=WEBHOOK_URL``     -- Discord webhook (``{"content": ...}``).
- ``webhook=URL``             -- generic POST of the JSON event payload.
- ``email=ADDR``              -- e-mail via SMTP; host/port/user/pass/from come from
                                 the ``[notifications]`` block or ``CADO_SMTP_*`` env
                                 (secrets stay out of the committed param snapshot).

Secrets (Slack/Discord URLs may be sensitive; SMTP credentials always are) are
read from the environment or the live (un-snapshotted) ``[notifications]`` block,
never hard-coded. A channel that fails to send logs a warning and is skipped --
a notification problem must never abort or fail a completed factorization.

>>> chans = parse_spec("desktop, ntfy=cado-runs , slack=https://h/x")
>>> [c["kind"] for c in chans]
['desktop', 'ntfy', 'slack']
>>> chans[1]["target"]
'cado-runs'
>>> title, body = format_event({"state": "done", "name": "c90",
...     "computation": "FACT", "input_digits": 90,
...     "factors": ["p1", "p2"], "elapsed": 197.9})
>>> title
'cado-nfs: c90 DONE (2 factors)'
>>> print(body)
Factorization c90 (FACT, 90 digits) finished: done.
Elapsed: 3m18s.
Factors: p1 p2
>>> t2, b2 = format_event({"state": "error", "name": "c120",
...     "computation": "FACT", "input_digits": 120, "elapsed": 5400})
>>> t2
'cado-nfs: c120 ERROR'
>>> print(b2)
Factorization c120 (FACT, 120 digits) finished: error.
Elapsed: 1h30m00s.
"""

import json
import os
import subprocess
import sys
import urllib.request


_VALID_KINDS = ("desktop", "ntfy", "slack", "discord", "webhook", "email")


def parse_spec(spec):
    """Parse a ``--notify`` spec string into a list of channel dicts
    ``{"kind": ..., "target": ...}``. Whitespace around items and the ``=`` is
    ignored; unknown kinds raise ValueError so a typo is caught early.

    >>> parse_spec("")
    []
    >>> parse_spec("desktop")
    [{'kind': 'desktop', 'target': None}]
    >>> parse_spec("webhook=https://x/y, email=me@h")
    [{'kind': 'webhook', 'target': 'https://x/y'}, {'kind': 'email', 'target': 'me@h'}]
    >>> parse_spec("bogus")
    Traceback (most recent call last):
        ...
    ValueError: unknown notify channel 'bogus' (valid: desktop, ntfy, slack, discord, webhook, email)
    """
    out = []
    for item in (spec or "").split(","):
        item = item.strip()
        if not item:
            continue
        if "=" in item:
            kind, target = item.split("=", 1)
            kind, target = kind.strip(), target.strip()
        else:
            kind, target = item, None
        if kind not in _VALID_KINDS:
            raise ValueError("unknown notify channel '%s' (valid: %s)"
                             % (kind, ", ".join(_VALID_KINDS)))
        out.append({"kind": kind, "target": target})
    return out


def _fmt_elapsed(seconds):
    """Human elapsed time.

    >>> _fmt_elapsed(42)
    '42s'
    >>> _fmt_elapsed(198)
    '3m18s'
    >>> _fmt_elapsed(5400)
    '1h30m00s'
    >>> _fmt_elapsed(None)
    'unknown'
    """
    if seconds is None:
        return "unknown"
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return "%dh%02dm%02ds" % (h, m, sec)
    if m:
        return "%dm%02ds" % (m, sec)
    return "%ds" % sec


def format_event(event):
    """Build a ``(title, body)`` pair from a status-like event dict (the keys of
    the ``cado-nfs-status/1`` snapshot, plus ``elapsed`` seconds)."""
    state = event.get("state", "done")
    name = event.get("name") or "cado-nfs"
    comp = event.get("computation") or "FACT"
    digits = event.get("input_digits")
    factors = event.get("factors") or []
    elapsed = _fmt_elapsed(event.get("elapsed"))

    if state == "done" and factors:
        title = "cado-nfs: %s DONE (%d factor%s)" % (
            name, len(factors), "" if len(factors) == 1 else "s")
    else:
        title = "cado-nfs: %s %s" % (name, state.upper())

    dig = ("%s digits" % digits) if digits else "size unknown"
    lines = ["Factorization %s (%s, %s) finished: %s." % (name, comp, dig, state),
             "Elapsed: %s." % elapsed]
    if factors:
        lines.append("Factors: " + " ".join(str(f) for f in factors))
    return title, "\n".join(lines)


class Notifier:
    """Holds the resolved channel list + the (un-snapshotted) config/secrets, and
    sends one notification per terminal event. Construct with :meth:`from_args`."""

    def __init__(self, channels, config=None, logger=None, timeout=10):
        self.channels = channels
        self.config = config or {}
        self.logger = logger
        self.timeout = timeout

    @classmethod
    def from_args(cls, spec, config=None, logger=None):
        """Build a Notifier from a ``--notify`` spec, or None if the spec is
        empty. ``config`` is the flattened ``[notifications]`` param block (for
        SMTP host/from and the ntfy server); env vars override it."""
        channels = parse_spec(spec)
        if not channels:
            return None
        return cls(channels, config=config, logger=logger)

    # -- config lookup: env wins over the param block (secrets stay uncommitted) -
    def _cfg(self, key, env, default=None):
        v = os.environ.get(env)
        if v is not None:
            return v
        return self.config.get(key, default)

    def notify(self, event):
        """Send the event through every configured channel. Never raises."""
        title, body = format_event(event)
        for ch in self.channels:
            try:
                self._dispatch(ch, title, body, event)
            except Exception as e:  # a bad channel must not break completion
                self._warn("notify channel '%s' failed: %s" % (ch["kind"], e))

    def _dispatch(self, ch, title, body, event):
        kind, target = ch["kind"], ch["target"]
        if kind == "desktop":
            self._desktop(title, body)
        elif kind == "ntfy":
            self._ntfy(target, title, body)
        elif kind == "slack":
            self._post_json(target, {"text": "*%s*\n%s" % (title, body)})
        elif kind == "discord":
            self._post_json(target, {"content": "**%s**\n%s" % (title, body)})
        elif kind == "webhook":
            payload = dict(event)
            payload["title"] = title
            self._post_json(target, payload)
        elif kind == "email":
            self._email(target, title, body)

    # -- channels -----------------------------------------------------------
    def _desktop(self, title, body):
        if sys.platform == "darwin":
            script = 'display notification %s with title %s' % (
                json.dumps(body), json.dumps(title))
            subprocess.run(["osascript", "-e", script],
                           timeout=self.timeout, check=False)
        else:
            subprocess.run(["notify-send", title, body],
                           timeout=self.timeout, check=False)

    def _ntfy(self, topic, title, body):
        if not topic:
            raise ValueError("ntfy needs a topic (ntfy=TOPIC)")
        server = (self._cfg("ntfy_server", "CADO_NTFY_SERVER",
                            "https://ntfy.sh")).rstrip("/")
        url = "%s/%s" % (server, topic)
        req = urllib.request.Request(
            url, data=body.encode("utf-8"), method="POST",
            headers={"Title": title})
        urllib.request.urlopen(req, timeout=self.timeout).read()

    def _post_json(self, url, payload):
        if not url:
            raise ValueError("this channel needs a webhook URL (kind=URL)")
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, method="POST",
            headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=self.timeout).read()

    def _email(self, to_addr, subject, body):
        if not to_addr:
            raise ValueError("email needs a recipient (email=ADDR)")
        import smtplib
        from email.message import EmailMessage
        host = self._cfg("smtp_host", "CADO_SMTP_HOST", "localhost")
        port = int(self._cfg("smtp_port", "CADO_SMTP_PORT", "25"))
        user = self._cfg("smtp_user", "CADO_SMTP_USER")
        passwd = self._cfg("smtp_pass", "CADO_SMTP_PASS")
        from_addr = self._cfg("smtp_from", "CADO_SMTP_FROM",
                              user or "cado-nfs@localhost")
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = from_addr
        msg["To"] = to_addr
        msg.set_content(body)
        use_ssl = self._cfg("smtp_ssl", "CADO_SMTP_SSL", "")
        smtp = (smtplib.SMTP_SSL if str(use_ssl).lower() in ("1", "yes", "true")
                else smtplib.SMTP)
        with smtp(host, port, timeout=self.timeout) as s:
            if user and passwd:
                try:
                    s.starttls()
                except smtplib.SMTPException:
                    pass  # server may already be TLS / not support STARTTLS
                s.login(user, passwd)
            s.send_message(msg)

    def _warn(self, msg):
        if self.logger:
            self.logger.warning(msg)
        else:
            sys.stderr.write(msg + "\n")


if __name__ == "__main__":
    import doctest
    doctest.testmod()
