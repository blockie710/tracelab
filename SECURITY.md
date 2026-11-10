# Security policy

## Threat model

`tracelab` is an offline analysis library. It opens files a user points it at and
generates HTML. That gives three realistic attack surfaces:

1. **Model bundles and trace files are untrusted input.** They are parsed as
   JSON/CSV only. There is deliberately no `pickle`, `eval`, `exec`, or dynamic
   import anywhere in the load path — a model file that can execute code is a
   supply-chain hole, and model files get shared.
2. **Generated HTML.** Trace ids, feature names and labels are escaped before
   they reach the report. If you add a field to the report, escape it; do not
   concatenate raw values into markup.
3. **Resource exhaustion.** A hostile trace file can be arbitrarily large.
   Callers scoring untrusted uploads should cap file size and trace count
   themselves; the library does not enforce a limit because the right limit
   depends on deployment.

There is no network access, no subprocess execution, and no privileged operation
in the library.

## Supported versions

The latest minor release receives fixes. Pre-1.0, only the current release.

## Reporting a vulnerability

Open a [private security advisory](https://github.com/blockie710/tracelab/security/advisories/new)
rather than a public issue. Include a minimal reproducer — a malformed trace or
model file is usually enough.

Expect an acknowledgement within a few days. If the issue is a parsing or
escaping flaw, the fix ships with a regression test and a CHANGELOG entry that
names the failure mode.

## What is out of scope

- Reports of "the detector can be fooled by a sufficiently good imitation". That
  is a documented limitation, not a vulnerability; see the README section on
  what this is not.
- Requests to add client-side collection or kernel components. Out of scope by
  design.
