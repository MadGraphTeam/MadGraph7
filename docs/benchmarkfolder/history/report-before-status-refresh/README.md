# Preserved intermediate report

Byte-for-byte snapshot of the PDF, its build ledger, context, and all declared document inputs before the status refresh. The retained measurements describe candidate c91cb85f3a; subsequent implementation qualification is separate. No numerical run was performed.

Verify from the MadGraph root:

```sh
source tools/tirust_env.sh
python3 docs/benchmarkfolder/history/report-before-status-refresh/build_report.py --verify-only
```
