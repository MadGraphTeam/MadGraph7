# Independent installer publication review

The final source patch at installer hash
`17cc6f8cc6117ff872cd7ea02ed953bd134606f8c7f92022f85f95cb9ec6f9ef`
has no remaining blocker within this bounded publication/rollback review.
This is a **source-only** signoff during the allocation timing quiet window:
the seven new injected-failure/retry test bodies were reviewed but have not yet
been executed by this reviewer or certified by this package. The implementation
owner retains the old/new execution evidence after the quiet window.

The old transaction ended after replacing the installation directory. Errors
publishing prefix/lib aliases could therefore report failure after discarding
the previous working installation. An error deleting the stage after successful
publication could also report a failed install despite the new working result.

The final helper includes destination and owned aliases in one recoverable
transaction. It records only aliases actually replaced, restores them in reverse
order, moves the validated new tree back to staging and restores the old tree.
It attempts the remaining restoration steps even if one fails; recovery failures
raise an actionable error naming retained build and alias-stage paths.
The previous destination and custom/dangling alias targets use lexical symlink
checks and verbatim readlink text, preserving their original meaning.

Two edges identified in this independent review were incorporated:

1. New and saved alias symlinks now live in a unique installer-owned temporary
   directory inside the common library directory. Publish **and rollback** rename
   on that filesystem, including a separately mounted or symlinked prefix/lib.
   Relative new targets are computed from the resolved common directory so the
   symlinked-directory case still points at the actual installed tree.
2. The transaction catches BaseException, restores the previous state and reraises
   the original exception, covering Ctrl-C during alias publication as well as
   ordinary filesystem errors. An injected KeyboardInterrupt test compares the
   previous tree and custom alias snapshot exactly.

Existing nonsymlink files/directories at public alias names are skipped. Unique
temporary names avoid clobbering user-owned hidden sentinel files left at old
fixed temporary-name paths. The helper cleans only its own alias-stage directory;
an existing common directory is never removed. A newly created common directory
is removed only after successful rollback cleanup and only with rmdir, which
refuses a nonempty directory.

Once all public aliases and the destination are committed, cleanup is outside
rollback. Alias-stage and build-stage cleanup are attempted independently.
Cleanup OSError produces a warning and returns the valid installed library path;
it cannot restore a backup that partial cleanup may already have removed.

The seven reviewed cases cover previous-install alias failure and retry,
first-install rollback and retry, common-directory failure and retry,
post-commit cleanup failure, recovery failure with retained backups, symlinked
common-directory behavior and interrupted publication. Their snapshots compare
regular bytes, directories and raw symlink text without following links.

This review assumes the ordinary single installer invocation controls publication;
it does not add a cross-process installer lock or claim transactional isolation
from another process modifying these paths concurrently. No numerical source,
precision, thresholds, backend policy or benchmark input was changed here.

The exact reviewed production/test/patch hashes are in `review.json`. Previous
sealed numerical/report audits remain unchanged. Actual new regression results
must be attached before treating the release-hardening tests as passed.
