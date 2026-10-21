# Installer cancellation refinement: independent source addendum

The final installer source `e2b83749c222077e309db3ebe99ee5847b7d66db6fb29d1cfce246344de36646` and test source `0c0a2913f7a338336e379ab85732d8252c41f6ef1f0395c8b8cc56346658a11b` have no remaining blocker in this bounded source review. The prior sealed audit is unchanged.

The new journal records each alias intent before `os.replace`. During rollback, a candidate symlink still present in the private staging directory means the rename did not commit; an absent candidate means the alias was published and must be restored. This closes the interruption interval between a successful rename and the next Python instruction. The validated destination uses the same private-source disappearance observation in addition to its explicit published flag. Moving the previous destination to its backup remains recoverable through the lexical backup check.

The revised KeyboardInterrupt regression calls the real rename before raising. Its two subtests interrupt after destination publication and after the third alias publication, then compare the complete previous installation and public alias snapshots. The preexisting missing-module symlink is restored verbatim. Existing nonsymlink user files, unique temporary namespaces, cross-filesystem alias staging, best-effort complete rollback and nonfatal postcommit cleanup retain their previously reviewed behavior.

This reviewer read the final source and regression bodies without running an installer, compiler or numerical engine. The owner separately retained an engine-free 37-test green run and three original-source failure controls (two failures and one error); their validation manifest is bound in review.json. That evidence is identified rather than represented as an independent execution by this review.

The atomic-commit observation assumes the ordinary single installer invocation controls its private staging paths. It does not promise isolation from concurrent external filesystem mutation or a second interruption during rollback. No numerical source or benchmark input changed.
