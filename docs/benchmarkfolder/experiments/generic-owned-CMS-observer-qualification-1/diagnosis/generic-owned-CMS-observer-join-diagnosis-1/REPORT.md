# Closed RUN1 binding diagnosis

The first journal rejection is a parser overconstraint. Both retained frames are native DP: CTMODE1/2, width/refwidth64. For each, all16 reference complex-word pairs, all16 current pairs, masks1111 and current_ok1 exactly match its consumer. Only frame.genuine1 versus consumer.genuine0 differs.

The source deliberately passes literal.TRUE. through the DP TIR wrapper to the observed builder. That legacy argument name does not describe current QP ownership for this typed DP path. The consumer records PM_GENUINE_QP, which is correctly false outside its mode4/5 capture gates. QP instead forwards the actual current-QP flag unchanged. Thus the universal equality at frozen analyzer line219 compares distinct DP semantics. No checked reference/current/mask/OK wire discrepancy was found.

Raw lines1709/1778 are the consumers and1847/2009 the corresponding frames. diagnosis.json binds the exact retained150580-byte raw SHAfeed964f..., source index2333cb..., frozen analyzer5f2b..., individual source hashes, source coordinates and every compared field. The36 producer events are DP only, at H1/4/13; consumer H17 is separate. This does not establish a QP path.

unchanged-QP-counterexample.json records the still-forbidden synthetic width128 frame.genuine1/consumer.genuine0 case. It is a source-contract counterexample only, not an analyzer execution or a corrected parser. Any later successor must distinguish the DP sentinel while retaining actual QP coupling and all other guards.

The original RUN remains CLOSEDFAILED. No retry, compiler, loader, scientific/native code or frozen analyzer was invoked. No corrected source was written. This diagnosis explains only the encountered failure and does not claim later journal gates, numerical accuracy, caller lifetime or science pass.
