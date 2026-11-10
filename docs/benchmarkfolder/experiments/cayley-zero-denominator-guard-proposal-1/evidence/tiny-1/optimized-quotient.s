_RNvCs5xnVew21iJH_22cayley_zero_guard_opt38baseline:
	.cfi_startproc
	divsd	%xmm1, %xmm0
	movq	%xmm0, %rax
	movabsq	$9223372036854775807, %rcx
	andq	%rax, %rcx
	movabsq	$9218868437227405312, %rdx
	xorl	%eax, %eax
	cmpq	%rdx, %rcx
	setl	%al
	retq
.Lfunc_end6:
	.size	_RNvCs5xnVew21iJH_22cayley_zero_guard_opt38baseline, .Lfunc_end6-_RNvCs5xnVew21iJH_22cayley_zero_guard_opt38baseline

_RNvCs5xnVew21iJH_22cayley_zero_guard_opt39candidate:
	.cfi_startproc
	xorpd	%xmm2, %xmm2
	ucomisd	%xmm2, %xmm1
	jne	.LBB7_2
	jp	.LBB7_2
	xorl	%eax, %eax
	retq
.LBB7_2:
	divsd	%xmm1, %xmm0
	movq	%xmm0, %rax
	movabsq	$9223372036854775807, %rcx
	andq	%rax, %rcx
	movabsq	$9218868437227405312, %rdx
	xorl	%eax, %eax
	cmpq	%rdx, %rcx
	setl	%al
	retq
.Lfunc_end7:
	.size	_RNvCs5xnVew21iJH_22cayley_zero_guard_opt39candidate, .Lfunc_end7-_RNvCs5xnVew21iJH_22cayley_zero_guard_opt39candidate

