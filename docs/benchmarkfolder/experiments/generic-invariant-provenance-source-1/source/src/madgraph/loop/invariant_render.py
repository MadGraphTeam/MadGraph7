"""Private exporter plumbing transformations; scientific dispatch is unchanged."""
import re


def once(text, before, after):
    if text.count(before)!=1: raise ValueError('unknown template anchor: '+before[:100])
    return text.replace(before,after)


def main_template(text):
    start=text.index('      SUBROUTINE %(proc_prefix)sSLOOPMATRIX_RETRY_IMPL(')
    after_start=text.index('      SUBROUTINE %(proc_prefix)sCLEAR_CACHES()',start)
    if after_start<0: after_start=len(text)
    before,after=text[:start],text[after_start:]
    text=text[start:after_start]
    text=once(text,'      USE ALOHA_OBJECT\n','      USE ALOHA_OBJECT\n'
        '      USE %(proc_prefix)sINVARIANT_BOUNDS, ONLY: READ_KINEMATIC_ENVIRONMENT_TOKEN\n'
        '      USE, INTRINSIC :: ISO_C_BINDING, ONLY: C_INT64_T\n')
    text=once(text,'      IMPLICIT NONE\n','      IMPLICIT NONE\n'
        '      INTEGER(C_INT64_T) GP_ENV_DP,GP_ENV_QP,GP_ENV\n'
        '      LOGICAL GP_FRESH_DP,GP_FRESH_QP,GP_FRESH\n')
    # Both initial and rotated points traverse 208. No cached event epoch.
    text=once(text,'208 CONTINUE\n','208 CONTINUE\n'
        'GP_ENV_DP=0_C_INT64_T\nGP_ENV_QP=0_C_INT64_T\n'
        'GP_FRESH_DP=.FALSE.\nGP_FRESH_QP=.FALSE.\n')
    old='CALL %(proc_prefix)sMP_COMPUTE_LOOP_COEFS(MP_P,BUFFR_BIS)'
    if text.count(old)!=2: raise ValueError('MP call sites')
    text=text.replace(old,'GP_FRESH_DP=.FALSE.\n'
        '  CALL %(proc_prefix)sMP_COMPUTE_LOOP_COEFS_PROVENANCE(MP_P,BUFFR_BIS,GP_ENV_QP,GP_FRESH_QP)')
    # Reset before the skip test (amplitude reduction) and before any producer.
    marker='  UVCT_REQ_SO_DONE=.FALSE.\n'
    text=once(text,marker,'  GP_FRESH_DP=.FALSE.\n  GP_ENV_DP=0_C_INT64_T\n'+marker)
    text=once(text,'C Helas calls for the born amplitudes and counterterms associated to given loops\n',
        'C Capture before the earliest actual W producer, also if Born calls are empty.\n'
        '  GP_ENV_DP=READ_KINEMATIC_ENVIRONMENT_TOKEN()\n'
        '  GP_FRESH_DP=.NOT.FILTER_SO\n'
        'C Helas calls for the born amplitudes and counterterms associated to given loops\n')
    # Do not split captures into batch conversion (which prepends MP_ to CALL).
    text=text.replace('%(loop_CT_calls)s',
        'IF(CTMODE.GE.4) THEN\n GP_ENV=GP_ENV_QP\n GP_FRESH=GP_FRESH_QP\n'
        'ELSE\n GP_ENV=GP_ENV_DP\n GP_FRESH=GP_FRESH_DP\nENDIF\n%(loop_CT_calls)s')
    # Source-bound reuse policy: fresh state may serve all SO batches in this
    # pass, but never a later GOTO300 / MP_DONE / coefficient reuse.
    if text.count('5000 CONTINUE\n  CTCALL_REQ_SO_DONE=.TRUE.\n')!=2: raise ValueError('SO exit count')
    text=text.replace('5000 CONTINUE\n  CTCALL_REQ_SO_DONE=.TRUE.\n',
        '5000 CONTINUE\n  GP_FRESH_DP=.FALSE.\n  GP_FRESH_QP=.FALSE.\n  CTCALL_REQ_SO_DONE=.TRUE.\n',1)
    marker='5001 CONTINUE\nENDDO\n## }\n'
    text=once(text,marker,marker+'GP_FRESH_DP=.FALSE.\nGP_FRESH_QP=.FALSE.\n')
    return before+text+after


def mp_template(text):
    text=once(text,'MP_COMPUTE_LOOP_COEFS(PS,ANSDP)',
        'MP_COMPUTE_LOOP_COEFS_PROVENANCE(PS,ANSDP,GP_ENV_OUT,GP_FRESH_OUT)')
    text=once(text,'      USE ALOHA_OBJECT\n','      USE ALOHA_OBJECT\n'
        '      USE %(proc_prefix)sINVARIANT_BOUNDS, ONLY: READ_KINEMATIC_ENVIRONMENT_TOKEN\n'
        '      USE, INTRINSIC :: ISO_C_BINDING, ONLY: C_INT64_T\n')
    text=once(text,'      IMPLICIT NONE\n','      IMPLICIT NONE\n'
        '      INTEGER(C_INT64_T) GP_ENV,GP_ENV_OUT\n'
        '      LOGICAL GP_FRESH,GP_FRESH_OUT\n')
    text=once(text,'COMPUTE_INTEGRAND_IN_QP =',
        'GP_ENV_OUT=0_C_INT64_T\nGP_FRESH_OUT=.FALSE.\n'
        'COMPUTE_INTEGRAND_IN_QP =')
    text=once(text,'  IF (COMPUTE_INTEGRAND_IN_QP) THEN  \n',
        '  GP_ENV=0_C_INT64_T\n  GP_FRESH=.FALSE.\n'
        '  GP_ENV_OUT=0_C_INT64_T\n  GP_FRESH_OUT=.FALSE.\n'
        '  IF (COMPUTE_INTEGRAND_IN_QP) THEN  \n')
    marker='  IF (COMPUTE_INTEGRAND_IN_QP) THEN\n    %(mp_born_ct_helas_calls)s'
    text=once(text,marker,'  IF (COMPUTE_INTEGRAND_IN_QP) THEN\n'
        '    GP_ENV=READ_KINEMATIC_ENVIRONMENT_TOKEN()\n'
        '    GP_FRESH=.NOT.FILTER_SO\n    %(mp_born_ct_helas_calls)s')
    # Executed after complete construction, while skip branch reaches label4000
    # with false state. Widened DP never captures a QP producer token.
    marker='  4000 CONTINUE\n'
    text=once(text,marker,marker+'  GP_ENV_OUT=GP_ENV\n  GP_FRESH_OUT=GP_FRESH\n')
    text=once(text,'5000 CONTINUE\n  CTCALL_REQ_SO_DONE=.TRUE.\n',
        '5000 CONTINUE\n  GP_FRESH_OUT=.FALSE.\n  GP_FRESH=.FALSE.\n  CTCALL_REQ_SO_DONE=.TRUE.\n')
    # Preserve old entry ABI, with an automatic ignored output contract.
    public='''      SUBROUTINE %(proc_prefix)sMP_COMPUTE_LOOP_COEFS(PS,ANSDP)
      USE, INTRINSIC :: ISO_C_BINDING, ONLY: C_INT64_T
      IMPLICIT NONE
      %(real_mp_format)s PS(0:3,%(nexternal)d)
      %(real_dp_format)s ANSDP(3,0:%(nSquaredSO)d)
      INTEGER(C_INT64_T) GP_ENV
      LOGICAL GP_FRESH
      CALL %(proc_prefix)sMP_COMPUTE_LOOP_COEFS_PROVENANCE(PS,ANSDP,GP_ENV,GP_FRESH)
      END
'''
    return public+text


def batch_template(text):
    text=once(text,'(P,NHEL,H,IC)','(P,NHEL,H,IC,GP_ENV,GP_FRESH)')
    text=once(text,'      USE ALOHA_OBJECT\n','      USE ALOHA_OBJECT\n'
        '      USE, INTRINSIC :: ISO_C_BINDING, ONLY: C_INT64_T\n')
    return once(text,'      IMPLICIT NONE\n','      IMPLICIT NONE\n'
        '      INTEGER(C_INT64_T) GP_ENV\n      LOGICAL GP_FRESH\n')


def wrapper(template, r):
    """One private call-key wrapper; original public LOOP_N remains untouched.

    The dispatch seam takes the complete contract and deliberately calls the
    original helper. Scientific matrix/direct-invariant integration is separate.
    """
    head=template[:template.index('C ----------\nC BEGIN CODE\nC ----------')]
    head=once(head,'LOOPNUM)', 'LOOPNUM,GP_ROUTE,GP_ENV,GP_FRESH)')
    head=once(head,'%(nwfsargs_header)s(', '%(nwfsargs_header)s_PROVENANCE(')
    head=once(head,'      USE ALOHA_OBJECT\n','      USE ALOHA_OBJECT\n'
        '      USE %(proc_prefix)sINVARIANT_BOUNDS, ONLY: IK, ROUTE_MATCHES,\n'
        '     $ EXTERNAL_SLOTS_FOR_ROUTE,BUILD_CHORD_BOUNDS,ENVIRONMENT_UNCHANGED,\n'
        '     $ PRODUCER_KIND_OK,SOURCE_ENVIRONMENT_SUPPORTED\n'
        '      USE, INTRINSIC :: ISO_C_BINDING, ONLY: C_INT64_T\n')
    head+='''      INTEGER GP_ROUTE,GP_SLOTS(NEXTERNAL),GP_WIDTH,GP_E,GP_MU
      INTEGER GP_CHORD(0:3,NLOOPLINE),GP_CLASS(NLOOPLINE,NLOOPLINE)
      INTEGER(C_INT64_T) GP_ENV,GP_PAIR64(2)
      INTEGER(IK) GP_EXTERNAL(2,0:3,NEXTERNAL),GP_PAIR128(2)
      LOGICAL GP_FRESH,GP_OK,GP_HAVE(NLOOPLINE),GP_ENV_OK
      %(weset)s
      %(pairingset)s
      GP_WIDTH=64
      IF(CTMODE.GE.4) GP_WIDTH=128
      GP_EXTERNAL=0_IK
      CALL EXTERNAL_SLOTS_FOR_ROUTE(GP_ROUTE,CTMODE,GP_SLOTS,GP_OK)
      GP_ENV_OK=ENVIRONMENT_UNCHANGED(GP_ENV)
      GP_OK=GP_OK.AND.GP_FRESH.AND.GP_ENV_OK.AND.SOURCE_ENVIRONMENT_SUPPORTED
      GP_OK=GP_OK.AND.ROUTE_MATCHES(GP_ROUTE,NLOOPLINE,SIZE(WE),WE,PAIRING)
      GP_OK=GP_OK.AND.PRODUCER_KIND_OK(CTMODE,GP_WIDTH,GP_FRESH)
      IF(GP_OK) THEN
        DO GP_E=1,NEXTERNAL
          IF(GP_SLOTS(GP_E).LT.1.OR.GP_SLOTS(GP_E).GT.NWAVEFUNCS) GP_OK=.FALSE.
        ENDDO
      ENDIF
      IF(GP_OK) THEN
        DO GP_E=1,NEXTERNAL
          DO GP_MU=0,3
            IF(GP_WIDTH.EQ.128) THEN
              GP_PAIR128=TRANSFER(MP_W(GP_SLOTS(GP_E))%P(GP_MU),GP_PAIR128)
              GP_EXTERNAL(:,GP_MU,GP_E)=GP_PAIR128
            ELSE
              GP_PAIR64=TRANSFER(W(GP_SLOTS(GP_E))%P(GP_MU),GP_PAIR64)
              GP_EXTERNAL(:,GP_MU,GP_E)=IAND(INT(GP_PAIR64,IK),INT(Z'FFFFFFFFFFFFFFFF',IK))
            ENDIF
          ENDDO
        ENDDO
      ENDIF
      CALL BUILD_CHORD_BOUNDS(GP_ROUTE,CTMODE,GP_WIDTH,NLOOPLINE,
     $ GP_EXTERNAL,GP_OK,GP_CHORD,GP_CLASS,GP_HAVE)
'''
    args=(r['pairingargs']+r['wfsargs']+r['margs']+'RANK,SQUAREDSOINDEX,LOOPNUM').strip()
    # No mass or other scientific actual argument is changed/evaluated in QP.
    head+='      CALL %(proc_prefix)sLOOP_%(nloopline)d%(nwfsargs_header)s_PROVENANCE_DISPATCH('+args+',GP_CHORD,GP_CLASS,GP_HAVE,GP_ENV)\n      END\n'
    # The seam is explicitly typed; a separate scientific patch must replace
    # only this dispatch, retaining the generic caller and public ABI.
    dispatch='''      SUBROUTINE %(proc_prefix)sLOOP_%(nloopline)d%(nwfsargs_header)s_PROVENANCE_DISPATCH('''+args+''',GP_CHORD,GP_CLASS,GP_HAVE,GP_ENV)
      USE, INTRINSIC :: ISO_C_BINDING, ONLY: C_INT64_T
      IMPLICIT NONE
      INTEGER %(wfsargsdecl)s
      %(mass_dp_format)s %(margsdecl)s
      %(pairingdecl)s
      INTEGER RANK,SQUAREDSOINDEX,LOOPNUM
      INTEGER GP_CHORD(0:3,%(nloopline)d)
      INTEGER GP_CLASS(%(nloopline)d,%(nloopline)d)
      LOGICAL GP_HAVE(%(nloopline)d)
      INTEGER(C_INT64_T) GP_ENV
C METADATA-ONLY SEAM: no coefficient or invariant is altered here.
      CALL %(proc_prefix)sLOOP_%(nloopline)d%(nwfsargs_header)s('''+args+''')
      END
'''
    return head+dispatch
