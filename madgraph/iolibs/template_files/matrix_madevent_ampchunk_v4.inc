      SUBROUTINE ORIGAMP%(proc_id)s_%(chunk_id)s(P,NHEL,IC,IVEC,FLAVOR,W,AMP%(amp_chunk_mask_arg)s)
C
%(process_lines)s
C
C     One slice of the HELAS call sequence of MATRIX%(proc_id)s, in a file
C     of its own: as one basic block inside one routine the sequence is
C     what makes a high-multiplicity matrix element uncompilable, and
C     split up it can also carry its own optimisation flag (AMP_FLAG).
C     The slices run in order and share W and AMP by reference: a
C     wavefunction slot is reused many times over the sequence, so a
C     slot number does not identify a wavefunction and the whole
C     array has to be threaded through.
C
      use aloha_object
      use model_object
      IMPLICIT NONE
C
C CONSTANTS
C
      INTEGER    NGRAPHS
      PARAMETER (NGRAPHS=%(ngraphs)d)
      INTEGER    NWAVEFUNCS
      PARAMETER (NWAVEFUNCS=%(nwavefuncs)d)
      REAL*8     ZERO
      PARAMETER (ZERO=0D0)
      COMPLEX*16 IMAG1
      PARAMETER (IMAG1=(0D0,1D0))
      include 'nexternal.inc'
C
C ARGUMENTS
C
      REAL*8 P(0:3,NEXTERNAL)
      INTEGER NHEL(NEXTERNAL)
      INTEGER IC(NEXTERNAL)
      INTEGER IVEC
      INTEGER FLAVOR(NEXTERNAL)
      type(aloha) W(NWAVEFUNCS)
      COMPLEX*16 AMP(NGRAPHS)
%(amp_chunk_mask_decl)s
C
C LOCAL VARIABLES
C
C   Needed for v4 models
      COMPLEX*16 DUM0,DUM1
      DATA DUM0, DUM1/(0d0, 0d0), (1d0, 0d0)/
C     The fake widths are recomputed here rather than read out of the
C     matrix element, which keeps them SAVEd locals: sharing them would
C     mean turning those into a common block, i.e. editing every matrix
C     element whether it is chunked or not. A handful of operations,
C     done once.
	%(fake_width_declaration)s
	logical first
	data first /.true./
	save first
C
C GLOBAL VARIABLES
C
      include '../../Source/vector.inc' ! defines VECSIZE_MEMMAX
      include 'coupl.inc' ! needs VECSIZE_MEMMAX (defined in vector.inc)
      double precision bwcutoff
      common/to_bwcutoff/ bwcutoff
      double precision small_width_treatment
      common/narrow_width/small_width_treatment
C ----------
C BEGIN CODE
C ----------
	if (first) then
		first=.false.
		%(fake_width_definitions)s
	endif
C     HELAS CALLS BEGIN
%(helas_calls)s
C     HELAS CALLS END
      END
