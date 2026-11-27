      SUBROUTINE ORIGAMP%(proc_id)s_%(chunk_id)s(P,NHEL,%(chunk_ic_arg)sIVEC,FLAVOR,W,AMP%(amp_chunk_mask_arg)s)
C
%(process_lines)s
C
C     One slice of the HELAS call sequence of MATRIX%(proc_id)s,
C     in a file of its own: as one basic block inside one
C     routine the sequence is what makes a high-multiplicity
C     matrix element uncompilable, and split up it can also
C     carry its own optimisation flag (AMP_FLAG). The slices run
C     in order and share W and AMP by reference: a wavefunction
C     slot is reused many times over the sequence, so a slot
C     number does not identify a wavefunction and the whole
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
%(chunk_ic_decl)s      INTEGER IVEC
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
C     The fake widths are recomputed here rather than read out
C     of the matrix element, which keeps them SAVEd locals:
C     sharing them would mean turning those into a common block,
C     i.e. editing every matrix element whether it is chunked or
C     not. A handful of operations, done once.
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
