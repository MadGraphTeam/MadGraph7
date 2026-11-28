      INTEGER IPOS, JPOS
      INTEGER FKS_I_D(1), FKS_J_D(1)
      INTEGER EXTRA_CNT_D(1), ISPLITORDER_BORN_D(1),
     $  ISPLITORDER_CNT_D(1)
      INTEGER REAL_FLAVOR_INDEX_D(1)
      INTEGER BORN_FLAVOR_INDEX_D(1)
      INTEGER EXTRA_CNT_FLAVOR_INDEX_D(1)
      INTEGER VIRTUAL_FLAVOR_INDEX_D(1)
      INTEGER MAX_VIRTUAL_FLAVOR_INDEX
      PARAMETER (MAX_VIRTUAL_FLAVOR_INDEX=1)
      INTEGER N_VIRTUAL_FLAVOR_CONFIGS
      PARAMETER (N_VIRTUAL_FLAVOR_CONFIGS=0)
C     True when MadLoop keeps a helicity filter for each of the
C     N_VIRTUAL_FLAVOR_CONFIGS virtual rows, in HelFilter_<row>.dat
      LOGICAL HELFILTER_PER_VIRTUAL_FLAVOR
      PARAMETER (HELFILTER_PER_VIRTUAL_FLAVOR=.FALSE.)
      LOGICAL HAS_PHYSICAL_FKS_CLASSES
      PARAMETER (HAS_PHYSICAL_FKS_CLASSES=.FALSE.)
      INTEGER FKS_FLAVOR_CLASS_D(1)
      INTEGER FKS_TOPOLOGY_D(1)
      INTEGER BORN_FKS_MAP_D(1)
      INTEGER NBORN_FLAVOR_CONFIGS
      PARAMETER (NBORN_FLAVOR_CONFIGS=1)
      INTEGER BORN_FKS_CONFIG_D(NBORN_FLAVOR_CONFIGS)
      INTEGER FKS_J_FROM_I_D(1, NEXTERNAL, 0:NEXTERNAL)
      INTEGER PARTICLE_TYPE_D(1, NEXTERNAL), PDG_TYPE_D(1, NEXTERNAL)
      LOGICAL PARTICLE_TAG_D(1, NEXTERNAL)
      REAL*8 PARTICLE_CHARGE_D(1, NEXTERNAL)
      INTEGER BORN_PDG_TYPE_D(1, NEXTERNAL-1)
      REAL*8 BORN_PARTICLE_CHARGE_D(1, NEXTERNAL-1)
      INTEGER EXTRA_CNT_PDG_D(1, NEXTERNAL-1)
      INTEGER EXTRA_CNT_COLOR_D(1, NEXTERNAL-1)
      REAL*8 EXTRA_CNT_CHARGE_D(1, NEXTERNAL-1)
      LOGICAL SPLIT_TYPE_D(1, 2)
      LOGICAL NEED_COLOR_LINKS_D(1), NEED_CHARGE_LINKS_D(1)

      DATA FKS_I_D / 5 /
      DATA FKS_J_D / 4 /

      DATA NEED_COLOR_LINKS_D / .TRUE. /
      DATA NEED_CHARGE_LINKS_D / .TRUE. /

C     Local matrix-element flavor rows selected by each physical FKS
C      class.
C     All indices are one-based; zero means that the optional matrix
C      element
C     (extra counterterm or virtual) is absent for this class.
      DATA REAL_FLAVOR_INDEX_D / 1 /
      DATA BORN_FLAVOR_INDEX_D / 1 /
      DATA EXTRA_CNT_FLAVOR_INDEX_D / 0 /
      DATA VIRTUAL_FLAVOR_INDEX_D / 0 /
      DATA FKS_FLAVOR_CLASS_D / 0 /
      DATA FKS_TOPOLOGY_D / 1 /
      DATA BORN_FKS_MAP_D / 1 /
      DATA BORN_FKS_CONFIG_D / 1 /

      DATA (FKS_J_FROM_I_D(1, 5, JPOS), JPOS = 0, 1)  / 1, 4 /

C     Split type: the order of elements is QCD, QED
      DATA (SPLIT_TYPE_D (1, IPOS), IPOS=1, 2) / .FALSE., .FALSE. /

C     Extra cnt: if the real emission needs an extra cnt (g/a >
C      qqbar), then
C     this is the id of the extra cnt. Otherwise it should be set to 0.
C     If not 0, store also the position in the splitorders array of
C      the 
C     coupling corresponding to the splitting type
      DATA EXTRA_CNT_D / 0 /
      DATA ISPLITORDER_BORN_D / 0 /
      DATA ISPLITORDER_CNT_D / 0 /

C     
C     Particle type:
C     octet = 8, triplet = 3, singlet = 1
      DATA (PARTICLE_TYPE_D(1, IPOS), IPOS=1, NEXTERNAL) / -3, 3, 3, 
     $ -3, 8 /

C     
C     Particle type according to PDG:
C     
      DATA (PDG_TYPE_D(1, IPOS), IPOS=1, NEXTERNAL) / -3, 3, 6, -6, 
     $ -21 /

C     
C     Particle charge:
C     charge is set 0. with QCD corrections, which is irrelevant
      DATA (PARTICLE_CHARGE_D(1, IPOS), IPOS=1, NEXTERNAL) /  
     $  0.333333333333333D0,  -0.333333333333333D0,  
     $  0.666666666666667D0,  -0.666666666666667D0,  
     $  0.000000000000000D0 /

C     Physical charges of the corresponding underlying-Born row.
      DATA (BORN_PDG_TYPE_D(1, IPOS), IPOS=1, NEXTERNAL-1) / 0, 0, 0,
     $  0 /
      DATA (BORN_PARTICLE_CHARGE_D(1, IPOS), IPOS=1, NEXTERNAL-1) /  
     $  0.333333333333333D0,  -0.333333333333333D0,  
     $  0.666666666666667D0,  -0.666666666666667D0 /

C     Physical metadata of the alternate extra-counterterm Born row.
C     Rows without an extra counterterm contain zero PDGs/charges and
C     color-singlet entries.
      DATA (EXTRA_CNT_PDG_D(1, IPOS), IPOS=1, NEXTERNAL-1) / 0, 0, 0,
     $  0 /
      DATA (EXTRA_CNT_COLOR_D(1, IPOS), IPOS=1, NEXTERNAL-1) / 1, 1, 1
     $ , 1 /
      DATA (EXTRA_CNT_CHARGE_D(1, IPOS), IPOS=1, NEXTERNAL-1) / 0D0,
     $  0D0, 0D0, 0D0 /

C     
C     Tagged particles:
C     
      DATA (PARTICLE_TAG_D(1, IPOS), IPOS=1, NEXTERNAL) / .FALSE.,
     $  .FALSE., .FALSE., .FALSE., .FALSE. /

