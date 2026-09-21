      INTEGER IPOS, JPOS
      INTEGER FKS_I_D(%(nconfs)d), FKS_J_D(%(nconfs)d)
      INTEGER EXTRA_CNT_D(%(nconfs)d), ISPLITORDER_BORN_D(%(nconfs)d), ISPLITORDER_CNT_D(%(nconfs)d)
      INTEGER REAL_FLAVOR_INDEX_D(%(nconfs)d)
      INTEGER BORN_FLAVOR_INDEX_D(%(nconfs)d)
      INTEGER EXTRA_CNT_FLAVOR_INDEX_D(%(nconfs)d)
      INTEGER VIRTUAL_FLAVOR_INDEX_D(%(nconfs)d)
      INTEGER MAX_VIRTUAL_FLAVOR_INDEX
      PARAMETER (MAX_VIRTUAL_FLAVOR_INDEX=%(max_virtual_flavor_index)d)
      INTEGER N_VIRTUAL_FLAVOR_CONFIGS
      PARAMETER (N_VIRTUAL_FLAVOR_CONFIGS=%(n_virtual_flavor_configs)d)
      LOGICAL HAS_PHYSICAL_FKS_CLASSES
      PARAMETER (HAS_PHYSICAL_FKS_CLASSES=%(has_physical_fks_classes)s)
      INTEGER FKS_FLAVOR_CLASS_D(%(nconfs)d)
      INTEGER FKS_TOPOLOGY_D(%(nconfs)d)
      INTEGER BORN_FKS_MAP_D(%(nconfs)d)
      INTEGER NBORN_FLAVOR_CONFIGS
      PARAMETER (NBORN_FLAVOR_CONFIGS=%(n_born_flavor_configs)d)
      INTEGER BORN_FKS_CONFIG_D(NBORN_FLAVOR_CONFIGS)
      INTEGER FKS_J_FROM_I_D(%(nconfs)d, NEXTERNAL, 0:NEXTERNAL)
      INTEGER PARTICLE_TYPE_D(%(nconfs)d, NEXTERNAL), PDG_TYPE_D(%(nconfs)d, NEXTERNAL)
      LOGICAL PARTICLE_TAG_D(%(nconfs)d, NEXTERNAL)
      REAL*8 PARTICLE_CHARGE_D(%(nconfs)d, NEXTERNAL)
      INTEGER BORN_PDG_TYPE_D(%(nconfs)d, NEXTERNAL-1)
      REAL*8 BORN_PARTICLE_CHARGE_D(%(nconfs)d, NEXTERNAL-1)
      INTEGER EXTRA_CNT_PDG_D(%(nconfs)d, NEXTERNAL-1)
      INTEGER EXTRA_CNT_COLOR_D(%(nconfs)d, NEXTERNAL-1)
      REAL*8 EXTRA_CNT_CHARGE_D(%(nconfs)d, NEXTERNAL-1)
      LOGICAL SPLIT_TYPE_D(%(nconfs)d, %(nsplitorders)d)
      LOGICAL NEED_COLOR_LINKS_D(%(nconfs)d), NEED_CHARGE_LINKS_D(%(nconfs)d)
      
data fks_i_D / %(fks_i_values)s /
data fks_j_D / %(fks_j_values)s /

data need_color_links_d / %(need_color_links)s /
data need_charge_links_d / %(need_charge_links)s /

C     Local matrix-element flavor rows selected by each physical FKS class.
C     All indices are one-based; zero means that the optional matrix element
C     (extra counterterm or virtual) is absent for this class.
data real_flavor_index_d / %(real_flavor_index_values)s /
data born_flavor_index_d / %(born_flavor_index_values)s /
data extra_cnt_flavor_index_d / %(extra_cnt_flavor_index_values)s /
data virtual_flavor_index_d / %(virtual_flavor_index_values)s /
data fks_flavor_class_d / %(fks_flavor_class_values)s /
data fks_topology_d / %(fks_topology_values)s /
data born_fks_map_d / %(born_fks_map_values)s /
data born_fks_config_d / %(born_fks_config_values)s /

%(fks_j_from_i_lines)s

C     Split type: the order of elements is %(splitorders_name)s
%(split_type_lines)s

C     Extra cnt: if the real emission needs an extra cnt (g/a > qqbar), then
C     this is the id of the extra cnt. Otherwise it should be set to 0.
C     If not 0, store also the position in the splitorders array of the 
C     coupling corresponding to the splitting type
data extra_cnt_D / %(extra_cnt_values)s /
data isplitorder_born_D / %(isplitorder_born_values)s /
data isplitorder_cnt_D / %(isplitorder_cnt_values)s /

C     
C     Particle type:
C     octet = 8, triplet = 3, singlet = 1
%(col_lines)s

C     
C     Particle type according to PDG:
C     
%(pdg_lines)s

C
C     Particle charge:
C     charge is set 0. with QCD corrections, which is irrelevant
%(charge_lines)s

C     Physical charges of the corresponding underlying-Born row.
%(born_pdg_lines)s
%(born_charge_lines)s

C     Physical metadata of the alternate extra-counterterm Born row.
C     Rows without an extra counterterm contain zero PDGs/charges and
C     color-singlet entries.
%(extra_cnt_pdg_lines)s
%(extra_cnt_color_lines)s
%(extra_cnt_charge_lines)s

C     
C     Tagged particles:
C     
%(tag_lines)s
