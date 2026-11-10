      PROGRAM CMS_PROBE
      IMPLICIT NONE
      INCLUDE 'coupl.inc'
      INCLUDE 'input.inc'
      CALL SETPARA('param_card.dat')
      OPEN(73,FILE='cms_masses.dat',STATUS='REPLACE')
      WRITE(73,'(ES26.17E3)') MDL_CMSPARAM
      WRITE(73,'(4ES26.17E3)') MDL_MW,MDL_WW,CMASS_MDL_MW**2
      WRITE(73,'(4ES26.17E3)') MDL_MZ,MDL_WZ,CMASS_MDL_MZ**2
      CLOSE(73)
      END
