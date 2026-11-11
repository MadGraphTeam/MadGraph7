"""Observer-only source additions outside scientific calls/IEEE snapshots."""


def observer_driver(parity):
    anchors={
      '      CHARACTER*16 COMMAND\n':'      CHARACTER*16 COMMAND\n      INTEGER BULK_WIRE_STATUS\n      INTEGER*8 BULK_WIRE_CALLS,BULK_WIRE_RECORDS\n',
      'C     BULK_PARITY_BEGIN\n':'C     BULK_PARITY_BEGIN\n         CALL QP_WIRE_BEGIN()\n',
      'C     BULK_PARITY_END\n':'''C     BULK_PARITY_END
         CALL QP_WIRE_END()
         CALL QP_WIRE_HEALTH(BULK_WIRE_STATUS,
     $        BULK_WIRE_CALLS,BULK_WIRE_RECORDS)
         OPEN(79,FILE='wire-health.dat',STATUS='NEW',
     $        ACTION='WRITE',IOSTAT=IOS)
         IF(IOS.NE.0) GOTO 800
         WRITE(79,'(3I24)') BULK_WIRE_STATUS,
     $        BULK_WIRE_CALLS,BULK_WIRE_RECORDS
         CLOSE(79)
''',
    }
    for old,new in anchors.items():
        if parity.count(old)!=1:raise ValueError('Observer source anchor not unique')
        parity=parity.replace(old,new)
    return parity
