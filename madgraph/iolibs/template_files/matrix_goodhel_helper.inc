     subroutine write_good_hel(stream_id)
        implicit none
        include 'maxamps.inc'
        integer stream_id
        INTEGER                 NCOMB
        PARAMETER (             NCOMB=%(ncomb)d)
        LOGICAL GOODHEL(NCOMB, MAXFLAVPERPROC)
        INTEGER NTRY(MAXFLAVPERPROC)
        common/BLOCK_GOODHEL/NTRY,GOODHEL
C       Persist the C-parity de-duplication verdict next to the good
C       helicities. Reading the file skips the scan (NTRY is forced past
C       MAXTRIES below), so a run that never completed its scan must hand on
C       "broken" rather than the optimistic default -- otherwise the next job
C       would de-duplicate on a pairing nothing ever verified.
        INTEGER NCSYMTOT
        PARAMETER (NCSYMTOT=MAXFLAVPERPROC*MAXSPROC)
        INTEGER CSYMBAD(NCSYMTOT), NCSCAN(NCSYMTOT)
        common/BLOCK_CSYM/CSYMBAD,NCSCAN
        INTEGER CSYMOUT(NCSYMTOT)
        INTEGER ICS
        write(stream_id,*) GOODHEL
        do ICS=1,NCSYMTOT
           if (CSYMBAD(ICS).eq.0.and.NCSCAN(ICS).ge.20) then
              CSYMOUT(ICS) = 0
           else
              CSYMOUT(ICS) = 1
           endif
        enddo
        write(stream_id,*) CSYMOUT
        return
        end


        subroutine read_good_hel(stream_id)
        implicit none
        include 'genps.inc'
        include 'maxamps.inc'
        integer stream_id
        INTEGER                 NCOMB
        PARAMETER (             NCOMB=%(ncomb)d)
        LOGICAL GOODHEL(NCOMB, MAXFLAVPERPROC)
        INTEGER NTRY(MAXFLAVPERPROC)
        common/BLOCK_GOODHEL/NTRY,GOODHEL
        INTEGER NCSYMTOT
        PARAMETER (NCSYMTOT=MAXFLAVPERPROC*MAXSPROC)
        INTEGER CSYMBAD(NCSYMTOT), NCSCAN(NCSYMTOT)
        common/BLOCK_CSYM/CSYMBAD,NCSCAN
        INTEGER IOCS
        read(stream_id,*) GOODHEL
        NTRY(:) = MAXTRIES + 1
C       Inherit the verdict. The good-helicity filter now gates the full-sum
C       loop, so the scan can no longer be redone in this job (the rows it
C       would compare are not all evaluated any more): treat a missing or
C       unfinished verdict as "do not de-duplicate".
        read(stream_id,*,iostat=IOCS) CSYMBAD
        if (IOCS.ne.0) then
           CSYMBAD(:) = 1
        endif
        NCSCAN(:) = 20
        return
        end

        subroutine init_good_hel()
        implicit none
        include 'maxamps.inc'
        INTEGER                 NCOMB
        PARAMETER (             NCOMB=%(ncomb)d)
        LOGICAL GOODHEL(NCOMB, MAXFLAVPERPROC)
        INTEGER NTRY(MAXFLAVPERPROC)
        INTEGER I

        GOODHEL(:,:) = .false.
        NTRY(:) = 0
        end
        
        integer function get_maxsproc()
        implicit none
        get_maxsproc = 1
        return 
        end
        
