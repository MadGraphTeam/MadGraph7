
subroutine ordered_clear_tir_cache()
logical done(1,0:2)
common/ordered_TIRCACHING/done
done=.false.
end
integer function ordered_tircache_index(mode)
integer mode
ordered_tircache_index=mode
end
subroutine ordered_inittir()
end
subroutine ordered_collierloop(mode,n,rank,p,pden,m,coefs,errors)
integer mode,n,rank,j,k
real*8 p(0:3,n),pden(0:3,n-1),v
complex*16 m(n),coefs(0:209,3),errors(0:209,3)
v=sum(pden)*1d-9+sum(real(m))*1d-5
do k=1,3
do j=0,209
coefs(j,k)=cmplx(v+(j+1)*1d-4,k*1d-3,8)
errors(j,k)=0
enddo
enddo
end
