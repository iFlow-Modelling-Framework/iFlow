      subroutine solve(kmin  ,kmax  ,
     *                 aa    ,bb    ,cc    ,dd   )
      
c************************************************************************
c
c         D e l f t      H y d r a u l i c s   -   Section    E & Z
c
c             Module: Subroutne SOLVE
c           Function: Solves tri-diagonal system in vertical
c
c        Method used: Gauss elimination for solving tri-diagonal system:
c                     aa(k)*x(k-1) + bb(k)*x(k) + cc(k)*x(k+1) = dd(k)
c
c               Date: 15-12-1995
c         Programmer: G.S. Stelling, J.v.Kester, R.E. Uittenbogaard
c************************************************************************
c
      integer            kmin, kmax
      double precision   aa(kmin:kmax), bb(kmin:kmax), 
     *                   cc(kmin:kmax), dd(kmin:kmax)      
      double precision   bi 

      bi       = 1./bb(kmin)
      bb(kmin) = bi
      cc(kmin) = cc(kmin)*bi
c
      do 10 k  = kmin+1,kmax
         bi    = 1./(bb(k)-aa(k)*cc(k-1))
         bb(k) = bi
         cc(k) = cc(k)*bi
   10 continue
c
c     Right hand side:
c
      dd(kmin) = dd(kmin)*bb(kmin)
c
      do 20 k  = kmin+1,kmax
         dd(k) = bb(k)*(dd(k)-aa(k)*dd(k-1))
   20 continue
c
      do 30 k   = kmax-1,kmin,-1
         dd(k)  = dd(k)-cc(k)*dd(k+1)
   30 continue
c
      return
      end
