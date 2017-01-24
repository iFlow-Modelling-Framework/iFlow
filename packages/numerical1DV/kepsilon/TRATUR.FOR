      subroutine tratur(rtur0 ,vicww,
     *                  dudz  ,dvdz  ,drhodz,ustb  ,ustw  ,z0, 
     *                  timest,kmax  ,thick ,zeta  ,dp    ,
     *                  aa    ,bb    ,cc    ,dd    ,
     *                  terDk ,terPk ,terEk ,terDp ,terPp ,terEp ,
     * 	              terBk ,terBp ,terDkk,terDpp,terDkp )
      
c************************************************************************
c
c         D e l f t      H y d r a u l i c s   -   Section    M-C-M
c
c             Module: Subroutine TRATUR
c           Function: Transport solver tke and epsilon in DPM
c
c        Method used: Teta method for integration in time.
c                     Tke and eps computed at layer interfaces.
c                     SANCTUM.
c               Date: 13:26 dinsdag 4 augustus 1998
c         Correction: YMD: Corrected error coefn2 (Bk term) & added sigrho to Bp term (dec 2013)
c                     YMD: Corrected taking TKE at old (instead of new) time level in Bp term (may 2014)
c      
c         Programmer:  R.E. Uittenbogaard
c************************************************************************

      integer     kmax
      double precision ustb, ustw, zeta, timest, dp, z0
      double precision  rtur0(0:kmax,2), vicww(0:kmax),
     *            dudz(0:kmax), dvdz(0:kmax), drhodz(0:kmax),
     *            thick(1:kmax),
     *            aa(0:kmax),bb(0:kmax),cc(0:kmax),dd(0:kmax),
     *            terDk(0:kmax,4) ,terPk(0:kmax,4) ,terEk(0:kmax,4),
     *            terDp(0:kmax,4) ,terPp(0:kmax,4) ,terEp(0:kmax,4),
     * 	        terBk(0:kmax,4) ,terBp(0:kmax,4) ,terDkk(0:kmax,4),
     *            terDpp(0:kmax,4),terDkp(0:kmax,4)
            
      double precision  sigrho(0:kmax),cmukep(0:kmax)
      double precision  buoflu(0:kmax),sourtu(0:kmax,2),sinktu(0:kmax,2)
      double precision  dtinv, vicu, fac, dzdz1, dzdz2, bruva,s2,dz,dz0,
     *                  vicd, difu, difd 
      data   epsd   / 1.e-32/

c Parameter definitions
      ag      = 9.81
      rhom    = 1000
      viscou  = 1.e-6
      sigrho0 = 0.7
      cmu0    = 0.09
      c1e     = 1.44
      c2e     = 1.92
      vonkar  = 0.41
      alpha   = 1.
      
      do k=0,kmax
          sigrho(k) = sigrho0
          cmukep(k) = cmu0
          !write(*,*) 'vicww', k, vicww(k)
      enddo
      
c Preparation      
      h0     = dp+zeta
      coefn2 =-ag/(rhom)!-ag/(sigrho*rhom)
      
c************************************************************************
c Turbulence model
c************************************************************************
      do 500 l   = 1,2
c Prandtl Schmidt number Tke and Epsilon
        if (l.eq.1) pransm=1.0
        if (l.eq.2) pransm=1.3
c Matrix initialization
        dtinv    = 1.0/timest
        do 10 k  = 0,kmax
           aa(k) = 0.0
           bb(k) = dtinv
           cc(k) = 0.0
           dd(k) = rtur0(k,l)*dtinv
10      continue
c Vertical diffusion; Neumann condition on surface;
c Dirichlet condition on bed ; teta method:
         vicu     = viscou+0.5*(vicww(0)+vicww(1))/pransm
         do 20 k  = 1,kmax-1
            ku    = k+1
            kd    = k-1
            fac   = 0.5*(thick(k)+thick(ku))*h0*h0
            dzdz1 = fac*thick(k )
            dzdz2 = fac*thick(ku)
            vicd  = vicu
            vicu  = viscou+0.5*(vicww(k)+vicww(ku))/pransm
            difu  = vicu/dzdz2
            difd  = vicd/dzdz1
            
            if (l.eq.1) then 
                terDk(k,1)=-difd
                terDk(k,2)= difd+difu
                terDk(k,3)=-difu
                terDk(k,4)= 0.
                
                aa(k) = aa(k)+terDk(k,1)
                bb(k) = bb(k)+terDk(k,2)
                cc(k) = cc(k)+terDk(k,3)
            else
                terDp(k,1)=-difd
                terDp(k,2)= difd+difu
                terDp(k,3)=-difu
                terDp(k,4)= 0.

                aa(k) = aa(k)+terDp(k,1)
                bb(k) = bb(k)+terDp(k,2)
                cc(k) = cc(k)+terDp(k,3)
            endif          
c            
c Source and sink terms
            bruva     = coefn2*drhodz(k)
            buoflu(k) = vicww(k)*bruva/sigrho(k)
            
        if (l.eq.1) then
c Production, dissipation, and buoyancy term in TKE equation;
c dissipation and positive buoyancy are split by Newton linearization:
             if (buoflu(k).gt.0.0) then
             	 terBk(k,2)=+2*buoflu(k)/max(epsd, rtur0(k,1))
             	 terBk(k,4)=buoflu(k)
                 
                 dd (k)  = dd(k)+ terBk(k,4)
                 bb (k)  = bb(k)+ terBk(k,2)
             else
                 terBk(k,2)=0
             	 terBk(k,4)=-buoflu(k)
                 dd (k)  = dd(k)+ terBk(k,4)
             endif
             
             s2          = dudz(k)**2+dvdz(k)**2
             
             sourtu(k,1) = vicww(k)*s2
             sinktu(k,1) = rtur0(k,2)/max(epsd, rtur0(k,1))
             
             terPk(k,4)=sourtu(k,1)
             terEk(k,2)=2.*sinktu(k,1)
             terEk(k,4)=sinktu(k,1)*rtur0(k,1)		              
c               
             bb(k) = bb(k)+terEk(k,2)
             dd(k) = dd(k)+terPk(k,4)+terEk(k,4)	
             !write(*,*) 'Bk', k, terBk(k,2), terBk(k,3)
             !write(*,*) 'Pk', k, terPk(k,2), terPk(k,3)
             !write(*,*) 'Dk', k, terDk(k,2), terDk(k,3)
             !write(*,*) 'Ek', k, terEk(k,2), terEk(k,3)
             !write(*,*) 'k-e-k/e', k, rtur0(k,1), rtur0(k,2), 
!     *        rtur0(k,2)/max(epsd, rtur0(k,1))
c Production, dissipation, and buoyancy term in TKE equation;
c dissipation and positive buoyancy are split by Newton linearization;
c buoyancy only for unstable stratification;
c notice: application of TKE at new time level:
        else
             if (buoflu(k).gt.0.0) then
                terBp(k,2)=0
             	terBp(k,4)=0
             	
             elseif (buoflu(k).lt.0.0) then
                terBp(k,2)=0
             	  terBp(k,4) = -cmukep(k)/sigrho(k)*c1e*bruva*
     *                         rtur0(k,1)
  	
                dd(k)    = dd(k)+terBp(k,4)
             endif
             s2          = dudz(k)**2+dvdz(k)**2

             sourtu(k,2) = c1e*cmukep(k)*rtur0(k,1)*s2
             sinktu(k,2) = c2e*rtur0(k,2)/max(epsd, rtur0(k,1))
             
             terPp(k,4)=sourtu(k,2)
             terEp(k,2)=2.*sinktu(k,2)
             terEp(k,4)=sinktu(k,2)*rtur0(k,2)
c               		 
             bb(k) = bb(k)+terEp(k,2)
             dd(k) = dd(k)+terPp(k,4)+terEp(k,4)	             
        endif
c Addition of production and of dissipation to matrix ;
c observe implicit treatment by Newton linearization.

20       continue
c Boundary conditions:
        if (l.eq.1) then
c TKE at free surface
          aa(0)      = 0.0
          bb(0)      = 1.0
          cc(0)      = 0.0
          dd(0)      = ustw**2/sqrt(cmu0)
c TKE at the bed:
          aa(kmax)   = 0.0
          bb(kmax)   = 1.0
          cc(kmax)   = 0.0
          dd(kmax)   = ustb**2/sqrt(cmu0)
        else
c epsilon at the free surface:
            dz       = thick(1)*h0
            aa(0)    = 0.0
            bb(0)    = 1.0
            cc(0)    =-1.0
            dd(0)    = 4*abs(ustw)**3/(vonkar*dz)
c Flux of epsilon at the bed:
            dz       = thick(kmax)*h0
            dz0      = 0.5*dz+alpha*z0
            aa(kmax) =-1.0
            bb(kmax) = 1.0
            cc(kmax) = 0.0
            dd(kmax) = dz*abs(ustb)**3/(vonkar*dz0*dz0)
c Dirichlet formulation for epsilon
c            aa(kmax) = 0.0
c            bb(kmax) = 1.0
c            cc(kmax) = 0.0
c            dd(kmax) = abs(ustb)**3/(vonkar*alpha*z0)        
c Flux of epsilon at the bed:
c            dz       = thick(kmax)*h0
c            aa(kmax) =-1.0
c            bb(kmax) = 1.0
c            cc(kmax) = 0.0
c            dd(kmax) = 4*abs(ustb)**3/(vonkar*dz)
        endif
c Solve tri-diagonal system of equations:
        kmin = 0
        do k=0,kmax
          !write(*,*) 'coefs', k, aa(k), bb(k), cc(k), dd(k)
        enddo
        call solve(kmin ,kmax, aa, bb, cc, dd )
c Notice: below the only statement with truncation to positive values:
        do 100 k      = 0,kmax
           rtur0(k,l) = max(epsd,dd(k))
           

100     continue
500   continue
c
c
      do k=0,kmax
          !write(*,*) 'k-eps', k, rtur0(k,1), rtur0(k,2)
      enddo
      return
      end
