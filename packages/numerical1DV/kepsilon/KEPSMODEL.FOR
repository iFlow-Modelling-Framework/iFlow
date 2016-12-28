      subroutine kepsmodel(vicww, rtur1,
     *            uz, rhoz, ustar, zeta, thick,
     *            kmax, pmax, tmax, llmax, 
     *            omega, dp, z0, relaxation  )
      
c************************************************************************
c         DELTARES DSC-NSS
c
c             Module: Subroutine kepsmodel
c
c           Function: load turbulence details from file and write to output
c                     
c
c               Date: 06-03-2014
c         Programmer: Y.M.Dijkstra
c************************************************************************
c Input parameters
      integer             kmax, pmax, llmax, tmax
      double precision    rtur1(0:1000,1:2)
      complex*16          vicww(0:1000,0:20)
      
      double precision    thick(1:kmax)
      complex*16          uz(0:kmax, 0:pmax), 
     *                    rhoz(0:kmax, 0:pmax), 
     *                    ustar(0:pmax), zeta(0:pmax)
      double precision    dt, omega, dp, z0, relaxation 

c Other parameters
      integer             tt, nn, rest, tcount
      double precision    uz_time(0:kmax), rhoz_time(0:kmax), 
     *                    zeta_time, ustb_time, vic(0:kmax),
     *                    vic_time(0:kmax, 0:63), vz(0:kmax)
      double precision    aa(0:kmax),bb(0:kmax),cc(0:kmax),dd(0:kmax),
     *            terDk(0:kmax,4) ,terPk(0:kmax,4) ,terEk(0:kmax,4),
     *            terDp(0:kmax,4) ,terPp(0:kmax,4) ,terEp(0:kmax,4),
     * 	        terBk(0:kmax,4) ,terBp(0:kmax,4) ,terDkk(0:kmax,4),
     *            terDpp(0:kmax,4),terDkp(0:kmax,4)
      
      double precision    time, vicdif, newvic, vicmax, victest
      complex*16          victem
      complex*8           comp_time
      complex*8           I
      data   epsd   / 1.e-32/      
      data   NFFT   / 64/         ! number of points needed for the FFT
c----------------------------------------------------------------------
c Initialisation
c----------------------------------------------------------------------
      alpha   = 1.
      vonkar  = 0.41
      cmukep  = 0.09
      viscou  = 1.e-6
      pi      = 2.*acos(0.0)
      dt      = (2*pi/omega)/tmax
      I       = (0.,1.)
c     set dv/dz to zero by default. Can be used later as extra source of mixing      
      do k=0,kmax
          vz(k) = 0.
      enddo
      
c     set arrays to zero 
      tcount = -1
      do k=0,kmax
          do n=1,4
            terDk (k,n)=0.0
	      terPk (k,n)=0.0
	      terEk (k,n)=0.0
	      terDp (k,n)=0.0
	      terPp (k,n)=0.0
	      terEp (k,n)=0.0
	      terBk (k,n)=0.0
	      terBp (k,n)=0.0
	      terDkk(k,n)=0.0
	      terDpp(k,n)=0.0
	      terDkp(k,n)=0.0
          enddo
          aa(k) = 0.0
          bb(k) = 0.0
          cc(k) = 0.0
          dd(k) = 0.0
      enddo

c     make sure that tmax is a multiple of NFFT
      rest = mod(tmax, NFFT)
      if (rest.gt.0) then
          tmax = tmax+(NFFT-rest)
      endif

c----------------------------------------------------------------------
c Loop over time 
c----------------------------------------------------------------------
      do 100 tt=0, llmax*tmax-1
          time = dt*tt
                    
c         1. Make time series of velocity gradient, density gradient, wl, friction, nu_t
c         u_z and rho_z
          do k=1,kmax-1
              uz_time(k) = 0.
              rhoz_time(k) = 0.
              do nn=0, pmax
                  comp_time = I*nn*omega*time
                  uz_time(k) = uz_time(k) + 
     *                real(uz(k,nn)*cexp(comp_time))
                  rhoz_time(k) = rhoz_time(k) + 
     *                real(rhoz(k,nn)*cexp(comp_time))
              enddo
          enddo
          
c         u at bed, bed friction and water level      
          ustb_time = 0.
          zeta_time = 0.
          do nn=0, pmax
              comp_time = I*nn*omega*time
              ustb_time = ustb_time + real(ustar(nn)*cexp(comp_time))
              zeta_time = zeta_time + real(zeta(nn)*cexp(comp_time))
          enddo
          
c         calculate eddy viscosity 
          do k=0,kmax
              vic(k) = cmukep*rtur1(k,1)**2
     *        /max(epsd,rtur1(k,2))
          enddo
      
c         2. Call turbulence model
          call tratur(  rtur1(0:kmax,:), vic,
     *                  uz_time ,vz  ,rhoz_time, ustb_time, 0  ,z0,
     *                  dt,kmax  ,thick ,zeta_time ,dp    ,
     *                  aa    ,bb    ,cc    ,dd    ,
     *                  terDk ,terPk ,terEk ,terDp ,terPp ,terEp ,
     * 	              terBk ,terBp ,terDkk,terDpp,terDkp )
             
c         load rtur1 to vic_time only on NFFT time steps
          if ((tt.ge.(llmax-1)*tmax).and.(mod(tt, tmax/NFFT).eq.0)) then
              tcount = tcount + 1
              do k=0,kmax
                  vic_time(k,tcount) = cmukep*rtur1(k,1)**2
     *                 /max(epsd,rtur1(k,2))
              enddo
          endif
          
100   continue      
      
c----------------------------------------------------------------------
c Fast fourier transform
c---------------------------------------------------------------------- 
c a. call FFT   
      do k=0,kmax
          call realft(vic_time(k,:),NFFT,1) 
      enddo      
      
c b. check convergence  & load results to vicww
c     NB. divide by NFFT and NFFT/2 due to fft 
c     i) residual
      do k=0,kmax
          vicww(k,0) = relaxation * vicww(k,0)
     *        + (1.-relaxation)*(vic_time(k,0)/NFFT+viscou)
          
c     ii) other frequencies              
          do nn=1,pmax
              victem = vic_time(k,2*nn)+I*vic_time(k,2*nn+1)
              victem = victem*2./NFFT
                  
              vicww(k,nn) = relaxation * vicww(k,nn)
     *                          + (1.-relaxation)*victem
          enddo
      enddo
              
c c. test positivity
      do k=0,kmax
          vicmax = 0.
          do tt=0,NFFT-1
              time = tt*dt
              victest = 0.
              do nn=1,pmax
                  comp_time = nn*I*omega*time
                  victest = victest+
     *                real(vicww(k,nn)*cexp(comp_time))
              enddo
              vicmax = min(vicmax,victest)
          enddo
          if (vicmax+real(vicww(k,0)).lt.0.) then
              do nn=1,pmax
c             correction by reducing oscillating components
              vicww(k,nn) = vicww(k,nn)
     *            *real(vicww(k,0))/abs(vicmax)
              enddo                      
          endif                  
      enddo
     
8888  continue
9999  continue 
      
      return
      end
