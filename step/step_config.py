savedpi = 300       # dpi for saving (only relevant for non-vector images)

# Normal
# dpi = 120           # dpi for visualisation
# wunit = 3.5         # single width unit (inch)
# hunit = 2.8        # single height unit (inch)
# markersize = 6.0    # markersize for all markers
# fontsize = 11       # font size for title and axis labels
# fontsize2 = 9      # font size for tick labels and legends
# fontsize3 = 8      # font size for tick labels and legends
# fontfamily = 'sans-serif'
# font = {}           #       optionally add a line font['sans-serif'] = 'fontname' or font['serif'] = 'fontname'
# forcefontsize = False        # force the above fontsizes

# Thesis
dpi = 170               # dpi for visualisation
wunit = 2.51            # single width unit (inch)
hunit = 2.0             # single height unit (inch)
markersize = 4.0        # markersize for all markers
fontsize = 8.2          # font size for title and axis labels
fontsize2 = 7.2         # font size for tick labels and legends
fontsize3 = 6.2
fontfamily = 'serif'
font = {}
font['serif'] = u'Utopia'
# forcefontsize = True        # force the above fontsizes
forcefontsize = False        # force the above fontsizes

names = {
    'zeta': '$\zeta$ total',
    'zeta0': '$\zeta^0$',
    'zeta1': '$\zeta^1$',
    'zeta2': '$\zeta^2$',
    'zeta3': '$\zeta^3$',
    'zeta4': '$\zeta^4$',
    'zeta5': '$\zeta^5$',
    'zeta6': '$\zeta^6$',
    'u': '$u$ total',
    'u0': '$u^0$',
    'u1': '$u^1$',
    'u2': '$u^2$',
    'u3': '$u^3$',
    'u4': '$u^4$',
    'u5': '$u^5$',
    'u6': '$u^6$',
    'w0': '$w^0$',
    'w1': '$w^1$',
    'w2': '$w^2$',
    'w3': '$w^3$',
    'w4': '$w^4$',
    'w5': '$w^5$',
    'w6': '$w^6$',
    's0': '$s^0$',
    's1': '$s^1$',
    's2': '$s^2$',
    's3': '$s^3$',
    'c': 'c',
    'c0': '$c^0$',
    'c1': '$c^1$',
    'c2': '$c^2$',
    'c3': '$c^3$',
    'csubt': 'Subtidal sediment\nconcentration',
    'Av': '$A_v^0$',
    'Av1': '$A_v^1$',
    'Av2': '$A_v^2$',
    'Av3': '$A_v^3$',
    'Av4': '$A_v^4$',
    'Av5': '$A_v^5$',
    'Av6': '$A_v^6$',
    'x': 'x',
    'z': 'z',
    'f': 'f',
    'H': 'H',
    '-H': 'H',
    'B': 'B',
    'baroc': 'Baroclinic',
    'adv': 'Advection',
    'stokes': 'Tidal return flow',
    'mixing': 'Mixing',
    'tide': 'Tide',
    'river': 'River',
    'nostress': 'Vel.-depth asym.',
    'densitydrift': 'Density-induced return flow',
    'R': 'R',
    'T': 'Total transport',
    'tide-baroc': 'Baroclinic',
    'tide-adv': 'Advection',
    'tide-stokes': 'Tidal return flow',
    'tide-tide': 'Tide',
    'tide-river': 'River',
    'tide-nostress': 'No-stress',
    'Tdiff': 'Diff. transport',
    'TM0': 'Res. transport',
    'TM2': 'M$_2$ transport',
    'TM4': 'M$_4$ transport',
    'Tstokes' : '$T_{stokes}$',
    'TM2M0': '$T_{M_2}^{M_0}$',
    'TM2M2': '$T_{M_2}^{M_2}$',
    'TM2M4': '$T_{M_2}^{M_4}$',
    'TM0stokes': 'Stokes transport'}

transportlabels = {
    'T' : 'T',
    'TM0' : '$T_{res}$',
    'TM2' : '$T_{M_2}$',
    'TM4' : '$T_{M_4}$',
    'Tdiff' : '$T_{diff}$',
    'Tstokes' : '$T_{stokes}$',
    'TM2M0' : '$T_{M_2}^{M_0}$',
    'TM2M2' : '$T_{M_2}^{M_2}$',
    'TM2M4' : '$T_{M_2}^{M_4}$',
    'TM0stokes': 'stokes',
    'baroc': 'baroc',
    'adv': 'advection',
    'stokes': 'tidal return flow',
    'tide': 'ext. $M_4$ tide',
    'river': 'river',
    'nostress': 'vel.-depth asym.',
    'river_river': 'river-river',
    'sedadv': 'sed. advection',
    'return': 'Stokes return flow',
    'drift': 'Stokes drift'
}

units = {
    'zeta': 'm',
    'zeta0': 'm',
    'zeta1': 'm',
    'zeta2': 'm',
    'zeta3': 'm',
    'zeta4': 'm',
    'zeta5': 'm',
    'zeta6': 'm',
    'u': 'm/s',
    'u0': 'm/s',
    'u1': 'm/s',
    'u2': 'm/s',
    'u3': 'm/s',
    'u4': 'm/s',
    'u5': 'm/s',
    'u6': 'm/s',
    'w0': 'm/s',
    'w1': 'm/s',
    'w2': 'm/s',
    'w3': 'm/s',
    'w4': 'm/s',
    'w5': 'm/s',
    'w6': 'm/s',
    's0': 'psu',
    's1': 'psu',
    's2': 'psu',
    's3': 'psu',
    'c0': 'mg/l',
    'c1': 'mg/l',
    'c2': 'mg/l',
    'c3': 'mg/l',
    'Av': '$m^2/s$',
    'Av1': '$m^2/s$',
    'Av2': '$m^2/s$',
    'Av3': '$m^2/s$',
    'Av4': '$m^2/s$',
    'Av5': '$m^2/s$',
    'Av6': '$m^2/s$',
    'x': 'km',
    'z': 'm',
    'f': '',
    'phase': '$^{\circ}$',
    'R': 'm',
    'T': 'kg/ms',
    'c': 'mg/l',
    'H': 'm',
    '-H': 'm',
    'B': 'm',
    'csubt': '$kg/m^3$'}

conversion = {
    'x': 1/1000.,
    'c': 1000.,
    'c0': 1000.,
    'c1': 1000.,
    'c2': 1000.
}