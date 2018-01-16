"""
fft

Date: 22-Feb-16
Authors: Y.M. Dijkstra
"""
import numpy as np
# from invfft import invfft
# from Timer import Timer
#
# t0 = Timer()
# t1 = Timer()

def invfft2(u, dim, enddim):
    N = u.shape[dim]
    omega = 2*np.pi/enddim
    M = np.exp(np.arange(0, N).reshape(N, 1)*1j*omega*np.arange(0, enddim).reshape(1, enddim)).reshape((1,)*dim+(N, enddim)+(1,)*(len(u.shape)-dim-1))
    c = u.reshape(u.shape[:dim]+(N, 1)+u.shape[dim+1:])*M
    c = np.real(np.sum(c, axis=dim))
    # c = np.real(np.fft.ifft(u, axis=dim)*N)
    return c

# u = np.random.randn(200,50, 3)

# for i in range(0, 10):
#     t0.tic()
#     c = invfft2(u, 2, 90)
#     t0.toc()
#
#     # t1.tic()
#     # u2 = np.concatenate((u, np.zeros((u.shape[0], u.shape[1], c.shape[2]-u.shape[2]))), 2)
#     # c2 = invfft(u2, 2)
#     # t1.toc()
#
# t0.disp('new: ')
# t1.disp('old: ')

# import step as st
# import matplotlib.pyplot as plt
# st.configure()
# plt.figure(1, figsize=(1,1))
# a = np.arange(80, 120, 1)
# for i in a:
#     t1.reset()
#     t1.tic()
#     u2 = np.concatenate((u, np.zeros((u.shape[0], u.shape[1], i))), 2)
#     c2 = invfft(u2, 2)
#     t1.toc()
#     t1.disp(str(i) + 'old: ')
#
#     plt.plot(i, np.log10(t1.timespent), 'ko')
# st.show()

# import step as st
# import matplotlib.pyplot as plt
# st.configure()
# plt.figure(1, figsize=(1,1))
# a = np.arange(80, 200, 1)
# for i in a:
#     t0.reset()
#     t0.tic()
#     c2 = invfft2(u, 2, i)
#     t0.toc()
#     t0.disp(str(i) + 'old: ')
#
#     plt.plot(i, np.log10(t0.timespent), 'ko')
# st.show()