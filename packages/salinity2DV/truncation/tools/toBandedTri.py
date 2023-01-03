import numpy as np


########################################################################################################################
## tools
########################################################################################################################
def toBandedTri(A, B, C):
    jmax = B.shape[0]-1
    M = B.shape[-1]

    Matbanded = np.zeros((4*M-1,(jmax+1)*M))
    # main diagonal block
    Matbanded[2*M-1, :] += np.diagonal(B, offset=0, axis1=1, axis2=2).flatten()
    for n in range(1, M):
        d = np.zeros((jmax+1, M))
        d[:, n:] = np.diagonal(B, offset=n, axis1=1, axis2=2)
        Matbanded[2*M-1-n, :] += d.flatten()

        dn = np.zeros((jmax+1, M))
        dn[:, :-n] = np.diagonal(B, offset=-n, axis1=1, axis2=2)
        Matbanded[2*M-1+n, :] += dn.flatten()

    # upper diagonal block
    d = np.zeros((jmax+1, M))
    d[1:, :] = np.diagonal(C[:-1], offset=0, axis1=1, axis2=2)
    Matbanded[1*M-1, :] += d.flatten()
    for n in range(1, M):
        d = np.zeros((jmax+1, M))
        d[1:, n:] = np.diagonal(C[:-1], offset=n, axis1=1, axis2=2)
        Matbanded[1*M-1-n, :] += d.flatten()

        dn = np.zeros((jmax+1, M))
        dn[1:, :-n] = np.diagonal(C[:-1], offset=-n, axis1=1, axis2=2)
        Matbanded[1*M-1+n, :] += dn.flatten()

    # lower diagonal block
    d = np.zeros((jmax+1, M))
    d[:-1, :] = np.diagonal(A[1:], offset=0, axis1=1, axis2=2)
    Matbanded[3*M-1, :] += d.flatten()
    for n in range(1, M):
        d = np.zeros((jmax+1, M))
        d[:-1, n:] = np.diagonal(A[1:], offset=n, axis1=1, axis2=2)
        Matbanded[3*M-1-n, :] += d.flatten()

        dn = np.zeros((jmax+1, M))
        dn[:-1, :-n] = np.diagonal(A[1:], offset=-n, axis1=1, axis2=2)
        Matbanded[3*M-1+n, :] += dn.flatten()

    return Matbanded

def toBandedPenta(A, B, C, D, E):
    jmax = B.shape[0]-1
    M = B.shape[-1]

    Matbanded = np.zeros((6*M-1,(jmax+1)*M))
    Matlist = [A, B, C, D, E]

    # iterate over blocks to put them in the total matrix
    for n, i in enumerate(range(-2, 3)):      # i<0: lower diags; i>0: upper diags
        U = Matlist[n]

        if np.any(U!=0):
            # main diagonal of block
            d = np.zeros((jmax+1, M))
            if i==0:
                d = np.diagonal(U, offset=0, axis1=1, axis2=2)
            elif i>0:
                d[i:, :] = np.diagonal(U[:-i], offset=0, axis1=1, axis2=2)
            else:
                d[:i, :] = np.diagonal(U[abs(i):], offset=0, axis1=1, axis2=2)
            Matbanded[(3-i)*M-1, :] += d.flatten()

            # other diagonals of block
            for n in range(1, M):
                # upper diagonals of block
                d = np.zeros((jmax+1, M))
                if i==0:
                    d[:, n:] = np.diagonal(U, offset=n, axis1=1, axis2=2)
                elif i>0:
                    d[i:, n:] = np.diagonal(U[:-i], offset=n, axis1=1, axis2=2)
                else:
                    d[:i, n:] = np.diagonal(U[abs(i):], offset=n, axis1=1, axis2=2)
                Matbanded[(3-i)*M-1-n, :] += d.flatten()

                # lower diagonals of block
                dn = np.zeros((jmax+1, M))
                if i==0:
                    dn[:, :-n] = np.diagonal(U, offset=-n, axis1=1, axis2=2)
                elif i>0:
                    dn[i:, :-n] = np.diagonal(U[:-i], offset=-n, axis1=1, axis2=2)
                else:
                    dn[:i, :-n] = np.diagonal(U[abs(i):], offset=-n, axis1=1, axis2=2)
                Matbanded[(3-i)*M-1+n, :] += dn.flatten()

    return Matbanded

def toBandedPenta2(A):
    jmax = A.shape[0]-1
    M = A.shape[1]

    Matbanded = np.zeros((6*M-1,(jmax+1)*M))
    Matlist = [A[:, :, :M], A[:, :, M:2*M], A[:, :, 2*M:3*M], A[:, :, 3*M:4*M], A[:, :, 4*M:5*M]]

    # iterate over blocks to put them in the total matrix
    for n, i in enumerate(range(-2, 3)):      # i<0: lower diags; i>0: upper diags
        U = Matlist[n]

        if np.any(U!=0):
            # main diagonal of block
            d = np.zeros((jmax+1, M))
            if i==0:
                d = np.diagonal(U, offset=0, axis1=1, axis2=2)
            elif i>0:
                d[i:, :] = np.diagonal(U[:-i], offset=0, axis1=1, axis2=2)
            else:
                d[:i, :] = np.diagonal(U[abs(i):], offset=0, axis1=1, axis2=2)
            Matbanded[(3-i)*M-1, :] += d.flatten()

            # other diagonals of block
            for n in range(1, M):
                # upper diagonals of block
                d = np.zeros((jmax+1, M))
                if i==0:
                    d[:, n:] = np.diagonal(U, offset=n, axis1=1, axis2=2)
                elif i>0:
                    d[i:, n:] = np.diagonal(U[:-i], offset=n, axis1=1, axis2=2)
                else:
                    d[:i, n:] = np.diagonal(U[abs(i):], offset=n, axis1=1, axis2=2)
                Matbanded[(3-i)*M-1-n, :] += d.flatten()

                # lower diagonals of block
                dn = np.zeros((jmax+1, M))
                if i==0:
                    dn[:, :-n] = np.diagonal(U, offset=-n, axis1=1, axis2=2)
                elif i>0:
                    dn[i:, :-n] = np.diagonal(U[:-i], offset=-n, axis1=1, axis2=2)
                else:
                    dn[:i, :-n] = np.diagonal(U[abs(i):], offset=-n, axis1=1, axis2=2)
                Matbanded[(3-i)*M-1+n, :] += dn.flatten()

    return Matbanded

def toBandedTriTensor(A, B, C, addDim):
    jmax = B.shape[0]-1
    M = B.shape[-1]
    N = B.shape[addDim]
    Matbanded = np.zeros((4*M-1,(jmax+1)*M, N))

    for n in range(0, N):
        Matbanded[:, :, n] = toBandedTri(A[[slice(None)]*addDim+[n, Ellipsis]], B[[slice(None)]*addDim+[n, Ellipsis]], C[[slice(None)]*addDim+[n, Ellipsis]])
    return Matbanded

def bandedMatVec(A, b):
    N = A.shape[0]
    M = A.shape[1]
    mid_index = int((N-1)/2)

    if len(A.shape)==2:
        B = b.reshape((1, M))*np.ones((N, 1))     # reshape vector to bandmatrix, copying its items
    else:
        B = b.reshape((1, M,1))*np.ones((N, 1,1))     # reshape vector to bandmatrix, copying its items
    AB = A*B                                # elementwise multiplication

    ABshift = np.zeros(AB.shape)
    for n, ind in enumerate(range(-mid_index, mid_index+1)):
        ABshift[n, np.maximum(0, n-mid_index):np.minimum(M, M+n-mid_index)] = AB[n, np.maximum(0, mid_index-n):np.minimum(M, M+mid_index-n)]
    # ABshift = shift(AB, range(-mid_index, mid_index+1))
    Ab = np.sum(ABshift, axis=0)
    return Ab

def bandedToFull(A):
    M = A.shape[1]
    AFull = np.zeros((M, M))
    midindex = int((A.shape[0]-1)/2)

    for n, ind in enumerate(range(-midindex, midindex+1)):
        if ind <= 0:
            AFull[range(0, M+ind), range(-ind, M)] = A[n, -ind:]
        else:
            AFull[range(ind, M), range(0, M-ind)] = A[n, :-ind]
    return AFull

