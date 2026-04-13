import numpy as np
def im2col_loops(x, kernel_size, stride=1, padding=0):
    if isinstance(kernel_size, int):
        K_h, K_w = kernel_size, kernel_size  # FIX: Duplicate the int
    else:
        K_h, K_w = kernel_size
    
    if isinstance(stride, int):
        s_h, s_w = stride, stride
    else:
        s_h, s_w = stride

    if isinstance(padding, int):
        p_h, p_w = padding, padding
    else:
        p_h, p_w = padding
    
    N, C, H_in, W_in = x.shape

    H_out = ((H_in + 2 * p_h - K_h) // s_h) + 1
    W_out = ((W_in + 2 * p_w - K_w) // s_w) + 1

    x_padded = np.pad(
        x,
        ((0, 0), (0, 0), (p_h, p_h), (p_w, p_w)),
        mode='constant',
        constant_values=0.0
    )
    
    col_ = C * K_h * K_w
    row_ = N * H_out * W_out
    out = np.zeros((col_, row_), dtype=np.float32) 
    
    idx = 0
    for n in range(N):
        for i in range(H_out):
            for j in range(W_out):

                h_start = i * s_h
                w_start = j * s_w

                patch = x_padded[
                    n,
                    :,  
                    h_start : h_start + K_h,
                    w_start : w_start + K_w
                ]
                out[:, idx] = patch.transpose(0, 2, 1).flatten()
                idx += 1  
    return out

def col2im_loops(cols, x_shape, kernel_size, stride=1, padding=0):
    N, C, H_in, W_in = x_shape
    
    if isinstance(kernel_size, int):
        K_h, K_w = kernel_size, kernel_size
    else:
        K_h, K_w = kernel_size
        
    if isinstance(stride, int):
        s_h, s_w = stride, stride
    else:
        s_h, s_w = stride
        
    if isinstance(padding, int):
        p_h, p_w = padding, padding
    else:
        p_h, p_w = padding
        
    H_out = ((H_in + 2 * p_h - K_h) // s_h) + 1
    W_out = ((W_in + 2 * p_w - K_w) // s_w) + 1
    
    dx_padded = np.zeros(
        (N, C , H_in+2*p_h , W_in+2*p_w),
        dtype= np.float32
    )
    
    idx = 0
    for n in range(N):
        for i in range(H_out):
            for j in range(W_out):
                h_start = i * s_h
                w_start = j * s_w
                
                col_patch = cols[:,idx]
                patch_3d = col_patch.reshape(C, K_w, K_h).transpose(0, 2, 1)
                
                dx_padded[
                    n,
                    :,
                    h_start : h_start + K_h,
                    w_start : w_start + K_w
                ] += patch_3d
                
                idx += 1
    if p_h > 0 or p_w > 0:
        dx = dx_padded[:, :, p_h : p_h + H_in, p_w : p_w + W_in]
    else:
        dx = dx_padded   
        
    return dx 

import numpy as np

def get_im2col_indices(x_shape, K_h, K_w, p_h, p_w, s_h, s_w):
    
    N, C, H_in, W_in = x_shape
    
    H_out = ((H_in + 2 * p_h - K_h) // s_h) + 1
    W_out = ((W_in + 2 * p_w - K_w) // s_w) + 1

    # 1. Base Patch Coordinates
    i0 = np.repeat(np.arange(K_h), K_w)
    i0 = np.tile(i0, C)
    j0 = np.tile(np.arange(K_w), K_h * C)
    
    # 2. Offset Coordinates (Stride)
    i1 = s_h * np.repeat(np.arange(H_out), W_out)
    j1 = s_w * np.tile(np.arange(W_out), H_out)
    
    # 3. Broadcast together
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    
    # 4. Channel Coordinates
    k = np.repeat(np.arange(C), K_h * K_w).reshape(-1, 1)

    return (k.astype(int), i.astype(int), j.astype(int))


def img2col(x, kernel_size, stride=1, padding=0):

    if isinstance(kernel_size, int):
        K_h, K_w = kernel_size, kernel_size
    else:
        K_h, K_w = kernel_size
    
    if isinstance(stride, int):
        s_h, s_w = stride, stride
    else:
        s_h, s_w = stride

    if isinstance(padding, int):
        p_h, p_w = padding, padding
    else:
        p_h, p_w = padding
    
    x_padded = np.pad(
        x,
        ((0, 0), (0, 0), (p_h, p_h), (p_w, p_w)),
        mode='constant',
        constant_values=0.0
    )

    k, i, j = get_im2col_indices(x.shape, K_h, K_w, p_h, p_w, s_h, s_w)

    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    
    cols = cols.transpose(1, 2, 0).reshape(K_h * K_w * C, -1)
    
    return cols


def col2img(cols, x_shape, kernel_size, stride=1, padding=0):
    
    N, C, H_in, W_in = x_shape
    
    if isinstance(kernel_size, int):
        K_h, K_w = kernel_size, kernel_size
    else:
        K_h, K_w = kernel_size
        
    if isinstance(stride, int):
        s_h, s_w = stride, stride
    else:
        s_h, s_w = stride
        
    if isinstance(padding, int):
        p_h, p_w = padding, padding
    else:
        p_h, p_w = padding
        
    dx_padded = np.zeros(
        (N, C, H_in + 2 * p_h, W_in + 2 * p_w), 
        dtype=np.float32
    )
    
    k, i, j = get_im2col_indices(x_shape, K_h, K_w, p_h, p_w, s_h, s_w)

    cols_reshaped = cols.reshape(C * K_h * K_w, N, -1).transpose(1, 0, 2)
    
    np.add.at(dx_padded, (slice(None), k, i, j), cols_reshaped)
    
    if p_h > 0 or p_w > 0:
        dx = dx_padded[:, :, p_h : p_h + H_in, p_w : p_w + W_in]
    else:
        dx = dx_padded
        
    return dx