import torch

from torch.nn.functional import max_unpool2d, max_pool2d, pad
from lrp.utils import LayerRelevance

def prop_SPPF(*args):

    inverter, mod, relevance = args

    #relevance = torch.cat([r.view(mod.m.out_shape) for r in relevance ], dim=0)
    bs = relevance.size(0)
    relevance = inverter(mod.cv2, relevance)
    msg = relevance.scatter(which=-1)
    ch = msg.size(1) // 4
    
    r3 = msg[:, 3*ch:4*ch, ...] 

    r2 = msg[:, 2*ch:3*ch, ...] + r3
    
    r1 = msg[:, ch:2*ch, ...] + r2

    rx = msg[:, :ch, ...] + r1
    
    msg = inverter(mod.cv1, rx)
    relevance.gather([(-1, msg)])

    return relevance

def SPPF_fwd_hook(m, in_tensor: torch.Tensor, out_tensor: torch.Tensor):

    x = m.cv1(in_tensor[0])
    y1, idx1 = max_pool2d(x, kernel_size = m.m.kernel_size,
                             stride=m.m.stride,
                             padding=m.m.padding,
                             dilation=m.m.dilation,
                             return_indices=True,
                             ceil_mode=m.m.ceil_mode)
    y2, idx2 = max_pool2d(y1, kernel_size = m.m.kernel_size,
                              stride=m.m.stride,
                              padding=m.m.padding,
                              dilation=m.m.dilation,
                              return_indices=True,
                              ceil_mode=m.m.ceil_mode)
    y3, idx3 = max_pool2d(y2, kernel_size = m.m.kernel_size,
                              stride=m.m.stride,
                              padding=m.m.padding,
                              dilation=m.m.dilation,
                              return_indices=True,
                              ceil_mode=m.m.ceil_mode)
    setattr(m, "indices", [idx1, idx2, idx3])

def Concat_fwd_hook(m, in_tensors: torch.Tensor, out_tensor: torch.Tensor):

    shapes = [in_tensor.shape[m.d] for in_tensor in in_tensors[0]]

    setattr(m, "in_shapes", shapes)
    setattr(m, "out_shape", out_tensor.shape)

def prop_Concat(*args):

    _, mod, relevance = args

    # Because concatenate
    slices = relevance.scatter(-1).split(mod.in_shapes, dim=mod.d)
    relevance.gather([(to, msg) for to, msg in zip(mod.f, slices)])

    return relevance

def prop_Detect(*args):

    inverter, mod, relevance = args
    relevance_out = []

    scattered = relevance.scatter()
    for m, rel in zip(mod.m, scattered[1:]):
        to, msg = rel
        msg = torch.cat([msg[..., i] for i in range(msg.size(-1))], dim=1)
        to = to if to != mod.reg_num else -1
        out = inverter(m, msg)
        relevance_out.append((to, out))

    relevance.gather(relevance_out)

    return relevance


def prop_Conv(*args):

    inverter, mod, relevance = args

    return inverter(mod.conv, relevance)
    


def prop_C3(*args):
    # print("######## prop_C3 start #######")
    inverter, mod, relevance = args
    msg = relevance.scatter(which=-1)

    msg = inverter(mod.cv3, msg)

    # print("Output shapes (cv2, m[-1])", mod.cv2.conv.out_channels, mod.m[-1].cv2.conv.out_tensor.shape)

    c_ = msg.shape[0] - mod.cv2.conv.out_channels - 1

    msg_cv1 = msg[:, : c_, ...]
    msg_cv2 = msg[:, c_ :, ...]

    # print("before cv1, cv2",msg_cv1.shape, msg_cv2.shape)

    for m1 in reversed(mod.m):
        ## account for complications created from pruning around addition layers 
        pad_size = m1.cv2.conv.out_channels - msg_cv1.shape[1]
        # print("msg_cv1 pad size:", pad_size)
        msg_cv1 = pad(msg_cv1, (0,0,0,0,0, pad_size), "constant", 0)
        # print("\t", m1.__class__.__name__, msg_cv1.shape, m1.cv2.conv.out_channels)
        msg_cv1 = inverter(m1, msg_cv1)


    # print("after  cv1, cv2",msg_cv1.shape, msg_cv2.shape)

    msg = inverter(mod.cv1, msg_cv1[:, :mod.cv1.conv.out_channels]) + inverter(mod.cv2, msg_cv2[:, :mod.cv2.conv.out_channels])

    relevance.gather([(-1, msg)])

    # print("######## prop_C3 end #######")

    return relevance

def prop_Bottleneck(*args):

    inverter, mod, relevance_in = args

    # print("BOTTLE:", relevance_in.shape, mod.cv2.conv.out_channels)

    ar = mod.cv2.conv.out_tensor.abs()
    ax = mod.cv1.conv.in_tensor.abs()

    relevance = relevance_in #* ar / (ax + ar)
    relevance = inverter(mod.cv2, relevance)
    relevance = inverter(mod.cv1, relevance)

    ## account for pruned parts between addition layers 
    pad_size = relevance_in.shape[1] - relevance.shape[1]
    end_idx = min(relevance_in.shape[1], relevance.shape[1])

    # print("\t", relevance_in.shape, relevance.shape, pad_size, end_idx)
    if pad_size > 0: 
        relevance = pad(relevance, (0,0,0,0,0, pad_size), "constant", 0)
    elif pad_size < 0:
        relevance_in = pad(relevance_in, (0,0,0,0,0, abs(pad_size)), "constant", 0)
    else: 
        pass

    # print("\t", relevance_in[:, :end_idx].shape, relevance[:, :end_idx].shape)
    # relevance = relevance[:, :end_idx] + relevance_in[:, :end_idx] #* ax / (ax + ar)
    # print("\t", relevance_in.shape, relevance.shape)
    relevance = relevance *(1 + relevance_in) #* ax / (ax + ar)


    return relevance
