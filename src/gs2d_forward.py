import pylab as pl
import triton
import triton.language as tl

import torch

@torch.no_grad()
def preprocess(means, sigmas, rhos, tile_h, tile_w, tile_num_h, tile_num_w):
    # TODO; maybe triton impl
    '''
    :param means: shape [N, 2]
    :param sigmas: shape [N, 2]
    :param rhos:  shape [N, 1]
    :return:
        bounding box, cumsum_areas
    '''
    det_cov = sigmas[:, :1]*sigmas[:, 1:]

    # get area
    mid = 0.5*(sigmas[:, 0:1] + sigmas[:, 1:])
    lambda1 = mid - torch.sqrt(mid*mid - det_cov)
    lambda2 = mid + torch.sqrt(mid*mid - det_cov)
    lambda3 = torch.cat([lambda1, lambda2], dim=-1)
    radius = torch.ceil_(3*torch.sqrt(torch.max(lambda3, dim=-1, keepdim=True).values))

    t = torch.clip(torch.floor_((means[:, 1:2] - radius)/tile_h), min=0).to(torch.int16)
    l = torch.clip(torch.floor_((means[:, 0:1] - radius)/tile_w), min=0).to(torch.int16)
    b = torch.clip(torch.ceil_((means[:, 1:2] + radius)/tile_h), max=tile_num_h).to(torch.int16)
    r = torch.clip(torch.ceil_((means[:, 0:1] + radius)/tile_w), max=tile_num_w).to(torch.int16)
    rects = torch.cat([t, l, b, r], dim=-1)
    areas = (b-t)*(r-l).to(torch.int32)
    cumsum_areas = torch.cumsum(areas, dim=0, out=areas) # shape: [B, 1]
    return rects, cumsum_areas

@triton.jit
def duplicate_renders(
        num_gaussians,
        num_per_thread: tl.constexpr,
        tile_num_h: tl.constexpr,
        tile_num_w: tl.constexpr,
        num_renders,
        ptr_bids,
        ptr_area_sum,
        ptr_rects,
        ptr_keys,
        ptr_values,
):
    id = tl.program_id(axis=0)
    start_id = id*num_per_thread
    end_id = start_id + num_per_thread

    for gaussian_id in range(start_id, end_id):

        render_offset_base = tl.load(ptr_area_sum+gaussian_id-1, mask=gaussian_id > 0)
        ptr_key_store = ptr_keys + render_offset_base
        ptr_value_store = ptr_values + render_offset_base

        bid = tl.load(ptr_bids + gaussian_id)
        t = tl.load(ptr_rects + 4*gaussian_id).to(tl.int64)
        l = tl.load(ptr_rects + 4*gaussian_id+1).to(tl.int64)
        b = (tl.load(ptr_rects + 4*gaussian_id+2).to(tl.int64))
        r = (tl.load(ptr_rects + 4*gaussian_id+3).to(tl.int64))
        # tl.device_print("", t, l, b, r)

        for pos_y in range(t, b):
            for pos_x in range(l, r):
                # pos_y: tl.int64
                # pos_x: tl.int64
                key = ((pos_y*tile_num_w) + (pos_x) + (bid *(tile_num_h*tile_num_w))) << 8
                value = gaussian_id
                tl.store(ptr_key_store, key, gaussian_id < num_gaussians)
                tl.store(ptr_value_store, value, gaussian_id < num_gaussians)
                ptr_key_store +=1
                ptr_value_store += 1


@triton.jit
def find_ranges(
        renders_num,
        num_per_thread: tl.constexpr,
        ptr_render_keys,
        ptr_ranges,
):
    pid = tl.program_id(axis=0)
    start_id = pid*num_per_thread
    end_id = start_id + num_per_thread

    # tl.device_print("", tile_id)
    for i in range(start_id, end_id):
        current_key = tl.load(ptr_render_keys + i, mask=i < renders_num)

        previous_mask = (i-1 >= 0) & (i < renders_num)
        previous_key = tl.load(ptr_render_keys + i-1, mask=previous_mask, other=0.0)
        current_tile_id = (current_key >> 8) & 0xffffff
        previous_tile_id = (previous_key >> 8) & 0xffffff
        mask = (current_tile_id != previous_tile_id) & (i < renders_num)
        # start position store
        tl.store(ptr_ranges+current_tile_id*2, i, mask)

        mask = mask & (i != 0)
        # end position store
        tl.store(ptr_ranges+previous_tile_id*2+1, i, mask)

        # last render
        tl.store(ptr_ranges+current_tile_id*2+1, i+1, i==renders_num-1)


@triton.jit
def render(
        renders_num,
        gaussians_num,
        tile_h: tl.constexpr,
        tile_w: tl.constexpr,
        tile_h_nums: tl.constexpr,
        tile_w_nums: tl.constexpr,
        image_h: tl.constexpr,
        image_w: tl.constexpr,
        ptr_render_ranges,
        ptr_render_gids,
        ptr_means,
        ptr_invcovs,
        ptr_colors,
        ptr_opacs,
        ptr_images,
        ptr_occs,
):
    pid_b = tl.program_id(axis=0)
    pid_tile_y = tl.program_id(axis=1)
    pid_tile_x = tl.program_id(axis=2)

    # pid = pid_b + pid_tile_x + pid_tile_y
    offset = pid_b*(tile_w_nums*tile_h_nums) + pid_tile_y*tile_w_nums + pid_tile_x
    start_render = tl.load(ptr_render_ranges + offset*2)
    end_render = tl.load(ptr_render_ranges + offset*2+1)
    # tl.device_print("", start_render, end_render)
    occs = tl.full((tile_h*tile_w, 1), tl.constexpr(1.0), dtype=ptr_occs.type.element_ty)
    render_color = tl.full((tile_h*tile_w, 4), tl.constexpr(0.0),dtype=ptr_occs.type.element_ty)

    y = tl.view(tl.broadcast_to(tl.arange(0, tile_h)[:, None], (tile_h, tile_w)), (tile_h*tile_w,)) + pid_tile_y * tile_h
    x = tl.view(tl.broadcast_to(tl.arange(0, tile_w)[None, :], (tile_h, tile_w)), (tile_h*tile_w,)) + pid_tile_x * tile_w


    image_size = image_h*image_w
    batch_offset = pid_b*(image_size)
    for render_id in range(start_render, end_render):

        render_mask = (render_id < renders_num) & (render_id >= 0)
        gid = tl.load(ptr_render_gids+render_id, mask=render_mask)

        # new stupid code
        mx = tl.load(ptr_means+gid*2)
        my = tl.load(ptr_means+gid*2+1)

        d1 = x - mx + 0.5
        d2 = y - my + 0.5

        ic1 = tl.load(ptr_invcovs+gid*4)
        ic2 = tl.load(ptr_invcovs+gid*4+1)
        ic3 = tl.load(ptr_invcovs+gid*4+2)
        ic4 = tl.load(ptr_invcovs+gid*4+3)

        s =  ic1*d1*d1 + d1*d2*(ic2+ic3) + ic4*d2*d2


        opac = tl.load(ptr_opacs + gid)
        alpha = (opac*tl.exp(-0.5*s)[:, None]).to(ptr_occs.type.element_ty)
        delta_c = tl.load(ptr_colors + gid*4 + tl.arange(0, 4)[None, :])
        render_color += delta_c*alpha*occs


        if render_mask:
            occs *= (1 - alpha)

    store_offsets =  batch_offset*4 + tl.arange(0, 4)[None, :]*image_size + y[:, None]*image_w + x[:, None]
    store_masks = (y[:, None] < image_h) & (x[:, None] < image_w)
    tl.store(ptr_images+store_offsets, render_color.to(ptr_images.type.element_ty), mask=store_masks)
    store_offsets = batch_offset + y[:, None]*image_w + x[:, None]
    tl.store(ptr_occs+store_offsets, occs, store_masks)





############################





# @triton.jit
# def render_bakup(
#         num_renders: tl.constexpr,
#         num_gaussians: tl.constexpr,
#         ptr_render_ranges,
#         ptr_render_gids,
#         ptr_means,
#         ptr_invcovs,
#         ptr_colors,
#         ptr_opacs,
#         ptr_images,
#         tile_h: tl.constexpr,
#         tile_w: tl.constexpr,
#         tile_h_nums: tl.constexpr,
#         tile_w_nums: tl.constexpr,
#         image_h: tl.constexpr,
#         image_w: tl.constexpr,
#         batch_nums: tl.constexpr,
# ):
#     pid_b = tl.program_id(axis=0)
#     pid_tile_y = tl.program_id(axis=1)
#     pid_tile_x = tl.program_id(axis=2)
#
#     # pid = pid_b + pid_tile_x + pid_tile_y
#     offset = pid_b*(tile_w_nums*tile_h_nums) + pid_tile_y*tile_w_nums + pid_tile_x
#     start_render = tl.load(ptr_render_ranges + offset*2)
#     end_render = tl.load(ptr_render_ranges + offset*2+1)
#
#     # if start_render!=-1 or end_render!=-1:
#     #     tl.device_print("", start_render, end_render)
#     T = tl.full((tile_h*tile_w, 1), tl.constexpr(1.0), dtype=tl.float32)
#     color = tl.full((tile_h*tile_w, 3), tl.constexpr(0.0), dtype=tl.float32)
#
#     y = tl.view(tl.broadcast_to(tl.arange(0, tile_h)[:, None], (tile_h, tile_w)), (tile_h*tile_w, )) + pid_tile_y * tile_h
#     x = tl.view(tl.broadcast_to(tl.arange(0, tile_w)[None, :], (tile_h, tile_w)), (tile_h*tile_w, )) + pid_tile_x * tile_w
#     xy = tl.view(tl.cat(x, y, can_reorder=True), (tile_h*tile_w, 2))
#     # tile = tl.cat(tile_x, tile_y, can_reorder=True)
#     # if (pid == 0):
#     #     tl.device_print("", tile)
#     image_size = image_h*image_w
#     batch_offset = pid_b*(image_size*3)
#     # total_nums = batch_nums*3*image_size
#     for render_id in range(start_render, end_render):
#
#         render_mask = (render_id < num_renders) & (render_id >= 0)
#         # tl.device_print("", render_id, pid_tile_y, pid_tile_x)
#         gid = tl.load(ptr_render_gids+render_id, mask=render_mask)
#
#         # tl.device_print("", gid)
#         # # gaussian_mask = gid < num_gaussians & render_mask
#
#         ptr_render_means = tl.make_block_ptr(
#             base=ptr_means,
#             shape=(num_gaussians, 2),
#             strides=(2, 1),
#             offsets=(gid, 0),
#             block_shape=(1, 2),
#             order=(0, 1)
#         )
#
#         # ptr_render_invcov = tl.make_block_ptr(
#         #     base=ptr_invcovs + gid*4,
#         #     shape=(2, 2),
#         #     strides=(2, 1),
#         #     offsets=(0, 0),
#         #     block_shape=(2, 1),
#         #     order=(0, 1)
#         # )
#
#         # ptr_render_colors = tl.make_block_ptr(
#         #     base=ptr_colors + gid*3,
#         #     shape=(3),
#         #     strides=(1),
#         #     offsets=(0),
#         #     block_shape=(1),
#         #     order=(0)
#         # )
#
#         mean = tl.broadcast(tl.load(ptr_render_means), xy)#[None, None, :]
#         # tl.device_print("", xy, mean)
#         opac = tl.load(ptr_opacs + gid)
#         # opac = tl.load(ptr_opacs + gid, mask=render_mask)
#
#         d = xy - mean
#         # dt = tl.trans(d)
#         invcov_col1 = tl.load(ptr_invcovs+gid*4+ tl.arange(0,2)[:, None]*2)#tl.load(tl.advance(ptr_render_invcov, offsets=(0, 0)))
#         invcov_col2 = tl.load(ptr_invcovs+gid*4+ tl.arange(0,2)[:, None]*2 +1) #tl.load(tl.advance(ptr_render_invcov, offsets=(1, 0)))
#         # tl.device_print("", invcov_col2)
#         v1 = tl.sum(d * tl.trans(invcov_col1), axis=1, )
#         v2 = tl.sum(d * tl.trans(invcov_col2), axis=1, )
#         # v1 = tl.view(tl.sum(tl.trans(d) * invcov_col1, axis=1), (tile_h*tile_w, ))
#         # v2 = tl.view(tl.sum(tl.trans(d) * invcov_col2, axis=1), (tile_h*tile_w, 2))
#         # tl.device_print("",xy)
#         v = tl.view(tl.cat(v1, v2, can_reorder=True), (tile_h*tile_w, 2))
#         alpha = opac*tl.view(tl.exp(tl.sum(v*d, axis=1,)*-0.5), (tile_h*tile_w, 1))
#         # if (pid_tile_x != 10) or (pid_tile_y != 10):
#
#
#
#
#         for i in range(3):
#             delta_c = tl.load(ptr_colors+gid*3+i)
#
#             delta_c = tl.view((delta_c*alpha*T), (tile_h*tile_w,))
#             # tl.device_print("", color)
#             store_offsets = batch_offset + y*image_w + x + i*image_size
#             # if render_mask:
#             #     tl.device_print("", x, y)
#             # tl.store(ptr_images+store_offsets, color)
#             store_masks = (y < image_h) & (x < image_w) & render_mask
#             tl.atomic_add(ptr_images+store_offsets, delta_c, store_masks)
#         T *= 1 - alpha
#
#         # tl.device_print("", tl.tensor(means.shape, type=tl.int32))
#
#
#
#         # alpha =
#
#
#         #
#         # # # local_patch = tl.zeros(tile_h, tile_w, 3)
#         # for dy in tl.static_range(tile_h):
#         #     for dx in tl.static_range(tile_w):
#         #         y = pid_tile_y * tile_h + dy
#         #         x = pid_tile_x * tile_w + dx
#                 # yx = tl.cat(y, x, can_reorder=True)
#                 # d = means
#                 # s = tl.dot(tl.dot(d, invcov), d)
#                 # eval gaussian dist
#                 # s = -0.5*(inv_cov[0]*(x-mean[0])**2 + 2*inv_cov[2]*(x-mean[0])*(y-mean[1]) + inv_cov[3]*(y-mean[1])**2)
#                 # alpha = opac#*tl.exp(s)
#
#
#                 # color = color *  * alpha
#
#                 # color = tl.view(color, (3,))
#                 # store_offsets = tl.arange(0, 3)*(image_size) + batch_offset + y*image_w + x
#                 # store_mask = store_offsets < total_nums
#
#                 # tl.store(ptr_images+store_offsets, color, store_mask)
#                 #
#                 # T[dy, dx] *= (1-alpha)
#

# @triton.jit
# def tile_render(
#         num_tiles,
#         num_renders,
#         num_gaussians,
#         ptr_render_ranges,
#         ptr_render_keys,
#         ptr_render_gids,
#         ptr_means,
#         ptr_invcovs,
#         ptr_colors,
#         ptr_opacs,
#         ptr_images,
#         tile_h,
#         tile_w,
#         tile_num_h,
#         tile_num_w,
#         image_h,
#         image_w
# ):
#     pid = tl.program_id(axis=0)
#     tile_area = tile_w*tile_h
#     tile_id = pid // tile_area
#     sub_tile_id_y = pid % tile_area // tile_w
#     sub_tile_id_x = pid % tile_area % tile_w
#     mask = tile_id < num_tiles
#
#     render_key = tl.load(ptr_render_keys+tile_id, mask=mask)
#     render_range  = tl.load(ptr_render_ranges+tile_id, mask=mask)
#     render_start, render_end = render_range[0], render_range[1]
#
#     bid = render_key // (tile_num_h*tile_num_w)
#     pix_y = render_key % (tile_num_h*tile_num_w) // tile_num_w * tile_h + sub_tile_id_y
#     pix_x = render_key % (tile_num_h*tile_num_w) % tile_num_w * tile_w+ sub_tile_id_x
#
#     store_id = bid * (image_h*image_w) + pix_y*image_w + pix_x
#     T = 1
#     for render_id in range(render_start, render_end):
#         render_mask = render_id < num_renders & mask
#         gid = tl.load(ptr_render_gids+render_id, mask=render_mask)
#         gaussian_mask = gid < num_gaussians & render_mask
#         mean = tl.load(ptr_means+gid, gaussian_mask)
#         inv_cov = tl.load(ptr_invcovs+gid, gaussian_mask)
#         color = tl.load(ptr_colors+gid, gaussian_mask)
#         opac = tl.load(ptr_opacs+gid, gaussian_mask)
#
#
#         # eval gaussian dist
#         s = -0.5*(inv_cov[0]*(pix_x-mean[0])**2 + 2*inv_cov[2]*(pix_x-mean[0])*(pix_y-mean[1]) + inv_cov[3]*(pix_y-mean[1])**2)
#         alpha = opac*tl.exp(s)
#         color = color * T * alpha
#         tl.store(ptr_images+store_id*3, color)
#         T *= (1-alpha)




# def preprocess_tl(BLOCK_SIZE_X: tl.constexpr,
#                BLOCK_SIZE_Y: tl.constexpr,
#                num_gaussians,
#                num_per_thread,
#                ptr_means,
#                ptr_sigmas,
#                ptr_rhos,
#                ):
#     pid = tl.program_id(axis=0)
#     start_id = pid*num_per_thread
#     end_id = pid*num_per_thread
#
#     offsets = start_id + tl.arange(0, num_per_thread)
#     mask = offsets < num_gaussians
#     means = tl.load(ptr_means+offsets, mask=mask)
#     sigmas = tl.load(ptr_sigmas+offsets, mask=mask)
#     rhos = tl.load(ptr_rhos+offsets, mask=mask)
#
#     # compute min rect and area
#     ...
#     tl.cumsum()
#
#
# def _gs2d_forward():
#     ...