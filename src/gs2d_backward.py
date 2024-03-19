import triton
import triton.language as tl


@triton.jit
def render_grad(
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
        ptr_images_grad,
        ptr_occs,
        ptr_means,
        ptr_invcovs,
        ptr_colors,
        ptr_opacs,
        ptr_means_grad,
        ptr_invcovs_grad,
        ptr_colors_grad,
        ptr_opacs_grad,

):
    # atom_mem not work
    pid_b = tl.program_id(axis=0)
    pid_tile_y = tl.program_id(axis=1)
    pid_tile_x = tl.program_id(axis=2)
    image_size = image_h*image_w
    batch_offset = pid_b*(image_size)
    # pid = pid_b + pid_tile_x + pid_tile_y
    offset = pid_b*(tile_w_nums*tile_h_nums) + pid_tile_y*tile_w_nums + pid_tile_x
    start_render = tl.load(ptr_render_ranges + offset*2)
    end_render = tl.load(ptr_render_ranges + offset*2+1)

    y = tl.broadcast_to(tl.arange(0, tile_h)[:, None], (tile_h, tile_w)) + pid_tile_y * tile_h
    x = tl.broadcast_to(tl.arange(0, tile_w)[None, :], (tile_h, tile_w)) + pid_tile_x * tile_w
    y = tl.view(y, (tile_h*tile_w, 1))
    x = tl.view(x, (tile_h*tile_w, 1))

    # current triton not support cat/interleave/stack and so on
    # so i can not get xy in a tensor, so stupid
    # maybe load a predefined meshgrid tensor??

    grad_offsets =  batch_offset*4 + y*image_w + x + tl.arange(0, 4)[None, :]*image_size
    grad_masks = (y < image_h) & (x < image_w)
    grad_tile = tl.load(ptr_images_grad+grad_offsets, mask=grad_masks)

    occs_offsets = batch_offset + y*image_w + x
    occs_masks = (y < image_h) & (x < image_w)
    occs_tile = tl.load(ptr_occs+occs_offsets, mask=occs_masks)

    backacc = tl.zeros_like(grad_tile)

    for render_id in range(end_render-1, start_render-1, -1):

        render_mask = (render_id < renders_num) & (render_id >= 0)
        gid = tl.load(ptr_render_gids+render_id, mask=render_mask)
        mx = tl.load(ptr_means+gid*2)
        my = tl.load(ptr_means+gid*2+1)

        d1 = x - mx + 0.5
        d2 = y - my + 0.5

        ic1 = tl.load(ptr_invcovs+gid*4)
        ic2 = tl.load(ptr_invcovs+gid*4+1)
        ic3 = tl.load(ptr_invcovs+gid*4+2)
        ic4 = tl.load(ptr_invcovs+gid*4+3)

        s = ic1*d1*d1 + d1*d2*(ic2+ic3) + ic4*d2*d2

        opac = tl.load(ptr_opacs + gid)
        exp_s = (tl.constexpr(-0.5)*s)
        alpha = (opac*exp_s).to(ptr_occs.type.element_ty)
        inv_alpha = (tl.constexpr(1.0)/(alpha+0.0001)).to(ptr_occs.type.element_ty)
        color = tl.load(ptr_colors + gid*4 + tl.arange(0, 4)[None, :])

        if render_mask:
            occs_tile*=inv_alpha


        # color_grad 1x4
        color_grad = tl.sum(grad_tile*alpha*occs_tile, axis=0)
        tl.atomic_add(ptr_colors_grad + gid*4 + tl.arange(0, 4), color_grad, render_mask)

        # alpha_grad Nx1
        alpha_grad = tl.sum(grad_tile*occs_tile*(color - backacc), axis=1)[:, None]

        # opac_grad 1
        opac_grad = tl.sum(exp_s* alpha_grad)
        tl.atomic_add(ptr_opacs_grad + gid, opac_grad, render_mask)
        # tl.store(ptr_opacs_grad + gid, opac_grad, render_mask)

        # s_grad  Nx1
        s_grad = alpha_grad*opac*exp_s*(-0.5)

        # mean_grad 2
        # invcov22 = tl.view(invcov, (1, 2, 2))
        # mean_grad = -tl.sum(s_grad*tl.sum(invcov22*dt, axis=2), axis=0)


        mx_grad = -tl.sum(s_grad*(2*ic1*d1 + (ic2+ic3)*d2))
        my_grad = -tl.sum(s_grad*(2*ic4*d1 + (ic2+ic3)*d1))
        # mean_grad = tl.cat(mx_grad, my_grad)
        # tl.atomic_add(ptr_means_grad+gid*2+tl.arange(0,2), mean_grad, render_mask)
        tl.atomic_add(ptr_means_grad+gid*2, mx_grad, render_mask)
        tl.atomic_add(ptr_means_grad+gid*2+1, my_grad, render_mask)

        # invcov_grad 4
        # invcov_grad = tl.sum(s_grad*dtd, axis=0)
        ic1_grad = tl.sum(s_grad*d1*d1)
        ic2_grad = tl.sum(s_grad*d1*d2)
        ic3_grad = tl.sum(s_grad*d1*d2)
        ic4_grad = tl.sum(s_grad*d2*d2)
        tl.atomic_add(ptr_invcovs_grad+gid*4, ic1_grad, render_mask)
        tl.atomic_add(ptr_invcovs_grad+gid*4+1, ic2_grad, render_mask)
        tl.atomic_add(ptr_invcovs_grad+gid*4+1, ic3_grad, render_mask)
        tl.atomic_add(ptr_invcovs_grad+gid*4+1, ic4_grad, render_mask)


        # maybe



        # sigma and rho grad
        # not this place
        # rho = tl.load(ptr_rhos)
        # sigma = tl.load(ptr_sigmas + gid*2 + tl.arange(0, 2))[None, :]
        # sigma_grad = tl.sum(invcov22_grad * sigma)
        if render_mask:
            backacc *= (1-alpha)
            backacc += alpha*color



import triton.language.semantic as tlsm

@triton.jit
def render_grad_dot(
        renders_num: tl.constexpr,
        gaussians_num: tl.constexpr,
        tile_h: tl.constexpr,
        tile_w: tl.constexpr,
        tile_h_nums: tl.constexpr,
        tile_w_nums: tl.constexpr,
        image_h: tl.constexpr,
        image_w: tl.constexpr,
        ptr_render_ranges,
        ptr_render_gids,
        ptr_images_grad,
        ptr_occs,
        ptr_means,
        ptr_invcovs,
        ptr_colors,
        ptr_opacs,
        ptr_means_grad,
        ptr_invcovs_grad,
        ptr_colors_grad,
        ptr_opacs_grad,

):
    # atom_mem not work
    pid_b = tl.program_id(axis=0)
    pid_tile_y = tl.program_id(axis=1)
    pid_tile_x = tl.program_id(axis=2)
    image_size = image_h*image_w
    batch_offset = pid_b*(image_size)
    # pid = pid_b + pid_tile_x + pid_tile_y
    offset = pid_b*(tile_w_nums*tile_h_nums) + pid_tile_y*tile_w_nums + pid_tile_x
    start_render = tl.load(ptr_render_ranges + offset*2)
    end_render = tl.load(ptr_render_ranges + offset*2+1)
    # tl.device_print("", start_render, end_render)

    y = tl.broadcast_to(tl.arange(0, tile_h)[:, None], (tile_h, tile_w)) + pid_tile_y * tile_h
    x = tl.broadcast_to(tl.arange(0, tile_w)[None, :], (tile_h, tile_w)) + pid_tile_x * tile_w

    y = tl.view(y, (tile_h*tile_w, 1))
    x = tl.view(x, (tile_h*tile_w, 1))

    xy = tl._experimental_interleave(x, y, )[:, None, :]

    grad_offsets =  batch_offset*4 + y*image_w + x + tl.arange(0, 4)[None, :]*image_size
    grad_masks = (y < image_h) & (x < image_w)
    grad_tile = tl.load(ptr_images_grad+grad_offsets, mask=grad_masks)

    occs_offsets = batch_offset + y*image_w + x
    occs_masks = (y < image_h) & (x < image_w)
    occs_tile = tl.load(ptr_occs+occs_offsets, mask=occs_masks)
    backacc = tl.zeros_like(grad_tile)

    for render_id in range(end_render-1, start_render-1, -1):

        render_mask = (render_id < renders_num) & (render_id >= 0)
        gid = tl.load(ptr_render_gids+render_id, mask=render_mask)
        mean = tl.load(ptr_means+gid*2+tl.arange(0,2)[None, None, :])# + tile_h/2
        d = xy - mean
        dt = tl.view(d, (tile_h*tile_w, 2, 1))
        dtd = tl.view(d*dt, (tile_h*tile_w, 4))
        invcov = tl.load(ptr_invcovs+gid*4+ tl.arange(0,4)[None, :])
        s = tl.sum(dtd*invcov, axis=1)[:, None]

        opac = tl.load(ptr_opacs + gid)
        exp_s = tl.exp(tl.constexpr(-0.5)*s)
        alpha = (opac*exp_s).to(ptr_occs.type.element_ty)
        inv_alpha = (tl.constexpr(1.0)/(alpha+0.0001)).to(ptr_occs.type.element_ty)
        color = tl.load(ptr_colors + gid*4 + tl.arange(0, 4)[None, :])

        if render_mask:
            occs_tile*=inv_alpha


        # color_grad 1x4
        color_grad = tl.sum(grad_tile*alpha*occs_tile, axis=0)
        tl.atomic_add(ptr_colors_grad + gid*4 + tl.arange(0, 4), color_grad, render_mask)

        # alpha_grad Nx1
        alpha_grad = tl.sum(grad_tile*occs_tile*(color - backacc), axis=1)[:, None]

        # opac_grad 1
        opac_grad = tl.sum(exp_s* alpha_grad)
        tl.atomic_add(ptr_opacs_grad + gid, opac_grad, render_mask)
        # tl.store(ptr_opacs_grad + gid, opac_grad, render_mask)

        # s_grad  Nx1
        s_grad = alpha_grad*opac*exp_s*(-0.5)

        # mean_grad 2
        invcov22 = tl.view(invcov, (1, 2, 2))
        mean_grad = -tl.sum(s_grad*tl.sum(invcov22*dt, axis=2), axis=0)
        tl.atomic_add(ptr_means_grad+gid*2+tl.arange(0,2), mean_grad, render_mask)


        # invcov_grad 4
        invcov_grad = tl.sum(s_grad*dtd, axis=0)
        tl.atomic_add(ptr_invcovs_grad+gid*4+tl.arange(0,4), invcov_grad, render_mask)

        # maybe


        backacc *= (1-alpha)
        backacc += alpha*color