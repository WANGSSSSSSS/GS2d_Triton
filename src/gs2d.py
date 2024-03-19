from typing import Any

import triton
import torch
import torch.nn as nn
import time

from src.gs2d_forward import preprocess
from src.gs2d_forward import duplicate_renders
from src.gs2d_forward import find_ranges
from src.gs2d_forward import render

from src.gs2d_backward import render_grad

class Gaussian2dRenderFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, batch_ids, means, inv_covs, colors, opacs, rects, cumsum_areas, render_cfg: dict) -> Any:
        t = time.time()
        device = batch_ids.device
        ctx.render_cfg = render_cfg
        ctx.GAUSSIANS_PER_THREAD = render_cfg["GAUSSIANS_PER_THREAD"]
        ctx.RENDERS_PER_THREAD = render_cfg["RENDERS_PER_THREAD"]
        ctx.TILE_H = render_cfg["FORWARD_TILE_H"]
        ctx.TILE_W = render_cfg["FORWARD_TILE_W"]
        ctx.IMAGE_H = render_cfg["IMAGE_H"]
        ctx.IMAGE_W = render_cfg["IMAGE_W"]

        ctx.TILE_NUM_H = (ctx.IMAGE_H + ctx.TILE_H -1) // ctx.TILE_H
        ctx.TILE_NUM_W = (ctx.IMAGE_W + ctx.TILE_W -1) // ctx.TILE_W
        ctx.BATCH_SIZE = batch_ids.max().item()+1
        ctx.GAUSSIANS_NUM = batch_ids.numel()
        ctx.RENDERS_NUM = cumsum_areas[-1].item()
        ctx.DTYPE = render_cfg["DTYPE"]
        # allocate render keys pairs for sorting
        render_keys = torch.zeros(ctx.RENDERS_NUM, dtype=torch.int32, device=device)-1   # bid, tid, depth
        render_values = torch.zeros(ctx.RENDERS_NUM, dtype=torch.int32, device=device)-1 # gaussian ids
        render_ranges = torch.full((ctx.BATCH_SIZE, ctx.TILE_NUM_H, ctx.TILE_NUM_W, 2), -1, dtype=torch.int32, device=device)
        render_ranges[0, 0, 0, 0] = 0
        # render_ranges[-1, -1, -1, -1] = renders_num

        t1 = time.time()
        # launch triton duplicate func
        grid = (triton.cdiv(ctx.GAUSSIANS_NUM,  ctx.GAUSSIANS_PER_THREAD), )
        duplicate_renders[grid](ctx.GAUSSIANS_NUM,
                                ctx.GAUSSIANS_PER_THREAD,
                                ctx.TILE_NUM_H,
                                ctx.TILE_NUM_W,
                                ctx.RENDERS_NUM,
                                batch_ids,
                                cumsum_areas,
                                rects,
                                render_keys,
                                render_values,
                                )
        print("duplicate time:", time.time()-t1)
        sorted_keys, sorted_inds = torch.sort(render_keys, stable=True)

        # tips, reuse memory buffer and this faster
        sorted_values = torch.index_select(render_values, dim=0, index=sorted_inds, out=render_keys) #render_values[sorted_inds]

        # launch find ranges func
        grid = (triton.cdiv(ctx.RENDERS_NUM, ctx.RENDERS_PER_THREAD), )
        find_ranges[grid](ctx.RENDERS_NUM, ctx.RENDERS_PER_THREAD, sorted_keys, render_ranges)

        # allocate image space
        images = torch.zeros((ctx.BATCH_SIZE, 4, ctx.IMAGE_H, ctx.IMAGE_W), dtype=ctx.DTYPE, device=device)
        occs = torch.zeros((ctx.BATCH_SIZE, ctx.IMAGE_H, ctx.IMAGE_W), dtype=ctx.DTYPE, device=device)
        # launch render func
        grid = (ctx.BATCH_SIZE, ctx.TILE_NUM_H, ctx.TILE_NUM_W)
        render[grid](
            renders_num=ctx.RENDERS_NUM,
            gaussians_num=ctx.GAUSSIANS_NUM,
            tile_h=ctx.TILE_H,
            tile_w=ctx.TILE_W,
            tile_h_nums=ctx.TILE_NUM_H,
            tile_w_nums=ctx.TILE_NUM_W,
            image_h=ctx.IMAGE_H,
            image_w=ctx.IMAGE_W,
            ptr_render_ranges=render_ranges,
            ptr_render_gids=sorted_values,
            ptr_means=means,
            ptr_invcovs=inv_covs,
            ptr_colors=colors,
            ptr_opacs=opacs,
            ptr_images=images,
            ptr_occs = occs,
        )


        ctx.save_for_backward(
            means, inv_covs, colors, opacs, occs,
            render_ranges,
            sorted_keys,
            sorted_values,
        )
        del render_keys, cumsum_areas
        print("forward time:", time.time()-t)
        return images

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        (   means, inv_covs, colors, opacs, occs,
            render_ranges,
            sorted_keys,
            sorted_values,
        ) = ctx.saved_tensors
        t = time.time()
        ctx.TILE_H = ctx.render_cfg["BACKWARD_TILE_H"]
        ctx.TILE_W = ctx.render_cfg["BACKWARD_TILE_W"]
        ctx.TILE_NUM_H = (ctx.IMAGE_H + ctx.TILE_H -1) // ctx.TILE_H
        ctx.TILE_NUM_W = (ctx.IMAGE_W + ctx.TILE_W -1) // ctx.TILE_W

        # render_ranges[..., 0] = render_ranges[..., 0] * ctx.render_cfg["FORWARD_TILE_H"] // ctx.TILE_H
        # render_ranges[..., 1] = render_ranges[..., 1] * ctx.render_cfg["FORWARD_TILE_W"] // ctx.TILE_W

        batch_ids_grad = None
        means_grad = torch.zeros_like(means)
        inv_covs_grad = torch.zeros_like(inv_covs)
        colors_grad = torch.zeros_like(colors)
        opacs_grad = torch.zeros_like(opacs)

        grid = (ctx.BATCH_SIZE, ctx.TILE_NUM_H, ctx.TILE_NUM_W)
        render_grad[grid](
            renders_num=ctx.RENDERS_NUM,
            gaussians_num=ctx.GAUSSIANS_NUM,
            tile_h=ctx.TILE_H,
            tile_w=ctx.TILE_W,
            tile_h_nums=ctx.TILE_NUM_H,
            tile_w_nums=ctx.TILE_NUM_W,
            image_h=ctx.IMAGE_H,
            image_w=ctx.IMAGE_W,
            ptr_render_ranges=render_ranges,
            ptr_render_gids=sorted_values,
            ptr_images_grad=grad_outputs[0],
            ptr_occs=occs,
            ptr_means=means,
            ptr_invcovs=inv_covs,
            ptr_colors=colors,
            ptr_opacs=opacs,
            ptr_means_grad=means_grad,
            ptr_invcovs_grad=inv_covs_grad,
            ptr_colors_grad=colors_grad,
            ptr_opacs_grad=opacs_grad,
        )
        print("backward time:", time.time()-t)
        # print(grid)
        # print("color_grad:", colors_grad/ctx.TILE_W/ctx.TILE_H)
        # print("opacs_grad:", opacs_grad)
        return batch_ids_grad, means_grad, inv_covs_grad, colors_grad, opacs_grad, None, None, None


class Gaussian2dRender(nn.Module):
    render_fn = Gaussian2dRenderFunction.apply
    render_fn_class = Gaussian2dRenderFunction
    def __init__(self, gaussians_per_thread, renders_per_thread, forward_tile, backward_tile, image_h, image_w, render_dtype):
        super().__init__()
        self.image_h = image_h
        self.image_w = image_w
        self.gaussians_per_thread = gaussians_per_thread
        self.renders_per_thread = renders_per_thread
        self.render_dtype = render_dtype

        # init render_fn
        self.render_cfg = dict(
            GAUSSIANS_PER_THREAD = gaussians_per_thread,
            RENDERS_PER_THREAD = renders_per_thread,
            IMAGE_H = image_h,
            IMAGE_W = image_w,
            FORWARD_TILE_H = forward_tile,
            FORWARD_TILE_W = forward_tile,
            BACKWARD_TILE_H = backward_tile,
            BACKWARD_TILE_W = backward_tile,
            DTYPE = render_dtype
        )

        self.preprocess_tile_h = forward_tile
        self.preprocess_tile_w = forward_tile
        self.preprocess_tile_num_h = (image_h + self.preprocess_tile_h -1) // self.preprocess_tile_h
        self.preprocess_tile_num_w = (image_w + self.preprocess_tile_w -1) // self.preprocess_tile_w


    def forward(self, batch_ids, means, sigmas, rhos, colors, opacs):
        # cast to low memory format
        means = means.to(self.render_dtype)
        sigmas = sigmas.to(self.render_dtype)
        rhos = rhos.to(self.render_dtype)
        colors = colors.to(self.render_dtype)
        opacs = opacs.to(self.render_dtype)

        sigmas = sigmas #+ 1e-6 # tips: numerical stable
        cos_rhos = torch.cos(rhos)
        sin_rhos = torch.sin(rhos)
        zeros = torch.zeros_like(sin_rhos)

        R = torch.cat([cos_rhos, -sin_rhos, sin_rhos, cos_rhos], dim=-1).view(-1, 2, 2)
        RT = torch.cat([cos_rhos, sin_rhos, -sin_rhos, cos_rhos], dim=-1).view(-1, 2, 2)
        invds = torch.cat([1/sigmas[:, :1], zeros, zeros, 1/sigmas[:, 1:]], dim=-1).view(-1,2,2)
        inv_covs = torch.bmm(torch.bmm(R, invds), RT)

        # no grad in preprocess
        rects, cumsum_areas = preprocess(means, sigmas, rhos,
                                         self.preprocess_tile_h,
                                         self.preprocess_tile_w,
                                         self.preprocess_tile_num_h,
                                         self.preprocess_tile_num_w)
        # print("area:", cumsum_areas)
        images = self.render_fn(batch_ids, means, inv_covs, colors, opacs, rects, cumsum_areas, self.render_cfg)
        return images


