import torch
import torch.nn as nn
from src.gs2d import Gaussian2dRender

import time

if __name__ == "__main__":
    batch_size = 100
    device = "cuda"
    num_gaussians = 2000 * 100
    image_h, image_w = 256, 256
    batch_ids = torch.randint(0, batch_size, (num_gaussians,), device=device, dtype=torch.int8)

    parameters = nn.Parameter(torch.randn(num_gaussians, 10, device=device, dtype=torch.float32))

    # colors = torch.rand((num_gaussians, 4), device=device, dtype=torch.float16,
    #                     requires_grad=True)  # torch.full((num_gaussians, 4), 1.0, device=device)
    means = torch.rand((num_gaussians, 2), device=device, dtype=torch.float16,
                    requires_grad=True) * image_w  # torch.tensor((3.5, 3.5), device=device).view(-1, 2).repeat(num_gaussians, 1)
    #
    # sigmas = torch.rand((num_gaussians, 2), device=device, dtype=torch.float32,
    #                     requires_grad=True) * 10  # torch.full((num_gaussians, 2), 10, device=device)
    # rhos = torch.full((num_gaussians, 1), 0.0, device=device, dtype=torch.float16, requires_grad=True)
    # opacs = torch.full((num_gaussians, 1), .2, device=device, dtype=torch.float16, requires_grad=True)

    render_cfg = dict(
        gaussians_per_thread=1,
        renders_per_thread=1,
        forward_tile=4,
        backward_tile=4,
        image_h=image_h,
        image_w=image_w,
        render_dtype=torch.float16,
    )
    print(torch.cuda.max_memory_reserved()/1024**3, "GB")

    torch_targets = torch.zeros((batch_size, 4, image_h, image_w), requires_grad=False, device=device,
                                dtype=torch.float16)

    for j in range(image_w):
        for i in range(image_h):
            d = (i - 128) ** 2 + (j - 128) ** 2
            torch_targets[:, 0, i, j] = torch.exp(-0.5 * torch.tensor(d, device=device))

    sigmas = parameters[:, 2:4].sigmoid() * 10
    rhos = parameters[:, 4:5]
    opacs = parameters[:, 5:6].sigmoid()
    colors = parameters[:, 6:10].sigmoid()

    rs_time = time.time()
    with torch.no_grad():
        torch_images = Gaussian2dRender(**render_cfg)(batch_ids, means, sigmas, rhos, colors, opacs)
    print(torch.cuda.max_memory_allocated() / 1024**3, "GB")
    np_image = torch_images[0, :3].permute(1, 2, 0).detach().cpu().numpy()
    import matplotlib.pyplot as plt
    import numpy as np
    plt.imshow((np_image * 255).astype(np.uint8))
    plt.show()