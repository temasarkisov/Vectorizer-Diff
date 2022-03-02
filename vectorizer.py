import pydiffvg
import argparse
import ttools.modules
import torch
import os
import skimage.io
# import svgutils.transform as st
# from skimage.transform import rescale
import time
gamma = 1.0

def main(args):
    
    h, w = 128, 128
    canvas_width, canvas_height, shapes, shape_groups = \
        pydiffvg.svg_to_scene(args.svg)
    scene_args = pydiffvg.RenderFunction.serialize_scene(\
        canvas_width, canvas_height, shapes, shape_groups)
    image = skimage.io.imread(args.target)
    target = torch.from_numpy(image).to(torch.float16) / 255.
    
    target = target.pow(gamma)
    target = target.to(pydiffvg.get_device())
    target = target.unsqueeze(0)
    target = target.permute(0, 3, 1, 2) # NHWC -> NCHW
    
    render = pydiffvg.RenderFunction.apply
    img = render(canvas_width, # width
                 canvas_height, # height
                 2,   # num_samples_x
                 2,   # num_samples_y
                 0,   # seed
                 None, # bg
                 *scene_args)
    # The output image is in linear RGB space. Do Gamma correction before saving the image.
    # pydiffvg.imwrite(img.cpu(), 'results/refine_svg/init.png', gamma=gamma)
    # pydiffvg.save_svg('results/refine_svg/init.svg',
    #                           canvas_width, canvas_height, shapes, shape_groups)

    points_vars = []
    for path in shapes:
        path.points.requires_grad = True
        points_vars.append(path.points)
    color_vars = {}
    for group in shape_groups:
        group.fill_color.requires_grad = True
        color_vars[group.fill_color.data_ptr()] = group.fill_color
    color_vars = list(color_vars.values())

    # Optimize
    # currently initial lr is optimized, you may change it
    points_optim = torch.optim.Adam(points_vars, lr=args.points_lr)
    color_optim = torch.optim.Adam(color_vars, lr=args.color_lr)
    
    # Schedulers
    points_scheduler = torch.optim.swa_utils.SWALR(points_optim, anneal_strategy="linear", anneal_epochs=int(args.num_iter//1), swa_lr=0.1)
    color_scheduler = torch.optim.swa_utils.SWALR(color_optim, anneal_strategy="linear", anneal_epochs=args.num_iter, swa_lr=0.01)
    # Adam iterations.
    for t in range(args.num_iter):
        print('iteration:', t)
        points_optim.zero_grad()
        color_optim.zero_grad()
        # Forward pass: render the image.
        scene_args = pydiffvg.RenderFunction.serialize_scene(\
            canvas_width, canvas_height, shapes, shape_groups)
        img = render(canvas_width, # width
                     canvas_height, # height
                     2,   # num_samples_x
                     2,   # num_samples_y
                     0,   # seed
                     None, # bg
                     *scene_args)
        # Compose img with white background
        img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], 
                                                          img.shape[1], 3, 
                                                          device = pydiffvg.get_device()) * (1 - img[:, :, 3:4])
        
        # Save the intermediate render.
        # pydiffvg.imwrite(img.cpu(), 'results/refine_svg/iter_{}.png'.format(t), gamma=gamma)
        
        img = img[:, :, :3]
        # Convert img from HWC to NCHW
        img = img.unsqueeze(0)
        img = img.permute(0, 3, 1, 2) # NHWC -> NCHW
        loss = (img - target).pow(2).sum()
        print('render loss:', loss.item())
    
        # Backpropagate the gradients.
        loss.backward()
    
        # Take a gradient descent step.
        points_optim.step()
        color_optim.step()
        points_scheduler.step()
        color_scheduler.step()
        for group in shape_groups:
            group.fill_color.data.clamp_(0.0, 1.0)

        # if t % 10 == 0 or t == args.num_iter - 1:
        #     pydiffvg.save_svg('results/refine_svg/iter_{}.svg'.format(t),
        #                       canvas_width, canvas_height, shapes, shape_groups)

    # Render and write the final result.
    os.makedirs(args.save_folder, exist_ok=True)
    pydiffvg.save_svg(os.path.join(args.save_folder, f"iter_{t}.svg"),
                              canvas_width, canvas_height, shapes, shape_groups)

    scene_args = pydiffvg.RenderFunction.serialize_scene(\
        canvas_width, canvas_height, shapes, shape_groups)
    img = render(canvas_width, # width
                 canvas_height, # height
                 2,   # num_samples_x
                 2,   # num_samples_y
                 0,   # seed
                 None, # bg
                 *scene_args)
    # # Save the intermediate render.
    # pydiffvg.imwrite(img.cpu(), 'results/refine_svg/final.png'.format(t), gamma=gamma)
    # # Convert the intermediate renderings to a video.
    # from subprocess import call
    # call(["ffmpeg", "-framerate", "24", "-i",
    #     "results/refine_svg/iter_%d.png", "-vb", "20M",
    #     "results/refine_svg/out.mp4"])

if __name__ == "__main__":
    start = time.time()


    parser = argparse.ArgumentParser()
    parser.add_argument("svg", help="source SVG path")
    parser.add_argument("target", help="target image path")
    parser.add_argument("--num_iter", type=int, default=100)
    parser.add_argument("--save_folder", type=str, help="folder for SVG result", default='.')
    parser.add_argument("--points_lr", type=float, default=0.4)
    parser.add_argument("--color_lr", type=float, default=0.05)
    args = parser.parse_args()
    main(args)
    end = time.time() - start
    print(end)