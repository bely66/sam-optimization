import numpy as np
from torch.profiler import profile, record_function, ProfilerActivity
import torch
import matplotlib.pyplot as plt
import cv2
import sys
import os
sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 

def profiler_runner(path, fn, *args, **kwargs):
    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True) as prof:
        result = fn(*args, **kwargs)
    print(f"Saving trace under {path}")
    prof.export_chrome_trace(path)
    return result

image = cv2.imread('notebooks/images/truck.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



sam_checkpoint = "model/sam_vit_b_01ec64.pth"
model_type = "vit_b"
device = "cuda"
logs_dir = "exp_logs"
def run_dtype_exp(dtype=None, trace_name=None, logs_name=None, img_name=None, batch_size=1):
    trace_path = os.path.join(logs_dir, trace_name)
    logs_path = os.path.join(logs_dir, logs_name)
    img_path = os.path.join(logs_dir, img_name)
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint, dtype=dtype)
    sam.to(device=device)

    # print the weights dtype
    print("Weights dtype:", sam.image_encoder.state_dict()["patch_embed.proj.weight"].dtype)

    generator = SamAutomaticMaskGenerator(sam, points_per_batch=batch_size)


    # Run multiple times for warmup
    for _ in range(3):
        generator.generate(image)


    # Benchmark
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(10):
        masks =  generator.generate(image)

    end_event.record()
    torch.cuda.synchronize()
    avg_elapsed_time = start_event.elapsed_time(end_event) / 10.
    print("Average Elapsed Time Per Picture", avg_elapsed_time)


    # prof.export_chrome_trace(trace_path)

    profiler_runner(trace_path, generator.generate, image)

    # Write out memory usage
    max_memory_allocated_bytes = torch.cuda.max_memory_allocated()
    _, total_memory = torch.cuda.mem_get_info()
    max_memory_allocated_percentage = int(100 * (max_memory_allocated_bytes / total_memory))
    max_memory_allocated_bytes = max_memory_allocated_bytes >> 20

    with open(logs_path, "w") as f:
        f.write("Average Elapsed Time Per Picture: {}\n".format(avg_elapsed_time))
        f.write("Average FPS: {}\n".format(1000 / avg_elapsed_time))
        f.write("Memory(MiB): {}\n".format(max_memory_allocated_bytes))
        f.write("Memory(%): {}\n".format(max_memory_allocated_percentage))

        plt.figure(figsize=(10,10))
        plt.imshow(image)
        show_anns(masks)
        plt.axis('off')
        plt.show() 
        plt.savefig(img_path, bbox_inches='tight', pad_inches=0)

if __name__ == "__main__":
    # run_dtype_exp(dtype=torch.float16, trace_name="trace_fp16.json", logs_name="logs_fp16.txt", img_name="img_fp16.jpg", batch_size=32)
    # run_dtype_exp(dtype=None, trace_name="trace_fp32.json", logs_name="logs_fp32.txt", img_name="img_fp32.jpg", batch_size=32)
    run_dtype_exp(dtype=torch.bfloat16, trace_name="trace_bf16.json", logs_name="logs_bf16.txt", img_name="img_bf16.jpg", batch_size=32)








