import cv2
import numpy as np
import torch
import copy
import lpips
from diffusers import AutoPipelineForText2Image
from src.attn_forward_fn import AttnProcessor2_0 as CustomAttnProcessor2_0
from diffusers.models.attention_processor import AttnProcessor2_0
import tomesd

# Initialize LPIPS loss function
loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()

def replace_attn_processors_in_unet(unet, split_factor=1):
    """
    Replace all instances of AttnProcessor2_0 in the deeply nested UNet with the custom implementation.
    """
    for name, module in unet.named_modules():
        if hasattr(module, "attn1") and isinstance(module.attn1.processor, (AttnProcessor2_0, CustomAttnProcessor2_0)):
            module.attn1.processor = CustomAttnProcessor2_0(split_factor=split_factor)
        if hasattr(module, "attn2") and isinstance(module.attn2.processor, (AttnProcessor2_0, CustomAttnProcessor2_0)):
            module.attn2.processor = CustomAttnProcessor2_0(split_factor=split_factor)

# Set seed for reproducibility
torch.manual_seed(111)

# Define parameters
prompts = ["a cat in the style of Van Gogh's Starry Night painting"]
        # "A medieval castle on a misty hilltop at dawn.",
        # "A futuristic robotic arm in a neon-lit room.",
        # "A serene mountain lake with a small wooden cabin in the distance.",
        # "A serene Japanese garden during autumn, with vibrant red maple leaves and a tranquil koi pond, rendered in the style of Studio Ghibli animation",
        # "A cosmic whale swimming through a colorful nebula, its body glowing with bioluminescent patterns, depicted as a surreal dreamscape",
        # "A neon-lit street in Tokyo, but all the buildings and signs are made entirely of holographic projections, rendered in ultra-realistic cyberpunk 3D.",
        # "koi fish in the clouds, fantasy storybook painting",
        # "An ancient library hidden deep within a misty forest, its towering bookshelves filled with glowing, magical tomes, illustrated in a whimsical fantasy style"  ,
        # "A futuristic city floating above the clouds, with waterfalls cascading from its edges and flying cars weaving through golden sunset skies, rendered in a cinematic sci-fi aesthetic",
        # "A colossal tree with bioluminescent leaves growing in the middle of an endless desert, its roots forming intricate patterns in the sand, depicted in a surreal dreamlike style",  
        # "A Viking warrior standing on the edge of a frozen fjord, gazing at the aurora borealis illuminating the night sky, painted in a semi-realistic historical fantasy style",
        # "A steampunk airship docked at a floating skyport, with intricate brass gears and steam billowing from its engines, designed in a highly detailed Victorian sci-fi aesthetic",  
        # "A peaceful cottage nestled in a valley of giant mushrooms, with glowing spores drifting through the air, illustrated in a soft and enchanting fairytale style",
        # "A cybernetic samurai standing under a cherry blossom tree, his armor reflecting neon lights from a futuristic cityscape in the distance, rendered in a blend of cyberpunk and traditional Japanese art",  
        # "An astronaut exploring an alien jungle filled with bioluminescent plants and strange floating creatures, depicted in a vibrant sci-fi concept art style",
        # "A gothic castle perched on the edge of a stormy cliff, with eerie green lights glowing from its windows, painted in a dark fantasy horror style",  
        # "A giant clockwork dragon soaring through the sky, its wings made of intricate golden gears and steam-powered pistons, illustrated in a highly detailed steampunk fantasy style"] 

for p_id, prompt in enumerate(prompts):
    split_factors = [1, 2, 3, 5]
    tome_ratio = 0.0 
    pipeline = AutoPipelineForText2Image.from_pretrained("stabilityai/sd-turbo", torch_dtype=torch.float16, variant="fp16")
    pipeline.to("cuda")

    pipeline = copy.deepcopy(pipeline)
    tomesd.apply_patch(pipeline, ratio=tome_ratio, max_downsample=3)

    images = []
    for split_factor in split_factors:
        replace_attn_processors_in_unet(pipeline.unet, split_factor=split_factor)
        pipeline.to("cuda")
        torch.manual_seed(11)
        image = pipeline(prompt=prompt, num_inference_steps=1, guidance_scale=0.0).images[0]
        images.append(np.array(image))

    lpips_scores = []
    for img, split in zip(images,split_factors):
        image_A = torch.tensor(img).permute(2, 0, 1).float() / 255.0
        image_B = torch.tensor(images[0]).permute(2, 0, 1).float() / 255.0
        lpips_score = loss_fn_vgg(image_A.cuda(), image_B.cuda()).item()
        lpips_scores.append(lpips_score)
        output_path = f"output_images/image_split_{split}_tome_{tome_ratio}.png"
        cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    stitched_image = np.hstack(images)
    for idx, (img, split_factor, lpips_score) in enumerate(zip(images, split_factors, lpips_scores)):
        ram_usage = 5.8 / (split_factor ** 2) * (1 - tome_ratio) ** 2
        text = f" Split: {split_factor} \n LPIPS: {lpips_score:.2f} \n RAM: {ram_usage:.2f}GB"
        lpips_text = f"LPIPS: {lpips_score:.2f}"
        ram_text = f"RAM: {ram_usage:.2f}GB"

        print(text)
        cv2.putText(stitched_image, lpips_text, (idx * img.shape[1] + 10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
        cv2.putText(stitched_image, ram_text, (idx * img.shape[1] + 10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)

    cv2.imshow("Generated Images", cv2.cvtColor(stitched_image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"output_images/final_prompt{p_id}_tome{tome_ratio}.png", cv2.cvtColor(stitched_image, cv2.COLOR_RGB2BGR))
    cv2.waitKey(1)
    # cv2.destroyAllWindows()