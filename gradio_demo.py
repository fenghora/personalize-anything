import os
import gradio as gr
from PIL import Image
import numpy as np
from src.utils import *
from scripts.grounding_sam import *
from diffusers.models.attention_processor import FluxAttnProcessor2_0
from src.attn_processor import (
    PersonalizeAnythingAttnProcessor,
    set_flux_transformer_attn_processor,
)
from src.pipeline import RFInversionParallelFluxPipeline

OUTPAINTING_EXAMPLES = [
    [
        "example_data/robot/background.png",
        "A close-up shot features a robot sitting on a chair on the right side of the frame. On the left side, there is a computer that the robot is using for work, set against a relatively sparse background.",
        600,
        600,
        1024,
        1024,
        350,
        50,
        60,
        28,
    ],
    [
        "example_data/cyberpunk/subject.png",
        "A stylish girl in a cyberpunk aesthetic, set against a vibrant futuristic city skyline.",
        768,
        768,
        1024,
        1024,
        128,
        0,
        50,
        28,
    ],
    [
        "example_data/knight/background.png",
        "A front view of a Western knight.",
        768,
        768,
        1024,
        1024,
        128,
        0,
        50,
        28,
    ],
    [
        "example_data/sandgirl/background.png",
        "A sand-sculpted girl at the seaside, positioned at the bottom of the scene, with plants growing in the sand on both the left and right sides of the image.",
        768,
        768,
        1024,
        1024,
        128,
        200,
        50,
        50,
    ],
    [
        "example_data/hutao/background.png",
        "A girl in a white hooded cloak, set against the background of a maple forest.",
        768,
        768,
        1024,
        1024,
        128,
        0,
        70,
        50,
    ],
]

INPAINTING_EXAMPLES = [
    [
        "example_data/robot/background.png",
        "A robot with a dog’s head stands in the corridor of a laboratory.",
    ],
    [
        "example_data/mountain/background.png",
        "A European-style castle sits atop a distant mountain.",
    ],
    [
        "example_data/boat/background.png",
        "A small white wooden boat sails through tumultuous waves.",
    ],
    [
        "example_data/kid/background.png",
        "A close-up photo of a little boy with a piece of bread at the corner of his mouth.",
    ],
    [
        "example_data/pucket/background.png",
        "A white bucket with the words 'Personalize Anything' written on it.",
    ],
]

FLUX_DEV = "black-forest-labs/FLUX.1-dev"

MAX_SEED = np.iinfo(np.int32).max

device = torch.device("cuda:0")
torch_type = torch.float16

pipe = RFInversionParallelFluxPipeline.from_pretrained(
    FLUX_DEV, torch_dtype=torch_type
).to(device)

os.environ["GRADIO_TEMP_DIR"] = "tmp"
os.environ["NO_PROXY"] = "localhost,127.0.0.1"

root_dir = "result"
save_dir = os.path.join(root_dir, "gradio")
os.makedirs(save_dir, exist_ok=True)


def predict_outpainting_mask(
    pil_image,
    r_x,
    r_y,
    resize_width,
    resize_height,
    full_width,
    full_height,
):
    background = Image.new("RGB", (full_width, full_height), (0, 0, 0))
    mask = Image.new("L", (full_width, full_height), 0)

    # resize
    resized = pil_image.resize((resize_width, resize_height), Image.LANCZOS)
    paste_x = max(r_x, 0)
    paste_y = max(r_y, 0)
    paste_max_x = min(r_x + resize_width, full_width)
    paste_max_y = min(r_y + resize_height, full_height)

    src_left = max(-r_x, 0)
    src_top = max(-r_y, 0)
    src_right = min(resize_width, full_width - r_x)
    src_bottom = min(resize_height, full_height - r_y)

    if (src_right > src_left) and (src_bottom > src_top):
        valid_region = resized.crop((src_left, src_top, src_right, src_bottom))
        background.paste(valid_region, (paste_x, paste_y))
        mask.paste(255, (paste_x, paste_y, paste_max_x, paste_max_y))

    background.save(os.path.join(save_dir, "background.png"))
    mask.save(os.path.join(save_dir, "mask.png"))

    return background


def predict_inpainting_mask(im, save_dir=save_dir):
    mask_img = os.path.join(save_dir, "mask.png")
    bg_img = os.path.join(save_dir, "background.png")
    save_array_as_png(im["background"], bg_img)
    convert_to_mask_inpainting(im["layers"][0], mask_img)


def generate_image(prompt, seed, timestep, tau, random_seed, height=None, width=None):

    if random_seed:
        seed = random.randint(0, MAX_SEED)

    init_image_path = os.path.join(save_dir, "background.png")
    init_image = Image.open(init_image_path).convert("RGB")

    if height is None or width is None:
        width, height = init_image.size

    latent_h = height // (pipe.vae_scale_factor * 2)
    latent_w = width // (pipe.vae_scale_factor * 2)
    img_dims = latent_h * latent_w
    height, width = latent_h * (pipe.vae_scale_factor * 2), latent_w * (
        pipe.vae_scale_factor * 2
    )
    init_image = init_image.resize((width, height))

    generator = torch.Generator(device=device).manual_seed(seed)

    mask_path = os.path.join(save_dir, "mask.png")
    mask = create_mask(mask_path, latent_w, latent_h)

    inverted_latents, image_latents, latent_image_ids = pipe.invert(
        source_prompt="",
        image=init_image,
        height=height,
        width=width,
        num_inversion_steps=timestep,
        gamma=1.0,
    )

    set_flux_transformer_attn_processor(
        pipe.transformer,
        set_attn_proc_func=lambda name, dh, nh, ap: PersonalizeAnythingAttnProcessor(
            name=name, tau=tau / 100, mask=mask, device=device, img_dims=img_dims
        ),
    )

    image = pipe(
        ["", prompt],
        inverted_latents=inverted_latents,
        image_latents=image_latents,
        latent_image_ids=latent_image_ids,
        height=height,
        width=width,
        start_timestep=0.0,
        stop_timestep=0.99,
        num_inference_steps=timestep,
        eta=1.0,
        generator=generator,
    ).images[-1]

    set_flux_transformer_attn_processor(
        pipe.transformer,
        set_attn_proc_func=lambda name, dh, nh, ap: FluxAttnProcessor2_0(),
    )

    return image


with gr.Blocks() as demo:
    with gr.Tab("Inpainting"):
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    im_inpainting = gr.Sketchpad(
                        label="Origin image", sources="upload", type="numpy"
                    )
                with gr.Row():
                    tb_inpainting = gr.Textbox(label="Prompt (Optional)", lines=1)
                with gr.Row():
                    tau_inpainting = gr.Slider(
                        minimum=0,
                        maximum=100,
                        value=50,
                        label="Tau",
                        info="Smaller tau enhances local consistency at the cost of overall harmony.",
                    )
                with gr.Row():
                    seed_inpainting = gr.Slider(
                        label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=18
                    )
                    random_seed_inpainting = gr.Checkbox(
                        label="Random seed", value=False
                    )
                    timestep_inpainting = gr.Slider(
                        label="Number of inference steps",
                        minimum=1,
                        maximum=50,
                        step=1,
                        value=28,
                    )

                button_inpainting = gr.Button(value="Run Generation", variant="primary")

            with gr.Column():
                result_im_inpainting = gr.Image(label="Generated image")

        examples = gr.Examples(
            INPAINTING_EXAMPLES,
            inputs=[im_inpainting, tb_inpainting],
            cache_examples=False,
        )

    with gr.Tab("Outpainting"):
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    im_outpainting = gr.Image(type="pil", label="Input Image")
                    process_im_outpainting = gr.Image(type="pil", label="Process Image")
                with gr.Row():
                    resize_height = gr.Slider(
                        label="Resize height", minimum=512, maximum=1024, value=768
                    )
                    resize_width = gr.Slider(
                        label="Resize width", minimum=512, maximum=1024, value=768
                    )
                with gr.Row():
                    height_outpainting = gr.Slider(
                        label="Target height", minimum=512, maximum=1024, value=1024
                    )
                    width_outpainting = gr.Slider(
                        label="Target width", minimum=512, maximum=1024, value=1024
                    )
                with gr.Row():
                    r_x = gr.Slider(
                        label="The X coordinate of the top-left corner pasted on the canvas.",
                        minimum=0,
                        maximum=512,
                        value=128,
                        step=1,
                    )
                    r_y = gr.Slider(
                        label="The Y coordinate of the top-left corner pasted on the canvas.",
                        minimum=0,
                        maximum=512,
                        value=0,
                        step=1,
                    )
                with gr.Row():
                    preview_button = gr.Button("Preview alignment and mask")

                with gr.Row():
                    tb_outpainting = gr.Textbox(label="Prompt (Optional)", lines=1)
                with gr.Row():
                    tau = gr.Slider(
                        minimum=0,
                        maximum=100,
                        value=50,
                        label="Tau",
                        info="Smaller tau enhances local consistency at the cost of overall harmony.",
                    )
                with gr.Row():
                    seed = gr.Slider(
                        label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=18
                    )
                    random_seed = gr.Checkbox(label="Random seed", value=False)
                    timestep = gr.Slider(
                        label="Number of inference steps",
                        minimum=1,
                        maximum=50,
                        step=1,
                        value=28,
                    )

                button_outpainting = gr.Button(
                    value="Run Generation", variant="primary"
                )

            with gr.Column():
                result_im_outpainting = gr.Image(label="Generated image")

        examples = gr.Examples(
            OUTPAINTING_EXAMPLES,
            inputs=[
                im_outpainting,
                tb_outpainting,
                resize_height,
                resize_width,
                height_outpainting,
                width_outpainting,
                r_x,
                r_y,
                tau,
                timestep,
            ],
            cache_examples=False,
        )

    # inpainting
    im_inpainting.change(predict_inpainting_mask, inputs=[im_inpainting])
    button_inpainting.click(
        generate_image,
        inputs=[
            tb_inpainting,
            seed_inpainting,
            timestep_inpainting,
            tau_inpainting,
            random_seed_inpainting,
        ],
        outputs=result_im_inpainting,
    )

    # outpainting
    preview_button.click(
        predict_outpainting_mask,
        inputs=[
            im_outpainting,
            r_x,
            r_y,
            resize_width,
            resize_height,
            width_outpainting,
            height_outpainting,
        ],
        outputs=process_im_outpainting,
    )
    button_outpainting.click(
        predict_outpainting_mask,
        inputs=[
            im_outpainting,
            r_x,
            r_y,
            resize_width,
            resize_height,
            width_outpainting,
            height_outpainting,
        ],
        outputs=process_im_outpainting,
    ).then(
        generate_image,
        inputs=[
            tb_outpainting,
            seed,
            timestep,
            tau,
            random_seed,
            height_outpainting,
            width_outpainting,
        ],
        outputs=result_im_outpainting,
    )

if __name__ == "__main__":

    demo.launch()
