# app.py
"""
🖼️→📝 Image-to-Text Attention Visualizer (Custom Model)
- Loads your custom model via create_complete_model()
- Accepts an image, applies your transform, then calls:
      model.generate(pixel_values=..., max_new_tokens=..., output_attentions=True)
- Selector lists ONLY generated words (no prompt tokens).
- Viewer (single row) shows:
    (1) original image,
    (2) original + colored attention heatmap overlay,
    (3) heatmap alone (colored).
- Heatmap is built from the first 1024 image tokens (32×32), then upscaled to the image size.
- Text block below shows word-level attention over generated tokens (no return_offsets_mapping used).
- Fixes deprecations: Matplotlib colormap API & Pillow mode inference.
"""

import os
import re
import random
from typing import List, Tuple, Optional

import gradio as gr
import torch
import numpy as np
from PIL import Image
from safetensors.torch import load_model 

# Optional: nicer colormap (Matplotlib >=3.7 API; no deprecation warnings)
try:
    import matplotlib as mpl
    _HAS_MPL = True
    _COLORMAP = mpl.colormaps.get_cmap("magma")
except Exception:
    _HAS_MPL = False
    _COLORMAP = None

# ========= Your utilities & model =========
from utils.processing import image_transform, pil_from_path
from utils.complete_model import create_complete_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = create_complete_model(device=DEVICE, attention_implementation="eager")
SAFETENSOR_PATH = "complete_model.safetensor"
try:
    load_model(model, SAFETENSOR_PATH)
except Exception as e:
    print(f"Error loading model: {e}, continuing with uninitialized weights.")
model.eval()
device = DEVICE

# --- Grab tokenizer from your model ---
tokenizer = getattr(model, "tokenizer", None)
if tokenizer is None:
    raise ValueError("Expected `model.tokenizer` to exist and be a HF-like tokenizer.")

# --- Fix PAD/EOS ambiguity (and resize embeddings if applicable) ---
needs_resize = False
pad_id = getattr(tokenizer, "pad_token_id", None)
eos_id = getattr(tokenizer, "eos_token_id", None)
if pad_id is None or (eos_id is not None and pad_id == eos_id):
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    needs_resize = True

# Try common resize hooks safely (only if your decoder actually uses tokenizer vocab)
if needs_resize:
    resize_fns = [
        getattr(getattr(model, "decoder", None), "resize_token_embeddings", None),
        getattr(model, "resize_token_embeddings", None),
    ]
    for fn in resize_fns:
        if callable(fn):
            try:
                fn(len(tokenizer))
                break
            except Exception:
                # If your model doesn't need resizing (separate vocab), it's fine.
                pass

# ========= Regex for words (words + punctuation) =========
WORD_RE = re.compile(r"\w+(?:'\w+)?|[^\w\s]")

# ========= Model metadata (for slider ranges) =========
def model_heads_layers():
    def _get(obj, *names, default=None):
        for n in names:
            if obj is None:
                return default
            if hasattr(obj, n):
                try:
                    return int(getattr(obj, n))
                except Exception:
                    return default
        return default

    cfg_candidates = [
        getattr(model, "config", None),
        getattr(getattr(model, "decoder", None), "config", None),
        getattr(getattr(model, "lm_head", None), "config", None),
    ]
    L = H = None
    for cfg in cfg_candidates:
        if L is None:
            L = _get(cfg, "num_hidden_layers", "n_layer", default=None)
        if H is None:
            H = _get(cfg, "num_attention_heads", "n_head", default=None)
    if L is None: L = 12
    if H is None: H = 12
    return max(1, L), max(1, H)

# ========= Attention utils =========
def get_attention_for_token_layer(
    attentions,
    token_index,
    layer_index,
    batch_index=0,
    head_index=0,
    mean_across_layers=True,
    mean_across_heads=True,
):
    """
    `attentions`:
      tuple length = #generated tokens
      attentions[t] -> tuple over layers; each layer tensor is (batch, heads, q, k)
    """
    token_attention = attentions[token_index]

    if mean_across_layers:
        layer_attention = torch.stack(token_attention).mean(dim=0)  # (batch, heads, q, k)
    else:
        layer_attention = token_attention[int(layer_index)]          # (batch, heads, q, k)

    batch_attention = layer_attention[int(batch_index)]              # (heads, q, k)

    if mean_across_heads:
        head_attention = batch_attention.mean(dim=0)                 # (q, k)
    else:
        head_attention = batch_attention[int(head_index)]            # (q, k)

    return head_attention.squeeze(0)  # q==1 -> (k,)

# ========= Tokens → words mapping (no offset_mapping needed) =========
def _words_and_map_from_tokens_simple(token_ids: List[int]) -> Tuple[List[str], List[int]]:
    """
    Works with slow/fast tokenizers. No return_offsets_mapping.
    Steps:
      1) detok token_ids
      2) regex-split words and get their char-end positions
      3) for each word-end (we), encode detok[:we] w/ add_special_tokens=False
         last token index = len(prefix_ids) - 1
    """
    if not token_ids:
        return [], []

    toks = tokenizer.convert_ids_to_tokens(token_ids)
    detok = tokenizer.convert_tokens_to_string(toks)

    matches = list(re.finditer(WORD_RE, detok))
    words = [m.group(0) for m in matches]
    ends = [m.span()[1] for m in matches]  # char end (exclusive)

    word2tok: List[int] = []
    for we in ends:
        prefix_ids = tokenizer.encode(detok[:we], add_special_tokens=False)
        if not prefix_ids:
            word2tok.append(0)
            continue
        last_idx = len(prefix_ids) - 1
        last_idx = max(0, min(last_idx, len(token_ids) - 1))
        word2tok.append(last_idx)

    return words, word2tok

def _strip_trailing_special(ids: List[int]) -> List[int]:
    specials = set(getattr(tokenizer, "all_special_ids", []) or [])
    j = len(ids)
    while j > 0 and ids[j - 1] in specials:
        j -= 1
    return ids[:j]

# ========= Visualization (word-level for generated text) =========
def generate_word_visualization_gen_only(
    words_gen: List[str],
    word_ends_rel: List[int],
    gen_attn_values: np.ndarray,
    selected_token_rel_idx: int,
) -> str:
    """
    words_gen: generated words only
    word_ends_rel: last-token indices of each generated word (relative to generation)
    gen_attn_values: length == len(gen_token_ids), attention over generated tokens only
                     (zeros for future tokens padded at the end)
    """
    if not words_gen or gen_attn_values is None or len(gen_attn_values) == 0:
        return (
            "<div style='width:100%;'>"
            "  <div style='background:#444;border:1px solid #eee;border-radius:8px;padding:10px;'>"
            "    <div style='color:#ddd;'>No text attention values.</div>"
            "  </div>"
            "</div>"
        )

    # compute word starts from ends (inclusive indexing)
    starts = []
    for i, end in enumerate(word_ends_rel):
        if i == 0:
            starts.append(0)
        else:
            starts.append(min(word_ends_rel[i - 1] + 1, end))

    # sum attention per word
    word_scores = []
    T = len(gen_attn_values)
    for i, end in enumerate(word_ends_rel):
        start = starts[i]
        if start > end:
            start = end
        s = max(0, min(start, T - 1))
        e = max(0, min(end,   T - 1))
        if e < s:
            s, e = e, s
        word_scores.append(float(gen_attn_values[s:e + 1].sum()))

    max_attn = max(0.1, float(max(word_scores)) if word_scores else 0.0)

    # find selected word (contains selected token idx)
    selected_word_idx = None
    for i, end in enumerate(word_ends_rel):
        if selected_token_rel_idx <= end:
            selected_word_idx = i
            break
    if selected_word_idx is None and word_ends_rel:
        selected_word_idx = len(word_ends_rel) - 1

    spans = []
    for i, w in enumerate(words_gen):
        alpha = min(1.0, word_scores[i] / max_attn) if max_attn > 0 else 0.0
        bg = f"rgba(66,133,244,{alpha:.3f})"
        border = "2px solid #fff" if i == selected_word_idx else "1px solid transparent"
        spans.append(
            f"<span style='display:inline-block;background:{bg};border:{border};"
            f"border-radius:6px;padding:2px 6px;margin:2px 4px 4px 0;color:#fff;'>"
            f"{w}</span>"
        )

    return (
        "<div style='width:100%;'>"
        "  <div style='background:#444;border:1px solid #eee;border-radius:8px;padding:10px;'>"
        "    <div style='white-space:normal;line-height:1.8;'>"
        f"      {''.join(spans)}"
        "    </div>"
        "  </div>"
        "</div>"
    )

# ========= Heatmap helpers for 1024 image tokens =========
def _attention_to_heatmap_uint8(attn_1d: np.ndarray, img_token_len: int = 1024, side: int = 32) -> np.ndarray:
    """
    attn_1d: (k,) attention over keys for a given generation step; first 1024 are image tokens.
    Returns a (32, 32) uint8 grayscale array.
    """
    # take first 1024 (image tokens); pad/truncate as needed
    if attn_1d.shape[0] < img_token_len:
        img_part = np.zeros(img_token_len, dtype=float)
        img_part[: attn_1d.shape[0]] = attn_1d
    else:
        img_part = attn_1d[:img_token_len]

    # normalize to [0,1]
    mn, mx = float(img_part.min()), float(img_part.max())
    denom = (mx - mn) if (mx - mn) > 1e-12 else 1.0
    norm = (img_part - mn) / denom

    # return uint8 (0–255)
    return (norm.reshape(side, side) * 255.0).astype(np.uint8)

def _colorize_heatmap(heatmap_u8: np.ndarray) -> Image.Image:
    """
    Convert (H,W) uint8 grayscale to RGB heatmap using matplotlib (if available) or a simple fallback.
    """
    if _HAS_MPL and _COLORMAP is not None:
        colored = (_COLORMAP(heatmap_u8.astype(np.float32) / 255.0)[:, :, :3] * 255.0).astype(np.uint8)
        return Image.fromarray(colored)  # Pillow infers RGB
    else:
        # Fallback: map grayscale to red-yellow (simple linear)
        g = heatmap_u8.astype(np.float32) / 255.0
        r = (g * 255.0).clip(0, 255).astype(np.uint8)
        g2 = (np.sqrt(g) * 255.0).clip(0, 255).astype(np.uint8)
        b = np.zeros_like(r, dtype=np.uint8)
        rgb = np.stack([r, g2, b], axis=-1)
        return Image.fromarray(rgb)  # Pillow infers RGB

def _resize_like(img: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
    return img.resize(target_size, resample=Image.BILINEAR)

def _make_overlay(orig: Image.Image, heatmap_rgb: Image.Image, alpha: float = 0.35) -> Image.Image:
    """
    Blend heatmap over original. alpha in [0,1].
    """
    if heatmap_rgb.size != orig.size:
        heatmap_rgb = _resize_like(heatmap_rgb, orig.size)
    base = orig.convert("RGBA")
    overlay = heatmap_rgb.convert("RGBA")
    # set global alpha
    r, g, b = overlay.split()[:3]
    a = Image.new("L", overlay.size, int(alpha * 255))
    overlay = Image.merge("RGBA", (r, g, b, a))
    return Image.alpha_composite(base, overlay).convert("RGB")

# ========= Core (image → generate) =========
def _prepare_image_tensor(pil_img, img_size=512):
    tfm = image_transform(img_size=img_size)
    tens = tfm(pil_img).unsqueeze(0).to(device, non_blocking=True)  # [1,3,H,W]
    return tens

def run_generation(pil_image, max_new_tokens, layer, head, mean_layers, mean_heads):
    """
    1) Transform image
    2) model.generate(pixel_values=..., max_new_tokens=..., output_attentions=True)
       expected to return (gen_ids, gen_text, attentions)
    3) Build selector over generated words only
    4) Initial visualization -> (orig, overlay, heatmap, word HTML)
    """
    if pil_image is None:
        # Return placeholders
        blank = Image.new("RGB", (256, 256), "black")
        return (
            None, None, 1024, None, None,
            gr.update(choices=[], value=None),
            blank,  # original
            blank,  # overlay
            np.zeros((256, 256, 3), dtype=np.uint8),  # heatmap RGB upscaled (placeholder)
            "<div style='text-align:center;padding:20px;'>Upload or load an image first.</div>",
        )

    pixel_values = _prepare_image_tensor(pil_image, img_size=512)

    with torch.no_grad():
        gen_ids, gen_text, attentions = model.generate(
            pixel_values=pixel_values,
            max_new_tokens=int(max_new_tokens),
            output_attentions=True
        )

    # Expect batch size 1
    if isinstance(gen_ids, torch.Tensor):
        gen_ids = gen_ids[0].tolist()
    gen_ids = _strip_trailing_special(gen_ids)

    words_gen, gen_word2tok_rel = _words_and_map_from_tokens_simple(gen_ids)

    display_choices = [(w, i) for i, w in enumerate(words_gen)]
    if not display_choices:
        # No generated tokens; still show original and blank heatmap/overlay
        blank_hm = np.zeros((32, 32), dtype=np.uint8)
        hm_rgb = _colorize_heatmap(blank_hm).resize(pil_image.size, resample=Image.NEAREST)
        overlay = _make_overlay(pil_image, hm_rgb, alpha=0.35)
        return (
            attentions, gen_ids, 1024, words_gen, gen_word2tok_rel,
            gr.update(choices=[], value=None),
            pil_image,                          # original
            overlay,                            # overlay
            np.array(hm_rgb),                   # heatmap RGB
            "<div style='text-align:center;padding:20px;'>No generated tokens to visualize.</div>",
        )

    first_idx = 0
    hm_rgb_init, overlay_init, html_init = update_visualization(
        selected_gen_index=first_idx,
        attentions=attentions,
        gen_token_ids=gen_ids,
        layer=layer,
        head=head,
        mean_layers=mean_layers,
        mean_heads=mean_heads,
        words_gen=words_gen,
        gen_word2tok_rel=gen_word2tok_rel,
        pil_image=pil_image,
    )

    return (
        attentions,            # state_attentions
        gen_ids,               # state_gen_token_ids
        1024,                  # state_img_token_len (fixed)
        words_gen,             # state_words_gen
        gen_word2tok_rel,      # state_gen_word2tok_rel
        gr.update(choices=display_choices, value=first_idx),
        pil_image,             # original image view
        overlay_init,          # overlay (PIL)
        hm_rgb_init,           # heatmap RGB (np array or PIL)
        html_init,             # HTML words viz
    )

def update_visualization(
    selected_gen_index,
    attentions,
    gen_token_ids,
    layer,
    head,
    mean_layers,
    mean_heads,
    words_gen,
    gen_word2tok_rel,
    pil_image: Optional[Image.Image] = None,
):
    """
    Recompute visualization for the chosen GENERATED word:
    - Extract attention vector for that generation step.
    - Build 32×32 heatmap from first 1024 values (image tokens), colorize and upscale to original image size.
    - Create overlay (original + heatmap with alpha).
    - Build word HTML from the portion corresponding to generated tokens.
      For step t, keys cover: 1024 image tokens + (t+1) generated tokens so far.
    """
    if selected_gen_index is None or attentions is None or gen_word2tok_rel is None:
        blank = np.zeros((256, 256, 3), dtype=np.uint8)
        return Image.fromarray(blank), Image.fromarray(blank), "<div style='text-align:center;padding:20px;'>Generate first.</div>"

    gidx = int(selected_gen_index)
    if not (0 <= gidx < len(gen_word2tok_rel)):
        blank = np.zeros((256, 256, 3), dtype=np.uint8)
        return Image.fromarray(blank), Image.fromarray(blank), "<div style='text-align:center;padding:20px;'>Invalid selection.</div>"

    step_index = int(gen_word2tok_rel[gidx])  # last token of that word (relative to generation)
    if not attentions or step_index >= len(attentions):
        blank = np.zeros((256, 256, 3), dtype=np.uint8)
        return Image.fromarray(blank), Image.fromarray(blank), "<div style='text-align:center;padding:20px;'>No attention for this step.</div>"

    token_attn = get_attention_for_token_layer(
        attentions,
        token_index=step_index,
        layer_index=int(layer),
        head_index=int(head),
        mean_across_layers=bool(mean_layers),
        mean_across_heads=bool(mean_heads),
    )

    attn_vals = token_attn.detach().cpu().numpy()
    if attn_vals.ndim == 2:
        attn_vals = attn_vals[-1]  # (k,) from (q,k)

    # ---- Heatmap over 1024 image tokens (colorized and upscaled to original size) ----
    heatmap_u8 = _attention_to_heatmap_uint8(attn_1d=attn_vals, img_token_len=1024, side=32)
    hm_rgb_pil = _colorize_heatmap(heatmap_u8)

    # If original image not provided (should be), create a placeholder size
    if pil_image is None:
        pil_image = Image.new("RGB", (256, 256), "black")

    hm_rgb_pil_up = hm_rgb_pil.resize(pil_image.size, resample=Image.NEAREST)
    overlay_pil = _make_overlay(pil_image, hm_rgb_pil_up, alpha=0.35)

    # ---- Word-level viz over generated tokens only ----
    k_len = int(attn_vals.shape[0])
    observed_gen = max(0, min(step_index + 1, max(0, k_len - 1024)))
    total_gen = len(gen_token_ids)

    gen_vec = np.zeros(total_gen, dtype=float)
    if observed_gen > 0:
        # slice generated part of attention vector
        start = 1024
        end = min(1024 + observed_gen, k_len)
        gen_slice = attn_vals[start:end]
        gen_vec[: len(gen_slice)] = gen_slice

    selected_token_rel_idx = step_index

    html_words = generate_word_visualization_gen_only(
        words_gen=words_gen,
        word_ends_rel=gen_word2tok_rel,
        gen_attn_values=gen_vec,
        selected_token_rel_idx=selected_token_rel_idx,
    )

    # Return (heatmap RGB, overlay, html)
    return np.array(hm_rgb_pil_up), overlay_pil, html_words

def toggle_slider(is_mean):
    return gr.update(interactive=not bool(is_mean))

# ========= Gradio UI =========
EXAMPLES_DIR = "examples"

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🖼️→📝 Image-to-Text Attention Visualizer (three views + text)")
    gr.Markdown(
        "Upload an image or click **Load random sample**, generate text, then select a **generated word**. "
        "Above: original image, overlay (original + attention), and heatmap (colored). "
        "Below: word-level attention over generated text."
    )

    # States
    state_attentions = gr.State(None)         # tuple over generation steps
    state_gen_token_ids = gr.State(None)      # list[int]
    state_img_token_len = gr.State(1024)      # fixed
    state_words_gen = gr.State(None)          # list[str]
    state_gen_word2tok_rel = gr.State(None)   # list[int]
    state_last_image = gr.State(None)         # PIL image of last input

    L, H = model_heads_layers()

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1) Image")
            img_input = gr.Image(type="pil", label="Upload image", height=280)
            btn_load_sample = gr.Button("Load random sample from /examples", variant="secondary")
            sample_status = gr.Markdown("")

            gr.Markdown("### 2) Generation")
            slider_max_tokens = gr.Slider(5, 200, value=40, step=5, label="Max New Tokens")
            btn_generate = gr.Button("Generate", variant="primary")

            gr.Markdown("### 3) Attention")
            check_mean_layers = gr.Checkbox(False, label="Mean Across Layers")
            check_mean_heads = gr.Checkbox(False, label="Mean Across Heads")
            slider_layer = gr.Slider(0, max(0, L - 1), value=0, step=1, label="Layer", interactive=True)
            slider_head  = gr.Slider(0, max(0, H - 1), value=0, step=1, label="Head",  interactive=True)

        with gr.Column(scale=3):
            # Three views row
            with gr.Row():
                img_original_view = gr.Image(
                    value=None,
                    label="Original image",
                    image_mode="RGB",
                    height=256
                )
                img_overlay_view = gr.Image(
                    value=None,
                    label="Overlay (image + attention)",
                    image_mode="RGB",
                    height=256
                )
                heatmap_view = gr.Image(
                    value=None,
                    label="Heatmap (colored)",
                    image_mode="RGB",
                    height=256
                )

            # Word selector & HTML viz below
            radio_word_selector = gr.Radio(
                [], label="Select Generated Word",
                info="Selector lists only generated words"
            )
            html_visualization = gr.HTML(
                "<div style='text-align:center;padding:20px;color:#888;border:1px dashed #888;border-radius:8px;'>"
                "Text attention visualization will appear here.</div>"
            )

    # Sample loader: always use `examples/`
    def _load_sample_from_examples():
        try:
            files = [f for f in os.listdir(EXAMPLES_DIR) if not f.startswith(".")]
            if not files:
                return gr.update(), "No files in /examples."
            fp = os.path.join(EXAMPLES_DIR, random.choice(files))
            pil_img = pil_from_path(fp)
            return gr.update(value=pil_img), f"Loaded sample: {os.path.basename(fp)}"
        except Exception as e:
            return gr.update(), f"Error loading sample: {e}"

    btn_load_sample.click(
        fn=_load_sample_from_examples,
        inputs=[],
        outputs=[img_input, sample_status]
    )

    # Generate
    def _run_and_store(pil_image, *args):
        out = run_generation(pil_image, *args)
        # store the original image for later updates
        return (*out, pil_image)

    btn_generate.click(
        fn=_run_and_store,
        inputs=[img_input, slider_max_tokens, slider_layer, slider_head, check_mean_layers, check_mean_heads],
        outputs=[
            state_attentions,
            state_gen_token_ids,
            state_img_token_len,
            state_words_gen,
            state_gen_word2tok_rel,
            radio_word_selector,
            img_original_view,   # original
            img_overlay_view,    # overlay
            heatmap_view,        # heatmap
            html_visualization,  # words HTML
            state_last_image,    # store original PIL
        ],
    )

    # Update viz on any control change
    def _update_wrapper(selected_gen_index, attn, gen_ids, lyr, hed, meanL, meanH, words, word2tok, last_img):
        hm_rgb, overlay, html = update_visualization(
            selected_gen_index,
            attn,
            gen_ids,
            lyr,
            hed,
            meanL,
            meanH,
            words,
            word2tok,
            pil_image=last_img
        )
        return overlay, hm_rgb, html

    for control in [radio_word_selector, slider_layer, slider_head, check_mean_layers, check_mean_heads]:
        control.change(
            fn=_update_wrapper,
            inputs=[
                radio_word_selector,
                state_attentions,
                state_gen_token_ids,
                slider_layer,
                slider_head,
                check_mean_layers,
                check_mean_heads,
                state_words_gen,
                state_gen_word2tok_rel,
                state_last_image,
            ],
            outputs=[img_overlay_view, heatmap_view, html_visualization],
        )

    # Toggle slider interactivity
    check_mean_layers.change(toggle_slider, check_mean_layers, slider_layer)
    check_mean_heads.change(toggle_slider, check_mean_heads, slider_head)

if __name__ == "__main__":
    print(f"Device: {device}")
    demo.launch(debug=True)
