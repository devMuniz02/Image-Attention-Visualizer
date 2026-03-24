[![LinkedIn](https://img.shields.io/badge/LinkedIn-devmuniz-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/devmuniz)
[![GitHub Profile](https://img.shields.io/badge/GitHub-devMuniz02-181717?logo=github&logoColor=white)](https://github.com/devMuniz02)
[![Portfolio](https://img.shields.io/badge/Portfolio-devmuniz02.github.io-0F172A?logo=googlechrome&logoColor=white)](https://devmuniz02.github.io/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-manu02-FFD21E?logoColor=black)](https://huggingface.co/manu02)

# [Github repo](https://github.com/devMuniz02/Image-Attention-Visualizer)

Image Attention Visualizer is an interactive Gradio app that visualizes **cross-modal attention** between image tokens and generated text tokens in a custom multimodal model. It allows researchers and developers to see how different parts of an image influence the model’s textual output, token by token.

An interactive Gradio app to **generate text from an image using a custom multimodal model** and **visualize attention in real time**. It provides 3 synchronized views — original image, attention overlay, and heatmap — plus a **word-level visualization** showing how each generated word attends to visual regions.

## Overview

Interactive Gradio app for visualizing image-to-text attention maps in custom vision–language models.

## Repository Structure

| Path | Description |
| --- | --- |
| `assets/` | Images, figures, or other supporting media used by the project. |
| `examples/` | Sample inputs, demos, or reference runs for the project. |
| `utils/` | Reusable helper modules and shared utility functions. |
| `.gitignore` | Top-level file included in the repository. |
| `app.py` | Top-level file included in the repository. |
| `LICENSE` | Repository license information. |
| `README.md` | Primary project documentation. |
| `requirements.txt` | Python dependency specification for local setup. |

## Getting Started

1. Clone the repository.

   ```bash
   git clone https://github.com/devMuniz02/Image-Attention-Visualizer.git
   cd Image-Attention-Visualizer
   ```

2. Prepare the local environment.

Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Run or inspect the project entry point.

Run the main application:
```bash
python app.py
```

## Quickstart

### 1) Clone

```bash
git clone https://github.com/devMuniz02/Image-Attention-Visualizer
cd Image-Attention-Visualizer
```

### 2) (Optional) Create a virtual environment

**Windows (PowerShell):**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS / Linux (bash/zsh):**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3) Install requirements

```bash
pip install -r requirements.txt
```

### 4) Run the app

```bash
python app.py
```

You should see something like:

```
Running on local URL:  http://127.0.0.1:7860
```

### 5) Open in your browser

Navigate to `http://127.0.0.1:7860` to use the app.

---

## How to use

1. **Upload an image** or load a random sample from your dataset folder.
2. **Set generation parameters**:

   * Max New Tokens
   * Layer/Head selection (or average across all)
3. Click **Generate** — the model will produce a textual description or continuation.
4. **Select a generated word** from the list:

   * The top row will show:

     * Left → **Original image**
     * Center → **Overlay (attention on image regions)**
     * Right → **Colored heatmap**
   * The bottom section highlights attention strength over the generated words.

---

## Files

* `app.py` — Main Gradio interface and visualization logic.
* `utils/models/complete_model.py` — Model definition and generation method.
* `utils/processing.py` — Image preprocessing utilities.
* `requirements.txt` — Dependencies.
* `README.md` — This file.

---

## ️ Troubleshooting

* **Black or blank heatmap:** Ensure your model returns `output_attentions=True` in `.generate()`.
* **Low resolution or distortion:** Adjust `img_size` or the interpolation method inside `_make_overlay`.
* **Tokenizer error:** Make sure `model.decoder.tokenizer` exists and is loaded correctly.
* **OOM errors:** Reduce `max_new_tokens` or use a smaller model checkpoint.
* **Color or shape mismatch:** Verify that your image tokens length = 1024 (for a 32×32 layout).

---

## Model integration notes

* The app is compatible with any **encoder–decoder or vision–language model** that:

  * Accepts `pixel_values` as input.
  * Returns `generate(..., output_attentions=True)` with `(gen_ids, gen_text, attentions)`.
* Uses the tokenizer from `model.decoder.tokenizer`.
* Designed for research in **vision-language interpretability**, **cross-modal explainability**, and **attention visualization**.

---

## Acknowledgments

* Built with [Gradio](https://www.gradio.app/) and [Hugging Face Transformers](https://huggingface.co/docs/transformers).
* Inspired by the original [Token-Attention-Viewer](https://github.com/devMuniz02/Token-Attention-Viewer) project.
* Special thanks to the open-source community advancing **vision-language interpretability**.

## What the app does

* **Generates text** from an image input using your custom model (`create_complete_model`).
* Displays **three synchronized views**:

  1. 🖼️ **Original image**
  2. 🔥 **Overlay** (original + attention heatmap)
  3. 🌈 **Heatmap alone**
* **Word-level attention viewer**: select any generated word to see how its attention is distributed across the image and previously generated words.
* Works directly with your **custom tokenizer (`model.decoder.tokenizer`)**.
* Fixed-length **1024 image tokens (32×32 grid)** projected as a visual heatmap.
* Adjustable options: **Layer**, **Head**, or **Mean Across Layers/Heads**.

---
