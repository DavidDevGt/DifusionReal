# 16 — Modelos de Generación de Imagen

> Matriz histórica y técnica de los modelos representativos, desde DDPM hasta los sistemas de agosto de 2026.

---

## Evolución cronológica

```mermaid
timeline
    title Evolución de Modelos de Generación de Imagen
    2020 : DDPM (Ho et al.)
         : U-Net, ε-prediction, 1000 pasos
    2021 : DDIM (Song et al.)
         : Sampling determinístico, 50 pasos
         : Guided Diffusion (Dhariwal & Nichol)
    2022 : Latent Diffusion / SD 1.x (Rombach et al.)
         : Difusión en latent space, CLIP conditioning
    2023 : SDXL (Podell et al.)
         : 2.6B U-Net, dual CLIP, v-prediction
         : PixArt-α (Chen et al.)
         : DiT eficiente, 600M params
    2024 : SD3 (Esser et al.) — MMDiT, Rectified Flow
         : FLUX.1 (BFL) — 12B Flow Transformer
         : AuraFlow — 6.8B, wider-shorter DiT
         : PixArt-δ — ControlNet + LCM
    2025 : SD 3.5 — MMDiT-X, QK-norm
         : FLUX.2 (BFL) — 32B + Mistral-3 24B VLM
         : Z-Image — S3-DiT 6B, sub-second
         : Qwen-Image 2.0 — LLM + generation unificado
    2026 : FLUX.2 Klein — distilled, consumer GPU
         : Qwen-Image 3.0 — 4.5k tokens, micro-detail
```

---

## Model Cards

### DDPM (2020)

| Aspecto | Detalle |
|:--------|:--------|
| **Paper** | *Denoising Diffusion Probabilistic Models* (Ho, Jain, Abbeel) |
| **Arquitectura** | U-Net con self-attention |
| **Parámetros** | ~35.7M (CIFAR-10) / ~114M (LSUN) |
| **Text Encoder** | Ninguno (class-conditional o unconditional) |
| **Conditioning** | Class label (embedding) |
| **VAE** | Ninguno (pixel space) |
| **Training Objective** | ε-prediction, loss simplificado del ELBO |
| **Noise Schedule** | Linear: β₁=10⁻⁴, β_T=0.02, T=1000 |
| **Sampling** | DDPM ancestral, 1000 pasos |
| **Guidance** | Ninguna |
| **Resolución** | 32×32 (CIFAR-10), 256×256 (LSUN) |
| **Fortalezas** | Fundacional, matemáticamente riguroso, training estable |
| **Debilidades** | Lento (1000 pasos), baja resolución, sin text conditioning |
| **Licencia** | MIT |
| **Fuente** | [arxiv:2006.11239](https://arxiv.org/abs/2006.11239) |

---

### Stable Diffusion 1.5 (2022)

| Aspecto | Detalle |
|:--------|:--------|
| **Paper** | *High-Resolution Image Synthesis with Latent Diffusion Models* (Rombach et al.) |
| **Arquitectura** | U-Net con cross-attention |
| **Parámetros** | ~860M (U-Net) + ~123M (CLIP) + ~83M (VAE) |
| **Text Encoder** | CLIP ViT-L/14 (77 tokens, 768d) |
| **Conditioning** | Cross-attention (text → image) |
| **VAE** | KL-regularized, f=8 (compresión 8×) |
| **Latent Shape** | 64×64×4 (para output 512×512) |
| **Training Objective** | ε-prediction |
| **Noise Schedule** | Scaled linear |
| **Sampling** | 20-50 pasos, DPM-Solver++ recomendado |
| **Guidance** | CFG (scale 7.5 default) |
| **Resolución** | 512×512 nativa |
| **VRAM** | ~4GB (FP16), ~6-8GB con safety checker |
| **Fortalezas** | Ecosistema masivo, LoRA/ControlNet/adapters, rápido |
| **Debilidades** | Baja resolución, adherencia al prompt limitada, anatomía deficiente |
| **Licencia** | CreativeML Open RAIL-M |
| **Fuente** | [arxiv:2112.10752](https://arxiv.org/abs/2112.10752) |

---

### SDXL (2023)

| Aspecto | Detalle |
|:--------|:--------|
| **Paper** | *SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis* |
| **Arquitectura** | U-Net (más grande, transformer blocks más profundos) |
| **Parámetros** | ~2.6B (U-Net base) + ~800M (refiner, opcional) |
| **Text Encoders** | CLIP ViT-L/14 (768d) + OpenCLIP ViT-bigG/14 (1280d) |
| **Conditioning** | Cross-attention (2048d concatenados) + size/crop embeddings |
| **VAE** | KL-regularized, f=8 (misma que SD 1.5 pero fine-tuned) |
| **Latent Shape** | 128×128×4 (para output 1024×1024) |
| **Training Objective** | v-prediction (cambio respecto a SD 1.5) |
| **Noise Schedule** | Offset noise + shifted schedule |
| **Sampling** | 25-50 pasos |
| **Guidance** | CFG (scale 5-9 típico) |
| **Resolución** | 1024×1024 nativa, multi-aspect |
| **VRAM** | ~8-12GB (FP16) |
| **Fortalezas** | Gran salto en calidad vs SD 1.5, mejor composición, ecosistema rico |
| **Debilidades** | Text rendering limitado, todavía U-Net, sin T5 |
| **Licencia** | SDXL License (permisiva con restricciones) |
| **Fuente** | [arxiv:2307.01952](https://arxiv.org/abs/2307.01952) |

---

### PixArt-α (2023)

| Aspecto | Detalle |
|:--------|:--------|
| **Paper** | *PixArt-α: Fast Training of Diffusion Transformer for Photorealistic Text-to-Image Synthesis* |
| **Arquitectura** | DiT (Diffusion Transformer) |
| **Parámetros** | ~600M |
| **Text Encoder** | T5-XXL (4096d) |
| **Conditioning** | Cross-attention (T5 tokens) |
| **VAE** | SD VAE (f=8) |
| **Training Objective** | ε-prediction |
| **Innovación** | Training strategy decomposition: (1) pixel distribution → (2) text alignment → (3) aesthetic quality |
| **Training Cost** | ~10.8% del coste de SDXL |
| **Sampling** | DPM-Solver, 14-20 pasos |
| **Resolución** | Hasta 1024×1024 |
| **VRAM** | ~6GB (FP16) |
| **Fortalezas** | Eficiente, demostró viabilidad de DiT, buena adherencia al prompt |
| **Debilidades** | Menor calidad que modelos más grandes |
| **Licencia** | Open weights |
| **Fuente** | [arxiv:2310.00426](https://arxiv.org/abs/2310.00426) |

---

### Stable Diffusion 3 (2024)

| Aspecto | Detalle |
|:--------|:--------|
| **Paper** | *Scaling Rectified Flow Transformers for High-Resolution Image Synthesis* |
| **Arquitectura** | MMDiT (Multimodal Diffusion Transformer) |
| **Parámetros** | ~2B (medium) / ~8B (large) |
| **Text Encoders** | CLIP-L (768d) + CLIP-G (1280d) + T5-XXL (4096d) |
| **Conditioning** | Joint attention (text ↔ image bidireccional) + AdaLN (pooled + timestep) |
| **VAE** | KL-regularized, f=8, 16 channels |
| **Training Objective** | Rectified Flow (velocity prediction) |
| **Innovación** | Joint attention, 3 text encoders, QK-normalization (3.5) |
| **Sampling** | FlowMatch Euler, 28 pasos |
| **Guidance** | CFG (scale ~5) |
| **Resolución** | 1024×1024 nativa |
| **VRAM** | ~12-24GB (FP16, según variante) |
| **Fortalezas** | Mejor text rendering, composición, joint attention |
| **Debilidades** | Calidad inconsistente en SD3 base, mejorado en 3.5 |
| **Licencia** | Stability Community License |
| **Fuente** | [arxiv:2403.03206](https://arxiv.org/abs/2403.03206) |

---

### AuraFlow (2024)

| Aspecto | Detalle |
|:--------|:--------|
| **Arquitectura** | Rectified Flow Transformer ("wider, shorter" design) |
| **Parámetros** | ~6.8B |
| **Text Encoder** | T5 |
| **Conditioning** | Reemplaza bloques MMDiT por single-modality DiT encoder blocks |
| **Training Objective** | Rectified Flow |
| **Innovación** | Mostró que single-modality blocks escalados > muchos MMDiT blocks |
| **Resolución** | 1024×1024 |
| **Fortalezas** | Open source grande, buena prompt adherence, diseño eficiente |
| **Debilidades** | Community model, menor pulido que FLUX |
| **Licencia** | Apache 2.0 |
| **Fuente** | fal.ai |

---

### FLUX.1 (2024)

| Aspecto | Detalle |
|:--------|:--------|
| **Organización** | Black Forest Labs |
| **Arquitectura** | Rectified Flow Transformer (double + single stream) |
| **Parámetros** | ~12B |
| **Text Encoders** | CLIP-L (pooled, 768d) + T5-XXL (sequence, 4096d, 512 tokens) |
| **Conditioning** | Double-stream: joint attention; Single-stream: unified self-attention |
| **VAE** | Custom VAE |
| **Training Objective** | Rectified Flow (velocity prediction) |
| **Blocks** | 19 DoubleStreamBlocks + 38 SingleStreamBlocks |
| **Sampling** | FlowMatch Euler, 20-50 pasos (dev), 1-4 pasos (schnell) |
| **Guidance** | CFG (dev) / guidance-free (schnell) |
| **Resolución** | ~1 megapíxel, multi-aspect |
| **VRAM** | ~24GB (FP16), ~12GB (FP8/NF4) |
| **Variantes** | dev (non-commercial), schnell (Apache 2.0, distilled) |
| **Fortalezas** | Calidad excepcional, text rendering, coherencia |
| **Debilidades** | Alto VRAM, dev es non-commercial |
| **Fuente** | [bfl.ai](https://bfl.ai) |

---

### FLUX.2 (2025)

| Aspecto | Detalle |
|:--------|:--------|
| **Organización** | Black Forest Labs |
| **Arquitectura** | VLM-coupled Rectified Flow Transformer |
| **Parámetros** | ~32B (generador) + ~24B (Mistral-3 VLM) |
| **VLM** | Mistral-3 24B (vision-language model) |
| **Conditioning** | VLM procesa prompt + ref images → rich conditioning |
| **VAE** | Reentrenada from scratch (optimizada learnability-quality-compression) |
| **Training Objective** | Rectified Flow |
| **Multi-reference** | Hasta 10 imágenes fuente |
| **Resolución** | Hasta 4 megapíxeles |
| **Typography** | Significativamente mejorada |
| **VRAM estimado** | ~40GB+ (FP8) |
| **Variantes** | FLUX.2, FLUX.2 Klein (sub-second, consumer GPU) |
| **Fortalezas** | Comprensión semántica profunda, multi-reference, 4MP |
| **Debilidades** | Muy alto VRAM, API-first |
| **Fuente** | [bfl.ai](https://bfl.ai) |

---

### Z-Image (2025)

| Aspecto | Detalle |
|:--------|:--------|
| **Organización** | Alibaba |
| **Arquitectura** | S3-DiT (Scalable Single-Stream Diffusion Transformer) |
| **Parámetros** | ~6B |
| **Conditioning** | Single-stream: texto e imagen en secuencia unificada |
| **Training Objective** | Flow Matching |
| **Variantes** | Z-Image base, Z-Image-Turbo (distilled, few-step) |
| **VRAM** | <16GB |
| **Velocidad** | Sub-segundo (enterprise hardware) |
| **Fortalezas** | Eficiencia extrema, consumer GPU friendly, buena calidad/parámetro |
| **Debilidades** | Menor capacidad absoluta que FLUX.2 |
| **Fuente** | [arxiv](https://arxiv.org), [zimage.run](https://zimage.run) |

---

### Qwen-Image (2025–2026)

| Aspecto | Detalle |
|:--------|:--------|
| **Organización** | Alibaba/Qwen |
| **Arquitectura** | MLLM + VAE + MMDiT |
| **Módulos** | (1) Multimodal LLM → (2) VAE tokenizer → (3) MMDiT backbone |
| **Innovación** | MSRoPE (Multimodal Scalable RoPE), unifica generación + edición |
| **Versiones** | Qwen-Image (2025) → 2512 → 2.0 (Feb 2026) → 3.0 (Jul 2026) |
| **Qwen-Image 3.0** | High-token (4.5k), micro-level detail rendering |
| **Fortalezas** | Text rendering, edición integrada, diseño infográfico |
| **Fuente** | [qwen.ai](https://qwen.ai) |

---

## Tabla resumen comparativa

| Modelo | Año | Arch | Params | Objective | Pasos | Resolución | VRAM (FP16) |
|:-------|:----|:-----|:-------|:----------|:------|:-----------|:------------|
| DDPM | 2020 | U-Net | 35-114M | ε-pred | 1000 | 256² | <2GB |
| SD 1.5 | 2022 | U-Net | 860M | ε-pred | 20-50 | 512² | 4-6GB |
| SDXL | 2023 | U-Net | 2.6B | v-pred | 25-50 | 1024² | 8-12GB |
| PixArt-α | 2023 | DiT | 600M | ε-pred | 14-20 | 1024² | ~6GB |
| SD3 | 2024 | MMDiT | 2-8B | RF velocity | 28 | 1024² | 12-24GB |
| AuraFlow | 2024 | RF Transf | 6.8B | RF velocity | ~25 | 1024² | ~20GB |
| FLUX.1 | 2024 | RF Transf | 12B | RF velocity | 20-50 | ~1MP | ~24GB |
| FLUX.2 | 2025 | VLM+RF | 32B+24B | RF velocity | ~30 | 4MP | ~40GB+ |
| Z-Image | 2025 | S3-DiT | 6B | FM velocity | ~20 | Variable | <16GB |
| Qwen-Image 3.0 | 2026 | LLM+MMDiT | Variable | FM velocity | ~25 | 2K+ | Variable |

---

## Diagrama de evolución de bottlenecks

```mermaid
graph LR
    A["2020: DDPM<br/>Bottleneck: velocidad<br/>(1000 pasos)"] --> B["2021: DDIM<br/>Resuelve: sampling rápido<br/>Nuevo bottleneck: resolución"]
    B --> C["2022: LDM/SD<br/>Resuelve: resolución<br/>Nuevo bottleneck: calidad/adherencia"]
    C --> D["2023: SDXL<br/>Resuelve: calidad<br/>Nuevo bottleneck: text rendering, escalabilidad"]
    D --> E["2024: SD3/FLUX.1<br/>Resuelve: text, escalabilidad<br/>Nuevo bottleneck: VRAM, latencia"]
    E --> F["2025-26: FLUX.2/Klein/Z-Image<br/>Resuelve: eficiencia<br/>Nuevo bottleneck: comprensión semántica profunda"]

    style A fill:#e94560,stroke:#fff,color:#fff
    style F fill:#0f3460,stroke:#fff,color:#fff
```

---

*← [15 — Pipelines](../15-pipelines/README.md) | [17 — Video Models →](../17-video-models/README.md)*
