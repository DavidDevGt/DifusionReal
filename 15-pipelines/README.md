# 15 — Descomposición de Pipelines

> Ningún modelo generativo moderno es una sola red neuronal. Cada uno es un **sistema de componentes** interconectados. Aquí los desarmamos pieza por pieza.

---

## Índice

1. [Anatomía general de un pipeline](#1-anatomía-general)
2. [Pipeline: Stable Diffusion 1.5 / SDXL](#2-sd-sdxl)
3. [Pipeline: Stable Diffusion 3 / 3.5](#3-sd3)
4. [Pipeline: FLUX.1](#4-flux1)
5. [Pipeline: FLUX.2](#5-flux2)
6. [Pipeline: Z-Image](#6-z-image)
7. [Pipeline: Wan 2.2 (Video)](#7-wan22)
8. [Pipeline: LTX-2.3 (Video)](#8-ltx23)
9. [Tabla comparativa de componentes](#9-tabla-componentes)
10. [¿Dónde está el cuello de botella?](#10-bottleneck)

---

## 1. Anatomía general

Todo pipeline de generación sigue este flujo:

```mermaid
graph LR
    A["📝 Prompt"] --> B["Text Encoder(s)"]
    B --> C["Conditioning"]
    D["🎲 Ruido z ~ N(0,I)"] --> E["Denoiser<br/>(U-Net / DiT)"]
    C --> E
    E --> F["Scheduler / Solver"]
    F -->|"loop N pasos"| E
    F --> G["VAE Decoder"]
    G --> H["🖼️ Imagen"]

    style B fill:#533483,stroke:#e94560,color:#fff
    style E fill:#0f3460,stroke:#e94560,color:#fff
    style G fill:#16213e,stroke:#e94560,color:#fff
```

### Componentes universales

| Componente | Función | Ejemplos |
|:-----------|:--------|:---------|
| **Text Encoder** | Convertir texto en embeddings | CLIP, T5, Mistral |
| **Conditioning** | Inyectar info de texto en el denoiser | Cross-attention, AdaLN, joint attention |
| **VAE Encoder** | Comprimir imagen a latent space | KL-VAE, VQ-VAE |
| **Denoiser** | Predecir ruido/velocidad | U-Net, DiT, MMDiT |
| **Scheduler** | Controlar el proceso de denoising | DDPM, Euler, DPM-Solver++ |
| **Guidance** | Amplificar adherencia al prompt | CFG, dynamic CFG |
| **VAE Decoder** | Reconstruir imagen desde latent | Decoder de VAE |

---

## 2. Pipeline: Stable Diffusion 1.5 / SDXL

### SD 1.5

```mermaid
graph TB
    PROMPT["Prompt"] --> CLIP["CLIP ViT-L/14<br/>77 tokens max<br/>768-dim embeddings"]
    CLIP --> CROSS["Cross-Attention<br/>en U-Net"]

    NOISE["z ~ N(0,I)<br/>64×64×4 latent"] --> UNET["U-Net<br/>~860M params<br/>ε-prediction"]
    CROSS --> UNET

    TIME["timestep t"] --> UNET

    UNET --> SCHED["Scheduler<br/>(e.g., DPM-Solver++)<br/>20-50 pasos"]
    SCHED -->|"loop"| UNET
    SCHED --> VAED["VAE Decoder<br/>64×64×4 → 512×512×3"]
    VAED --> IMG["Imagen 512×512"]

    subgraph CFG ["Classifier-Free Guidance"]
        UNET_COND["U-Net(z, t, text)"] --> COMBINE["ε = ε_uncond + w·(ε_cond - ε_uncond)"]
        UNET_UNCOND["U-Net(z, t, ∅)"] --> COMBINE
    end

    style CLIP fill:#533483,stroke:#e94560,color:#fff
    style UNET fill:#0f3460,stroke:#e94560,color:#fff
    style VAED fill:#16213e,stroke:#e94560,color:#fff
    style CFG fill:#1a1a2e,stroke:#e94560,color:#fff
```

| Componente | Especificación |
|:-----------|:--------------|
| Text Encoder | CLIP ViT-L/14 (77 tokens, 768d) |
| Denoiser | U-Net ~860M params |
| Prediction | ε-prediction |
| VAE | KL-f8 (factor 8 de compresión) |
| Latent shape | 64×64×4 (para 512×512 output) |
| Scheduler default | PNDM, luego DPM-Solver++ |
| Guidance | CFG (2 forward passes) |
| VRAM típico | ~4-6 GB (FP16) |

### SDXL

```mermaid
graph TB
    PROMPT["Prompt"] --> CLIP1["CLIP ViT-L/14<br/>768d"]
    PROMPT --> CLIP2["OpenCLIP ViT-bigG/14<br/>1280d"]
    CLIP1 --> CONCAT["Concatenar<br/>768d + 1280d = 2048d"]
    CLIP2 --> CONCAT

    CONCAT --> CROSS["Cross-Attention"]

    SIZE["size/crop conditioning<br/>(resolución, crop coords)"] --> EMBED["Conditioning embeddings"]
    EMBED --> UNET

    NOISE["z ~ N(0,I)<br/>128×128×4"] --> UNET["U-Net ~2.6B params<br/>v-prediction"]
    CROSS --> UNET
    TIME["timestep t"] --> UNET

    UNET --> SCHED["Scheduler<br/>25-50 pasos"]
    SCHED -->|"loop"| UNET
    SCHED --> VAED["VAE Decoder<br/>→ 1024×1024"]
    VAED --> IMG["Imagen 1024×1024"]

    style CLIP1 fill:#533483,stroke:#e94560,color:#fff
    style CLIP2 fill:#533483,stroke:#e94560,color:#fff
    style UNET fill:#0f3460,stroke:#e94560,color:#fff
```

| Cambio vs SD 1.5 | Detalle |
|:-----------------|:--------|
| **2 text encoders** | CLIP + OpenCLIP para embeddings más ricos |
| **U-Net más grande** | 2.6B vs 860M params |
| **v-prediction** | Cambio de ε a v-prediction |
| **Resolución nativa** | 1024×1024 vs 512×512 |
| **Size conditioning** | El modelo conoce la resolución objetivo durante training |

---

## 3. Pipeline: Stable Diffusion 3 / 3.5

```mermaid
graph TB
    PROMPT["Prompt"] --> CLIP_L["CLIP-L<br/>768d"]
    PROMPT --> CLIP_G["CLIP-G<br/>1280d"]
    PROMPT --> T5["T5-XXL<br/>4096d, 256 tokens"]

    CLIP_L --> POOL["Pooled embeddings<br/>(global conditioning)"]
    CLIP_G --> POOL
    T5 --> TEXT_SEQ["Text sequence<br/>(token-level conditioning)"]

    NOISE["z ~ N(0,I)<br/>latent patches"] --> MMDIT["MMDiT<br/>~2B-8B params"]
    TEXT_SEQ --> MMDIT
    POOL -->|"AdaLN"| MMDIT
    TIME["timestep t"] -->|"AdaLN"| MMDIT

    subgraph JointAttn ["Joint Attention (dentro de MMDiT)"]
        TT["Text tokens<br/>(Q_t, K_t, V_t)"]
        IT["Image tokens<br/>(Q_i, K_i, V_i)"]
        JA["Attention([Q_t;Q_i], [K_t;K_i], [V_t;V_i])"]
        TT --> JA
        IT --> JA
    end

    MMDIT --> SCHED["FlowMatch Euler<br/>28 pasos"]
    SCHED -->|"Rectified Flow loop"| MMDIT
    SCHED --> VAED["VAE Decoder<br/>→ 1024×1024"]
    VAED --> IMG["Imagen"]

    style MMDIT fill:#0f3460,stroke:#e94560,color:#fff
    style T5 fill:#533483,stroke:#e94560,color:#fff
    style JointAttn fill:#1a1a2e,stroke:#0f3460,color:#fff
```

| Componente | SD3 | SD 3.5 Large | SD 3.5 Medium |
|:-----------|:----|:-------------|:--------------|
| Text Encoders | CLIP-L + CLIP-G + T5-XXL | CLIP-L + CLIP-G + T5-XXL | CLIP-L + CLIP-G + T5-XXL |
| Denoiser | MMDiT ~2B | MMDiT ~8B | MMDiT-X ~2.5B |
| Prediction | Rectified Flow (velocity) | Rectified Flow | Rectified Flow |
| QK-Norm | No | Sí | Sí |
| Pasos default | 28 | 28 | 28 |
| VRAM (FP16) | ~12GB (sin T5) | ~24GB | ~12GB |

### Cambios fundamentales vs SDXL

1. **U-Net → MMDiT**: Cambio arquitectónico completo
2. **ε-prediction → Rectified Flow**: Cambio de paradigma de entrenamiento
3. **Cross-attention → Joint attention**: Texto e imagen interactúan bidireccionalmente
4. **T5-XXL añadido**: Comprensión de texto mucho más profunda (pero costoso en VRAM)

---

## 4. Pipeline: FLUX.1

```mermaid
graph TB
    PROMPT["Prompt"] --> CLIP_L["CLIP-L<br/>768d pooled"]
    PROMPT --> T5["T5-XXL<br/>4096d, 512 tokens"]

    CLIP_L --> POOL["Pooled → guidance embed"]
    T5 --> TXT_EMB["txt_in embedder"]

    NOISE["z ~ N(0,I)"] --> IMG_EMB["img_in embedder<br/>(patchify + PE)"]

    subgraph DoubleStream ["Double-Stream Blocks (×19)"]
        DS_TXT["Text stream<br/>(self-attn + cross-attn + FFN)"]
        DS_IMG["Image stream<br/>(self-attn + cross-attn + FFN)"]
        DS_TXT <-->|"joint attention"| DS_IMG
    end

    subgraph SingleStream ["Single-Stream Blocks (×38)"]
        SS["Concatenated text+image<br/>→ Self-attention + FFN"]
    end

    TXT_EMB --> DoubleStream
    IMG_EMB --> DoubleStream
    POOL -->|"AdaLN"| DoubleStream
    TIME["timestep t"] -->|"AdaLN"| DoubleStream

    DoubleStream --> SingleStream
    SingleStream --> LINEAR["Linear projection"]
    LINEAR --> SCHED["FlowMatch Euler<br/>20-50 pasos"]
    SCHED -->|"Rectified Flow loop"| DoubleStream
    SCHED --> VAED["VAE Decoder"]
    VAED --> IMG["Imagen ~1MP"]

    style DoubleStream fill:#533483,stroke:#e94560,color:#fff
    style SingleStream fill:#0f3460,stroke:#e94560,color:#fff
    style T5 fill:#16213e,stroke:#e94560,color:#fff
```

| Componente | FLUX.1 [dev] | FLUX.1 [schnell] |
|:-----------|:-------------|:-----------------|
| Text Encoders | CLIP-L + T5-XXL | CLIP-L + T5-XXL |
| Params | 12B | 12B |
| Blocks | 19 double + 38 single | 19 double + 38 single |
| Prediction | Rectified Flow (velocity) | Rectified Flow (distilled) |
| Guidance | CFG (dev) | No CFG (schnell) |
| Pasos | 20-50 | 1-4 |
| Licencia | Non-commercial | Apache 2.0 |

### Innovación: Double + Single stream

La arquitectura FLUX.1 usa un diseño **híbrido**:
- **Double-stream blocks**: Texto e imagen tienen streams separados con joint attention (como MMDiT)
- **Single-stream blocks**: Después de la fusión inicial, texto e imagen se procesan juntos (más eficiente)

Este diseño captura lo mejor de ambos mundos: fusión profunda + eficiencia.

---

## 5. Pipeline: FLUX.2

```mermaid
graph TB
    PROMPT["Prompt"] --> VLM["Mistral-3 24B<br/>(Vision-Language Model)"]
    REF_IMGS["Reference images<br/>(hasta 10)"] --> VLM

    VLM --> RICH_COND["Rich conditioning<br/>- Comprensión espacial<br/>- Atributos<br/>- Relaciones<br/>- Contexto visual"]

    NOISE["z ~ N(0,I)"] --> FLOW_T["Rectified Flow Transformer<br/>32B params"]
    RICH_COND --> FLOW_T
    TIME["timestep"] --> FLOW_T

    FLOW_T --> SCHED["FlowMatch Euler"]
    SCHED -->|"loop"| FLOW_T
    SCHED --> VAED["VAE Decoder (reentrenada)<br/>Optimizada: learnability-quality-compression"]
    VAED --> IMG["Imagen hasta 4MP"]

    style VLM fill:#e94560,stroke:#fff,color:#fff
    style FLOW_T fill:#0f3460,stroke:#e94560,color:#fff
    style VAED fill:#16213e,stroke:#e94560,color:#fff
```

| Componente | FLUX.2 | vs FLUX.1 |
|:-----------|:-------|:----------|
| Comprensión semántica | Mistral-3 24B VLM | CLIP-L + T5-XXL |
| Generador | 32B Flow Transformer | 12B Flow Transformer |
| VAE | Reentrenada from scratch | Heredada |
| Multi-reference | Hasta 10 imágenes | No soportado |
| Resolución máx | 4 megapíxeles | ~1 megapíxel |
| Typography | Mejorada significativamente | Básica |
| VRAM estimado | ~40GB+ (FP8) | ~24GB (FP8) |

### FLUX.2 Klein

Variante distilada para consumer GPUs:
- **Sub-segundo** de generación
- Objetivo: <16GB VRAM
- Distillation + cuantización agresiva

---

## 6. Pipeline: Z-Image

```mermaid
graph TB
    PROMPT["Prompt"] --> TE["Text Encoder"]

    TE --> CONCAT["Concatenar text + image<br/>en secuencia única"]

    NOISE["z ~ N(0,I)"] --> CONCAT

    CONCAT --> S3DIT["S3-DiT (Single-Stream)<br/>6B params<br/>Self-Attention unificada"]
    TIME["timestep"] --> S3DIT

    S3DIT --> SCHED["Flow Matching Solver"]
    SCHED -->|"loop"| S3DIT
    SCHED --> VAED["VAE Decoder"]
    VAED --> IMG["Imagen"]

    style S3DIT fill:#0f3460,stroke:#e94560,color:#fff
```

| Aspecto | Z-Image | vs FLUX.1 |
|:--------|:--------|:----------|
| Arquitectura | S3-DiT (single-stream) | Double+Single stream |
| Params | 6B | 12B |
| VRAM | <16GB | ~24GB |
| Velocidad | Sub-segundo (enterprise) | 5-15 segundos |
| Filosofía | Eficiencia > escala | Escala > eficiencia |

---

## 7. Pipeline: Wan 2.2 (Video)

```mermaid
graph TB
    PROMPT["Text prompt"] --> TE["Text Encoder"]
    IMG_COND["Image conditioning<br/>(opcional: first/last frame)"] --> VAE_ENC["3D Causal VAE Encoder<br/>Compresión espacio-temporal"]

    NOISE["z ~ N(0,I)<br/>3D latent<br/>(T×H×W×C)"] --> MOEDIT["MoE-DiT<br/>Multiple expert blocks"]

    TE --> MOEDIT
    VAE_ENC --> MOEDIT
    TIME["timestep t"] --> ROUTER["Router → selecciona experts"]
    ROUTER --> MOEDIT

    subgraph MoE ["Mixture of Experts"]
        EXP_H["Expert High-Noise<br/>Layout, composición"]
        EXP_M["Expert Mid-Noise<br/>Estructura, semántica"]
        EXP_L["Expert Low-Noise<br/>Detalle, textura"]
    end

    MOEDIT --> SCHED["Flow Matching Solver"]
    SCHED -->|"loop"| MOEDIT
    SCHED --> VAE_DEC["3D Causal VAE Decoder<br/>Latent → Video frames"]
    VAE_DEC --> VIDEO["🎬 Video<br/>720p, 24fps"]

    style MOEDIT fill:#0f3460,stroke:#e94560,color:#fff
    style MoE fill:#1a1a2e,stroke:#533483,color:#fff
    style VAE_ENC fill:#533483,stroke:#e94560,color:#fff
    style VAE_DEC fill:#16213e,stroke:#e94560,color:#fff
```

### ¿Qué cambia de imagen a video?

| Aspecto | Image Pipeline | Video Pipeline (Wan 2.2) |
|:--------|:--------------|:-------------------------|
| VAE | 2D (H×W) | **3D Causal** (T×H×W) |
| Latent | 3D tensor (h×w×c) | **4D tensor** (t×h×w×c) |
| Attention | Spatial only | **Spatial + Temporal** |
| Conditioning | Text + (image) | Text + first/last frame + motion |
| Compute | ~1x | **~50-200x** (proporcional a frames) |
| VRAM | 6-40GB | **24-80GB** |
| Experts | N/A | MoE por timestep |

---

## 8. Pipeline: LTX-2.3 (Video)

```mermaid
graph TB
    PROMPT["Prompt"] --> TE["Text Encoder"]

    subgraph Inputs ["Inputs multimodales"]
        TXT_IN["Text"]
        IMG_IN["Image (I2V)"]
        VID_IN["Video (V2V)"]
        AUD_IN["Audio (A2V)"]
    end

    Inputs --> COND["Conditioning unificado"]

    NOISE["z ~ N(0,I)<br/>Highly compressed latent<br/>(ratio 1:192)"] --> DIT["DiT<br/>~19B params (video+audio)"]
    COND --> DIT
    TIME["timestep"] --> DIT

    DIT --> SCHED["Solver"]
    SCHED -->|"loop"| DIT

    SCHED --> VAE_DEC["VAE Decoder<br/>(dual role:<br/>latent→pixel + final denoise)"]
    VAE_DEC --> OUTPUT["🎬 Video 4K@50fps<br/>+ 🔊 Audio sincronizado"]

    style DIT fill:#0f3460,stroke:#e94560,color:#fff
    style VAE_DEC fill:#533483,stroke:#e94560,color:#fff
```

### Innovación: VAE con doble rol

En LTX, la VAE decoder no solo convierte latents a píxeles — también realiza el **último paso de denoising**. Esto compensa la pérdida de detalle causada por la alta compresión (1:192).

### Evolución de LTX

| Versión | Año | Capacidades clave |
|:--------|:----|:-----------------|
| LTX-Video | Nov 2024 | DiT, faster-than-real-time |
| LTX-2 | Oct 2025 | 19B, audio-video simultáneo |
| LTX-2.3 | Mar 2026 | 4K@50fps, pose/depth/edge, multimodal |
| LTX-2.5 | Mid 2026 | Adaptive keyframes, robotics, HDR |

---

## 9. Tabla comparativa de componentes

| | SD 1.5 | SDXL | SD3 | FLUX.1 | FLUX.2 | Z-Image | Wan 2.2 | LTX-2.3 |
|:--|:-------|:-----|:----|:-------|:-------|:--------|:--------|:--------|
| **Text Enc** | CLIP-L | CLIP-L+G | CLIP-L+G+T5 | CLIP-L+T5 | Mistral-3 24B | TE | TE | TE |
| **Denoiser** | U-Net | U-Net | MMDiT | Flow Transf | Flow Transf | S3-DiT | MoE-DiT | DiT |
| **Params** | 860M | 2.6B | 2-8B | 12B | 32B | 6B | 5-14B | 19B |
| **Prediction** | ε | v | velocity (RF) | velocity (RF) | velocity (RF) | velocity | velocity | velocity |
| **VAE** | KL-f8 | KL-f8 | KL-f8 | Custom | Reentrenada | Custom | 3D Causal | 1:192 compression |
| **Attention** | Self+Cross | Self+Cross | Joint | Double+Single | VLM+Flow | Unified | Spatial+Temporal | Spatiotemporal |
| **Guidance** | CFG | CFG | CFG | CFG | CFG | CFG | CFG | CFG |
| **Pasos** | 20-50 | 25-50 | 28 | 20-50 | ~30 | ~20 | ~30 | ~30 |
| **Output** | 512² | 1024² | 1024² | ~1MP | 4MP | Variable | 720p video | 4K video |
| **Modality** | Image | Image | Image | Image | Image | Image | Video | Video+Audio |

---

## 10. ¿Dónde está el cuello de botella?

```mermaid
pie title "Distribución típica del tiempo de inferencia (FLUX.1, 1024×1024, 30 pasos)"
    "Text Encoding" : 3
    "Denoising loop (30× forward pass)" : 85
    "VAE Decode" : 8
    "Guidance overhead (2× per step)" : 4
```

### El denoiser domina absolutamente

| Componente | Tiempo relativo | VRAM relativo | Optimizable? |
|:-----------|:---------------|:-------------|:-------------|
| Text Encoder | ~3% | ~10-30% | Offload a CPU |
| **Denoiser** | **~85%** | **~60-70%** | **Cuantización, distillation, fewer steps** |
| VAE Decoder | ~8% | ~5% | torch.compile |
| CFG overhead | ~2× denoiser | +50% VRAM peak | CFG distillation, PAG |

### Implicación para optimización

1. **Reducir pasos** (distillation) → reducción lineal del tiempo
2. **Reducir precisión** (FP8/INT4) → reducción ~2-4× de VRAM del denoiser
3. **Eliminar CFG** (distilled guidance) → reducción ~2× de compute
4. **Offload text encoder** → liberación de VRAM sin impacto notable en latencia

---

## Referencias

- Rombach, R. et al. (2022). *High-Resolution Image Synthesis with Latent Diffusion Models*. CVPR.
- Podell, D. et al. (2024). *SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis*.
- Esser, P. et al. (2024). *Scaling Rectified Flow Transformers for High-Resolution Image Synthesis*.
- Black Forest Labs (2024/2025). *FLUX.1 / FLUX.2 Technical Reports*.
- Alibaba/Tongyi Lab (2025). *Wan 2.1/2.2 Technical Reports*.
- Lightricks (2024–2026). *LTX-Video Series Documentation*.

---

*← [14 — Quantization](../14-quantization/README.md) | [16 — Image Models →](../16-image-models/README.md)*
