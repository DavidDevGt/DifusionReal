# 17 — Modelos de Generación de Video

> El video no es "muchas imágenes seguidas". Es un problema fundamentalmente diferente que requiere representaciones espacio-temporales, VAEs 3D, y órdenes de magnitud más compute.

---

## Índice

1. [De imagen a video: qué cambia](#1-qué-cambia)
2. [Arquitecturas de video](#2-arquitecturas)
3. [VAEs 3D y Causal VAEs](#3-vaes-3d)
4. [Temporal attention](#4-temporal-attention)
5. [Conditioning en video](#5-conditioning)
6. [Model cards de video](#6-model-cards)
7. [Scaling computacional: imagen vs video](#7-scaling)

---

## 1. De imagen a video: qué cambia

```mermaid
graph TB
    subgraph Image ["Generación de Imagen"]
        I1["Latent 2D: h × w × c"]
        I2["Attention: spatial only"]
        I3["VAE: 2D encoder/decoder"]
        I4["Output: 1 frame"]
    end

    subgraph Video ["Generación de Video"]
        V1["Latent 3D: t × h × w × c"]
        V2["Attention: spatial + temporal"]
        V3["VAE: 3D causal encoder/decoder"]
        V4["Output: N frames @ fps"]
    end

    style Image fill:#1a1a2e,stroke:#0f3460,color:#fff
    style Video fill:#1a1a2e,stroke:#e94560,color:#fff
```

### Dimensiones del problema

| Aspecto | Imagen (1024²) | Video (720p, 5s, 24fps) | Factor |
|:--------|:---------------|:------------------------|:-------|
| Píxeles totales | 1M | 1M × 120 frames = **120M** | 120× |
| Latent size (f=8) | 128² = 16K | 30 × 90 × 160 = **432K** | 27× |
| Attention O(n²) | n=16K → 268M | n=432K → **186B** | ~700× |
| VRAM típico | 6-24 GB | **24-80+ GB** | 3-10× |
| Latencia | 2-10s | **30s - 5min** | 10-30× |

> **El video multiplica todo**: compute, memoria, latencia, datos de entrenamiento, y complejidad de evaluación.

---

## 2. Arquitecturas de video

### Evolución

```mermaid
timeline
    title Arquitecturas de Video Generation
    2023 : 3D U-Net (Space-Time)
         : Temporal layers insertadas en U-Net 2D
    2024 : Video DiT
         : Transformer con attention espacio-temporal
         : LTX-Video (DiT + high-compression VAE)
    2025 : Wan 2.1 (DiT + 3D Causal VAE)
         : MoE-DiT (Wan 2.2)
         : LTX-2 (audio+video)
    2026 : LTX-2.3 (4K, multimodal)
         : Extensiones multimodales
```

### 3D U-Net (Space-Time)

```mermaid
graph TB
    subgraph ST_UNET ["3D Space-Time U-Net"]
        SPATIAL["Spatial Conv/Attn<br/>(procesa H×W)"]
        TEMPORAL["Temporal Conv/Attn<br/>(procesa T)"]
        SPATIAL --> TEMPORAL
        TEMPORAL --> SPATIAL
    end

    style ST_UNET fill:#1a1a2e,stroke:#e94560,color:#fff
```

- Extiende U-Net 2D insertando **temporal layers** (1D conv + temporal attention)
- Pro: Puede inicializar desde U-Net 2D pre-entrenada (image weights)
- Con: Limitado por la arquitectura U-Net subyacente

### Video DiT

```mermaid
graph TB
    VIDEO["Video latent<br/>t × h × w × c"] --> PATCH3D["3D Patchify<br/>→ secuencia de tokens"]
    PATCH3D --> PE["+ 3D Positional Embedding"]
    PE --> TB["Transformer Blocks ×N<br/>Full spatiotemporal attention"]
    TB --> UNPATCH["Unpatchify<br/>→ predicted noise"]

    COND["Text + timestep"] -->|"AdaLN"| TB

    style TB fill:#0f3460,stroke:#e94560,color:#fff
    style PATCH3D fill:#533483,stroke:#e94560,color:#fff
```

---

## 3. VAEs 3D y Causal VAEs

### ¿Por qué no usar la VAE 2D frame por frame?

| Enfoque | Consistencia temporal | Eficiencia | Calidad |
|:--------|:---------------------|:-----------|:--------|
| VAE 2D por frame | ✗ (flickering) | Baja (N decodificaciones) | Buena por frame |
| **VAE 3D** | ✓ (temporal coherence) | Alta (1 decodificación) | Buena + consistente |

### 3D Causal VAE (Wan)

```mermaid
graph LR
    subgraph Encoder ["3D Causal Encoder"]
        V_IN["Video frames<br/>f₁, f₂, ..., fₙ"]
        CONV3D["3D Causal Convolutions<br/>(temporal causal: solo ve pasado)"]
        Z_OUT["Latent 3D<br/>t × h × w × c"]
        V_IN --> CONV3D --> Z_OUT
    end

    subgraph Decoder ["3D Causal Decoder"]
        Z_IN["Latent 3D"]
        DECONV3D["3D Causal DeConvolutions"]
        V_OUT["Video frames reconstruidos"]
        Z_IN --> DECONV3D --> V_OUT
    end

    style Encoder fill:#1a1a2e,stroke:#0f3460,color:#fff
    style Decoder fill:#1a1a2e,stroke:#533483,color:#fff
```

### ¿Por qué "causal"?

**Causal** = cada frame solo puede depender de frames **anteriores**, no futuros. Esto permite:
- Generar video de **longitud arbitraria** (streaming)
- Mantener consistencia temporal sin requerir todos los frames en memoria
- Compatible con **auto-regressive extension** (generar más frames)

### Comparación de VAEs de video

| VAE | Compresión espacial | Compresión temporal | Ratio total | Modelo |
|:----|:--------------------|:-------------------|:------------|:-------|
| Wan 3D Causal VAE | 8× | 4× | 256× | Wan 2.1/2.2 |
| Wan Advanced VAE | 8× | 8× | 512× | Wan 2.2 (variantes) |
| LTX Video VAE | 32× | ~6× | **192×** | LTX-Video |
| CogVideoX VAE | 8× | 4× | 256× | CogVideoX |

---

## 4. Temporal attention

### Tipos de attention en video

```mermaid
graph TB
    subgraph SpatialOnly ["Spatial-Only Attention"]
        S1["Frame 1: attn(H×W)"]
        S2["Frame 2: attn(H×W)"]
        S3["Frame 3: attn(H×W)"]
    end

    subgraph SpatialTemporal ["Spatial + Temporal Attention"]
        ST1["Spatial attn(H×W) per frame"]
        ST2["Temporal attn(T) per position"]
        ST1 --> ST2
    end

    subgraph FullST ["Full Spatiotemporal (costoso)"]
        FST["attn(T×H×W)<br/>Todos los tokens"]
    end

    style SpatialOnly fill:#e94560,stroke:#fff,color:#fff
    style SpatialTemporal fill:#533483,stroke:#fff,color:#fff
    style FullST fill:#0f3460,stroke:#fff,color:#fff
```

| Tipo | Complejidad | Consistencia temporal | Uso |
|:-----|:-----------|:---------------------|:----|
| Spatial only | O((H·W)²) por frame | ✗ (sin conexión temporal) | No usado solo |
| **Factorized (spatial + temporal)** | O((H·W)² + T²) | ✓ (buena) | Wan, la mayoría |
| Full spatiotemporal | O((T·H·W)²) | ✓✓ (mejor) | LTX (en latent comprimido) |
| Causal temporal | O(T·(H·W)) | ✓ (con restricción causal) | Generación streaming |

---

## 5. Conditioning en video

### Tipos de conditioning para video

```mermaid
graph TB
    subgraph TextCond ["Text-to-Video (T2V)"]
        T2V_P["'A cat jumping over a fence'"] --> T2V_M["Model"] --> T2V_V["Video"]
    end

    subgraph ImgCond ["Image-to-Video (I2V)"]
        I2V_I["Imagen (first frame)"] --> I2V_M["Model"] --> I2V_V["Video animado"]
        I2V_T["Texto (opcional)"] --> I2V_M
    end

    subgraph VidCond ["Video-to-Video (V2V)"]
        V2V_V1["Video original"] --> V2V_M["Model"] --> V2V_V2["Video editado"]
        V2V_T["Texto de edición"] --> V2V_M
    end

    subgraph MultiCond ["Multi-Conditioning"]
        MC_F["First frame"] --> MC_M["Model"] --> MC_V["Video"]
        MC_L["Last frame"] --> MC_M
        MC_T["Texto"] --> MC_M
        MC_P["Pose / depth"] --> MC_M
        MC_A["Audio"] --> MC_M
    end

    style TextCond fill:#0d1117,stroke:#0f3460,color:#fff
    style ImgCond fill:#0d1117,stroke:#533483,color:#fff
    style VidCond fill:#0d1117,stroke:#e94560,color:#fff
    style MultiCond fill:#0d1117,stroke:#16213e,color:#fff
```

### Tabla de capacidades por modelo

| Capability | Wan 2.1 | Wan 2.2 | LTX-2 | LTX-2.3 |
|:-----------|:--------|:--------|:------|:--------|
| T2V | ✓ | ✓ | ✓ | ✓ |
| I2V | ✓ | ✓ | ✓ | ✓ |
| V2V | ✗ | ✓ | ✗ | ✓ |
| First/Last frame | ✓ | ✓ | ✓ | ✓ |
| Pose conditioning | ✗ | ✗ | ✗ | ✓ |
| Depth conditioning | ✗ | ✗ | ✗ | ✓ |
| Audio generation | ✗ | ✓ (ext.) | ✓ | ✓ |
| Camera control | ✗ | ✓ | ✗ | ✓ |
| Video extension | ✓ | ✓ | ✓ | ✓ |

---

## 6. Model cards de video

### Wan 2.1

| Aspecto | Detalle |
|:--------|:--------|
| **Organización** | Alibaba / Tongyi Lab |
| **Arquitectura** | DiT + 3D Causal VAE |
| **Parámetros** | 1.3B (small) / 14B (large) |
| **VAE** | 3D Causal VAE (compresión espacio-temporal) |
| **Framework** | Flow Matching |
| **Resolución** | Hasta 720p |
| **FPS** | 24 |
| **Duración** | Hasta ~10 segundos |
| **Capacidades** | T2V, I2V, first/last frame |
| **Fortalezas** | Open-source, buen balance calidad/eficiencia |
| **Fuente** | [arxiv:2503.20314](https://arxiv.org/abs/2503.20314) |

### Wan 2.2

| Aspecto | Detalle |
|:--------|:--------|
| **Innovación clave** | MoE-DiT (Mixture of Experts por timestep) |
| **VAE** | Advanced 3D Causal VAE (mayor compresión) |
| **Datos** | Dataset significativamente ampliado |
| **Nuevas capacidades** | Animate (character animation), S2V (sound-to-video) |
| **Control estético** | Etiquetas de iluminación, composición, contraste |
| **Variante eficiente** | Wan2.2-TI2V-5B (720p@24fps en RTX 4090) |
| **Fuente** | [wan.video](https://wan.video) |

### LTX-Video (original)

| Aspecto | Detalle |
|:--------|:--------|
| **Organización** | Lightricks |
| **Arquitectura** | DiT + High-compression Video-VAE |
| **Innovación** | VAE unificada con transformer (pipeline integrado) |
| **Compresión** | 1:192 (extrema) |
| **Velocidad** | Faster-than-real-time |
| **Fuente** | Lightricks, 2024 |

### LTX-2

| Aspecto | Detalle |
|:--------|:--------|
| **Parámetros** | ~19B (asimétrico: video + audio) |
| **Innovación clave** | Audio-video sincronizado en un solo pass |
| **Streams** | Streams separados para video y audio |
| **Fuente** | Lightricks, Oct 2025 |

### LTX-2.3

| Aspecto | Detalle |
|:--------|:--------|
| **Resolución** | Nativa 4K @ 50fps |
| **Duración** | Hasta 20 segundos |
| **VAE** | Reconstruida, menor blur, texturas más nítidas |
| **Conditioning** | Pose, depth, edge (built-in) |
| **Multimodal** | T2V + I2V + audio en un solo modelo |
| **Capacidades pro** | Clean plate (object removal), SDR→HDR |
| **Fuente** | Lightricks, Mar 2026 |

---

## 7. Scaling computacional: imagen vs video

### El cubo del compute

```mermaid
graph TB
    subgraph Compute ["Factores de scaling"]
        SPATIAL["Resolución espacial<br/>512² → 720p → 1080p → 4K"]
        TEMPORAL["Duración temporal<br/>1s → 5s → 10s → 30s"]
        QUALITY["Calidad/fidelidad<br/>Draft → Standard → Cinema"]
    end

    SPATIAL --> TOTAL["Compute total ∝<br/>resolución² × frames × model_size × NFE"]
    TEMPORAL --> TOTAL
    QUALITY --> TOTAL

    style TOTAL fill:#e94560,stroke:#fff,color:#fff
```

### Tabla de scaling

| Configuración | Tokens latentes | Relativo a SD (512²) | VRAM est. |
|:-------------|:---------------|:---------------------|:----------|
| SD 1.5 (512²) | 4,096 | 1× | 4-6 GB |
| SDXL (1024²) | 16,384 | 4× | 8-12 GB |
| FLUX.1 (1024²) | 16,384 | 4× | 24 GB |
| Wan 2.1 (720p, 3s, 24fps) | ~200,000 | **49×** | 24-40 GB |
| Wan 2.2 (720p, 5s, 24fps) | ~350,000 | **85×** | 40-80 GB |
| LTX-2.3 (4K, 5s, 50fps) | ~2,000,000 | **488×** | 80+ GB |

### La jerarquía de bottlenecks en video

```mermaid
graph TB
    B1["1. Attention O(n²)<br/>n = T×H×W tokens"]
    B2["2. VRAM para activaciones<br/>(gradientes, KV cache)"]
    B3["3. VAE decode<br/>(3D decode es costoso)"]
    B4["4. Datos de entrenamiento<br/>(video >> imágenes)"]
    B5["5. Temporal consistency<br/>(evaluar es subjetivo)"]

    B1 --> B2 --> B3 --> B4 --> B5

    style B1 fill:#e94560,stroke:#fff,color:#fff
    style B5 fill:#0f3460,stroke:#fff,color:#fff
```

---

## Diagrama: la evolución hacia multimodal

```mermaid
graph LR
    T2I["Text → Image<br/>(2022)"] --> T2V["Text → Video<br/>(2023-2024)"]
    T2V --> I2V["Image → Video<br/>(2024)"]
    I2V --> AV["Audio + Video<br/>(2025)"]
    AV --> MULTI["Text + Image + Video + Audio<br/>→ Video<br/>(2026)"]

    style T2I fill:#e94560,stroke:#fff,color:#fff
    style MULTI fill:#0f3460,stroke:#fff,color:#fff
```

---

## Referencias

- Alibaba/Tongyi (2025). *Wan: Open and Advanced Large-Scale Video Generative Models*.
- Lightricks (2024-2026). *LTX-Video Series*.
- Ho, J. et al. (2022). *Video Diffusion Models*.
- Blattmann, A. et al. (2023). *Align Your Latents: High-Resolution Video Synthesis with Latent Diffusion Models*.

---

*← [16 — Image Models](../16-image-models/README.md) | [18 — Evaluation →](../18-evaluation/README.md)*
