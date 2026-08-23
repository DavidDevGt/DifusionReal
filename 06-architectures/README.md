# 06 — Evolución Arquitectónica

> De U-Net a Diffusion Transformers: cómo y por qué cambió la columna vertebral de los modelos generativos.

---

## Índice

1. [Visión general de la evolución](#1-visión-general)
2. [U-Net: la arquitectura fundacional](#2-u-net)
3. [DiT: Diffusion Transformers](#3-dit)
4. [MMDiT: Multi-Modal DiT](#4-mmdit)
5. [S3-DiT: Single-Stream Scalable DiT](#5-s3-dit)
6. [MoE-DiT: Mixture of Experts](#6-moe-dit)
7. [VLM-coupled architectures](#7-vlm-coupled)
8. [Tabla comparativa completa](#8-tabla-comparativa)
9. [¿Por qué ocurrió la transición?](#9-por-qué)

---

## 1. Visión general

```mermaid
timeline
    title Evolución Arquitectónica de Modelos de Difusión
    2020 : U-Net (DDPM)
         : ResNet blocks + Self-attention
    2022 : U-Net + Cross-Attention (Stable Diffusion)
         : Latent Diffusion con text conditioning
    2023 : DiT (Peebles & Xie)
         : Transformer puro reemplaza U-Net
         : PixArt-α (600M params, DiT eficiente)
    2024 : MMDiT (SD3)
         : Joint attention text+image
         : FLUX.1 (12B, Rectified Flow Transformer)
         : AuraFlow (6.8B, wider-shorter DiT)
    2025 : FLUX.2 (32B + Mistral-3 24B VLM)
         : S3-DiT / Z-Image (single-stream, 6B)
         : MoE-DiT / Wan 2.2 (experts por timestep)
    2026 : Qwen-Image 3.0 (LLM + MMDiT unificado)
         : Arquitecturas VLM-coupled dominan
```

---

## 2. U-Net

### Arquitectura

La U-Net para difusión es una red **encoder-decoder** con skip connections.

```mermaid
graph TB
    subgraph Encoder ["Encoder (downsampling)"]
        E1["Conv Block 64"] --> E2["Conv Block 128"]
        E2 --> E3["Conv Block 256"]
        E3 --> E4["Conv Block 512"]
    end

    E4 --> BN["Bottleneck<br/>Self-Attention + ResNet"]

    subgraph Decoder ["Decoder (upsampling)"]
        D4["Conv Block 512"] --> D3["Conv Block 256"]
        D3 --> D2["Conv Block 128"]
        D2 --> D1["Conv Block 64"]
    end

    BN --> D4

    E1 -.->|"skip connection"| D1
    E2 -.->|"skip connection"| D2
    E3 -.->|"skip connection"| D3
    E4 -.->|"skip connection"| D4

    T["Timestep t"] -->|"sinusoidal embedding"| E1
    T -->|"inyectado en cada bloque"| E2

    style Encoder fill:#1a1a2e,stroke:#e94560,color:#fff
    style Decoder fill:#1a1a2e,stroke:#0f3460,color:#fff
    style BN fill:#533483,stroke:#e94560,color:#fff
```

### Componentes clave

#### ResNet Block
```
Input → GroupNorm → SiLU → Conv3x3 → GroupNorm → SiLU → Conv3x3 → + Input (residual)
                                  ↑
                          time_embedding (projection lineal)
```

#### Self-Attention (dentro de U-Net)
- Se aplica solo en resoluciones intermedias (e.g., 16×16, 32×32) por coste computacional
- Complejidad: O(n²) donde n = H×W (número de spatial tokens)

#### Cross-Attention (Stable Diffusion)
```
Q = projection(image_features)
K = projection(text_embeddings)    ← CLIP text encoder
V = projection(text_embeddings)

Attention(Q, K, V) = softmax(QKᵀ / √d) · V
```

> **Insight**: La cross-attention es el mecanismo principal por el cual el texto controla la generación. Cada token de texto puede influenciar cada posición espacial de la imagen.

### Limitaciones de U-Net

| Limitación | Consecuencia |
|:-----------|:-------------|
| Inductive bias espacial fuerte | Difícil de escalar a resoluciones variables |
| Attention solo en resoluciones bajas | Limita la coherencia global |
| Complejidad arquitectónica | Encoder/decoder/skip connections/attention interactúan de formas complejas |
| Escalado no trivial | Añadir capacidad requiere decisiones de diseño no obvias |
| Acoplamiento resolución-parámetros | Cambiar resolución puede requerir re-entrenar |

---

## 3. DiT

### La idea central

**Peebles & Xie (2023)**: Reemplazar completamente la U-Net por un **Vision Transformer estándar** (ViT).

```mermaid
graph TB
    IMG["Latent z ∈ ℝ^(h×w×c)"] --> PATCH["Patchify<br/>z → patches p ∈ ℝ^(N×d)"]
    PATCH --> PE["+ Positional Embedding"]
    PE --> TB1["Transformer Block 1"]
    TB1 --> TB2["Transformer Block 2"]
    TB2 --> TBN["... × L bloques"]
    TBN --> NORM["Final LayerNorm"]
    NORM --> LINEAR["Linear projection"]
    LINEAR --> UNPATCH["Unpatchify<br/>→ predicted noise ε̂"]

    TIME["Timestep t"] --> ADALN["AdaLN-Zero"]
    LABEL["Class/Text"] --> ADALN
    ADALN -.->|"modula γ, β, α"| TB1
    ADALN -.->|"modula"| TB2
    ADALN -.->|"modula"| TBN

    style IMG fill:#16213e,stroke:#e94560,color:#fff
    style PATCH fill:#0f3460,stroke:#e94560,color:#fff
    style ADALN fill:#533483,stroke:#e94560,color:#fff
    style UNPATCH fill:#16213e,stroke:#0f3460,color:#fff
```

### Mecanismo AdaLN-Zero

En lugar de inyectar el timestep como un bias (como en U-Net), DiT usa **Adaptive Layer Normalization**:

```
AdaLN(h, c) = γ(c) · LayerNorm(h) + β(c)
```

donde `γ(c)` y `β(c)` son funciones del conditioning (timestep + class/text).

**"-Zero"**: Los parámetros de escala se inicializan a cero, haciendo que cada bloque transformer actúe como identidad al inicio del training → training más estable.

### Escalado de DiT

| Variante | Layers | Hidden dim | Heads | Params |
|:---------|:-------|:-----------|:------|:-------|
| DiT-S | 12 | 384 | 6 | 33M |
| DiT-B | 12 | 768 | 12 | 130M |
| DiT-L | 24 | 1024 | 16 | 458M |
| DiT-XL | 28 | 1152 | 16 | 675M |

> **Resultado clave**: DiT escala de manera predecible — mayor modelo = menor FID. Esto conecta con la literatura de scaling laws de LLMs y justifica construir modelos cada vez más grandes.

### ¿Por qué funciona mejor que U-Net?

1. **Sin inductive bias espacial forzado**: El transformer ve toda la imagen como una secuencia de tokens
2. **Escalado directo**: Más layers, más dimensión → mejor rendimiento predecible
3. **Atención global**: Cada patch atiende a todos los demás desde el primer layer
4. **Flexibilidad de resolución**: Los patches se adaptan más naturalmente a diferentes resoluciones
5. **Infraestructura de LLMs**: Optimizaciones existentes (FlashAttention, tensor parallelism) se aplican directamente

---

## 4. MMDiT

### Motivación

En un DiT estándar, el text conditioning entra como un "side input" a través de AdaLN o cross-attention separada. MMDiT (SD3) propone tratar **texto e imagen como ciudadanos de primera clase** dentro del mismo mecanismo de atención.

### Arquitectura

```mermaid
graph TB
    subgraph Input ["Inputs separados"]
        TEXT["Text tokens<br/>(CLIP + T5)"] --> TEXT_EMB["Text Embedder<br/>(Linear + PE)"]
        IMAGE["Image latents<br/>(VAE encoded)"] --> IMG_EMB["Image Embedder<br/>(Patchify + PE)"]
        TIME["Timestep"] --> TIME_EMB["Time Embedder"]
    end

    subgraph MMDiTBlock ["MMDiT Block (×N)"]
        direction TB
        TEXT_STREAM["Text Stream<br/>LayerNorm → QKV"]
        IMG_STREAM["Image Stream<br/>LayerNorm → QKV"]

        TEXT_STREAM --> JOINT["Joint Attention<br/>concat([Q_txt, Q_img])<br/>concat([K_txt, K_img])<br/>concat([V_txt, V_img])"]
        IMG_STREAM --> JOINT

        JOINT --> TEXT_OUT["Text output<br/>(pesos separados)"]
        JOINT --> IMG_OUT["Image output<br/>(pesos separados)"]

        TEXT_OUT --> TEXT_FFN["Text FFN"]
        IMG_OUT --> IMG_FFN["Image FFN"]
    end

    TEXT_EMB --> TEXT_STREAM
    IMG_EMB --> IMG_STREAM
    TIME_EMB -.->|"modula via AdaLN"| MMDiTBlock

    IMG_FFN --> OUTPUT["Unpatchify → ε̂"]

    style Input fill:#0d1117,stroke:#e94560,color:#fff
    style MMDiTBlock fill:#1a1a2e,stroke:#0f3460,color:#fff
    style JOINT fill:#533483,stroke:#e94560,color:#fff
```

### Joint Attention vs Cross-Attention

| Mecanismo | Cómo funciona | Flujo de información |
|:----------|:-------------|:--------------------|
| **Cross-Attention** (SD 1.x) | Q=image, KV=text | Unidireccional: text→image |
| **Joint Attention** (MMDiT) | QKV = concat(text, image) | Bidireccional: text↔image |

> **Por qué importa**: La joint attention permite que los tokens de texto se "actualicen" basándose en lo que la imagen está generando. Esto mejora dramáticamente la adherencia al prompt y la capacidad de renderizar texto en imágenes.

### Variante MMDiT-X (SD 3.5 Medium)

- Añade **dual attention layers** en los primeros bloques
- Incluye **self-attention por modalidad** antes de la joint attention
- Usa **QK-normalization** para estabilidad de training

---

## 5. S3-DiT

### Arquitectura (Z-Image, 2025)

**Single-Stream**: En lugar de tener streams separados para texto e imagen que se fusionan en la attention (como MMDiT), S3-DiT procesa texto e imagen en una **secuencia unificada desde el principio**.

```mermaid
graph LR
    TEXT["Text tokens"] --> CONCAT["Concatenar<br/>en una sola<br/>secuencia"]
    IMAGE["Image patches"] --> CONCAT
    CONCAT --> TB["Transformer Block ×N<br/>(Self-Attention sobre<br/>secuencia completa)"]
    TB --> SPLIT["Separar<br/>text / image"]
    SPLIT --> OUT["Image output → ε̂"]

    style CONCAT fill:#533483,stroke:#e94560,color:#fff
    style TB fill:#0f3460,stroke:#e94560,color:#fff
```

### Comparación de streams

```mermaid
graph TB
    subgraph DualStream ["MMDiT (Dual-Stream)"]
        DS_T["Text stream<br/>Pesos separados"]
        DS_I["Image stream<br/>Pesos separados"]
        DS_J["Joint Attention"]
        DS_T --> DS_J
        DS_I --> DS_J
    end

    subgraph SingleStream ["S3-DiT (Single-Stream)"]
        SS_C["Secuencia concatenada"]
        SS_A["Self-Attention unificada<br/>Mismos pesos"]
        SS_C --> SS_A
    end

    style DualStream fill:#1a1a2e,stroke:#e94560,color:#fff
    style SingleStream fill:#1a1a2e,stroke:#0f3460,color:#fff
```

| Aspecto | MMDiT (Dual-Stream) | S3-DiT (Single-Stream) |
|:--------|:--------------------|:-----------------------|
| Pesos | Separados por modalidad | Compartidos |
| Params (mismo rendimiento) | Mayor | ~50% menor |
| VRAM | Mayor | Menor |
| Latencia | Mayor | Menor |
| Expresividad por modalidad | Mayor | Menor (compartida) |

---

## 6. MoE-DiT

### Motivación (Wan 2.2, 2025)

¿Y si diferentes timesteps requieren diferentes "expertos"?

- **Timesteps altos (mucho ruido)**: El modelo necesita resolver la composición global (layout, objetos principales)
- **Timesteps bajos (poco ruido)**: El modelo necesita refinar detalles finos (texturas, iluminación)

### Arquitectura

```mermaid
graph TB
    INPUT["Input tokens + timestep t"] --> ROUTER["Router Network<br/>f(t) → expert selection"]

    ROUTER -->|"t alto (ruido)"| EXP1["Expert 1<br/>Layout & Composition"]
    ROUTER -->|"t medio"| EXP2["Expert 2<br/>Structure & Semantics"]
    ROUTER -->|"t bajo (detalle)"| EXP3["Expert 3<br/>Detail & Texture"]

    EXP1 --> COMBINE["Combine<br/>(weighted sum)"]
    EXP2 --> COMBINE
    EXP3 --> COMBINE

    COMBINE --> OUTPUT["Output"]

    style ROUTER fill:#533483,stroke:#e94560,color:#fff
    style EXP1 fill:#e94560,stroke:#fff,color:#fff
    style EXP2 fill:#0f3460,stroke:#fff,color:#fff
    style EXP3 fill:#16213e,stroke:#fff,color:#fff
```

### Ventaja

- **Más capacidad sin más cómputo**: Solo un subconjunto de expertos se activa por paso
- **Especialización**: Cada experto se optimiza para un régimen de ruido específico
- **Escalado eficiente**: Se pueden añadir expertos sin aumentar la latencia de inferencia proporcionalmente

---

## 7. VLM-Coupled Architectures

### La tendencia de 2025–2026

La frontera actual acopla un **Vision-Language Model** completo con el generador de difusión/flow.

```mermaid
graph TB
    PROMPT["Prompt de texto"] --> VLM["Vision-Language Model<br/>(e.g., Mistral-3 24B)"]
    REF["Reference images (opcional)"] --> VLM

    VLM --> UNDERSTANDING["Comprensión semántica profunda:<br/>- Relaciones espaciales<br/>- Composición<br/>- Atributos<br/>- Contexto"]

    UNDERSTANDING --> COND["Rich Conditioning Signal"]
    COND --> FLOW["Flow Transformer<br/>(e.g., 32B)"]
    NOISE["z ~ N(0,I)"] --> FLOW
    FLOW --> OUTPUT["Imagen generada"]

    style VLM fill:#533483,stroke:#e94560,color:#fff
    style FLOW fill:#0f3460,stroke:#e94560,color:#fff
```

### Ejemplo: FLUX.2 (2025)

| Componente | Detalle |
|:-----------|:--------|
| VLM | Mistral-3 24B (vision-language) |
| Generador | Rectified Flow Transformer 32B |
| VAE | Reentrenada from scratch (optimizada para learnability-quality-compression) |
| Capacidad | Multi-reference editing (hasta 10 imágenes fuente) |
| Resolución | Hasta 4 megapíxeles |

> **¿Por qué acoplar un VLM?** Los text encoders anteriores (CLIP, T5) tienen limitaciones fundamentales en comprensión espacial, composicional y de conteo. Un VLM completo entiende semántica visual profunda, permitiendo generaciones mucho más fieles al prompt.

---

## 8. Tabla comparativa completa

| Arquitectura | Año | Attention | Conditioning | Params típicos | Modelos representativos |
|:-------------|:----|:----------|:-------------|:---------------|:-----------------------|
| **U-Net** | 2020 | Self (solo low-res) | Timestep via bias | 50M–900M | DDPM, SD 1.x |
| **U-Net + Cross-Attn** | 2022 | Self + Cross | CLIP text via cross-attn | 860M–2.6B | SD 1.5, SD 2.x, SDXL |
| **DiT** | 2023 | Full self-attention | AdaLN-Zero | 33M–675M | DiT, PixArt-α (600M) |
| **MMDiT** | 2024 | Joint (text+image) | CLIP + T5, joint attn | 2B–8B | SD3, SD 3.5 |
| **Rectified Flow Transformer** | 2024 | Full self + double/single stream | T5 + CLIP | 12B | FLUX.1 |
| **S3-DiT** | 2025 | Unified self-attention | Single-stream concat | 6B | Z-Image |
| **MoE-DiT** | 2025 | Full self + MoE routing | Timestep-routed experts | 5B–14B active | Wan 2.2 |
| **VLM + Flow Transformer** | 2025 | VLM attn + Flow attn | VLM (Mistral-3 24B) | 32B + 24B | FLUX.2 |
| **LLM + MMDiT** | 2026 | LLM attn + MMDiT | MLLM integrado | Variable | Qwen-Image 3.0 |

---

## 9. ¿Por qué ocurrió la transición?

### De U-Net a DiT

```mermaid
graph LR
    A["U-Net funciona bien<br/>pero no escala limpiamente"] --> B["Scaling Laws de LLMs<br/>muestran que Transformers<br/>escalan predeciblemente"]
    B --> C["DiT: ¿Y si reemplazamos<br/>U-Net por un ViT?"]
    C --> D["Resultado: FID escala<br/>con compute como en LLMs"]
    D --> E["Implicación: más grande<br/>= mejor, predeciblemente"]

    style A fill:#e94560,stroke:#fff,color:#fff
    style E fill:#0f3460,stroke:#fff,color:#fff
```

### De DiT a MMDiT

```mermaid
graph LR
    A["DiT usa cross-attention<br/>separada para texto"] --> B["Texto condiciona<br/>unidireccionalmente"]
    B --> C["Adherencia al prompt<br/>es limitada"]
    C --> D["MMDiT: joint attention<br/>texto ↔ imagen"]
    D --> E["Resultado: mejor<br/>prompt adherence,<br/>text rendering"]

    style A fill:#e94560,stroke:#fff,color:#fff
    style E fill:#0f3460,stroke:#fff,color:#fff
```

### De MMDiT a VLM-coupled

```mermaid
graph LR
    A["CLIP/T5 son buenos<br/>pero no entienden<br/>composición compleja"] --> B["VLMs entienden<br/>relaciones espaciales,<br/>conteo, atributos"]
    B --> C["Acoplar VLM como<br/>encoder semántico"]
    C --> D["Resultado: comprensión<br/>visual profunda antes<br/>de generar"]

    style A fill:#e94560,stroke:#fff,color:#fff
    style D fill:#0f3460,stroke:#fff,color:#fff
```

### Resumen: cada transición resolvió un bottleneck

| Transición | Bottleneck resuelto |
|:-----------|:-------------------|
| U-Net → DiT | Escalabilidad predecible |
| DiT → MMDiT | Adherencia al prompt |
| MMDiT → S3-DiT | Eficiencia (misma calidad, menos params) |
| DiT → MoE-DiT | Especialización por régimen de ruido |
| MMDiT → VLM-coupled | Comprensión semántica profunda |

---

## Referencias

- Ronneberger, O. et al. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation*
- Peebles, W. & Xie, S. (2023). *Scalable Diffusion Models with Transformers (DiT)*. ICCV.
- Esser, P. et al. (2024). *Scaling Rectified Flow Transformers for High-Resolution Image Synthesis (SD3)*. ICML.
- Black Forest Labs (2024). *FLUX.1 Technical Report*.
- Black Forest Labs (2025). *FLUX.2 Technical Report*.
- Alibaba/Tongyi (2025). *Wan 2.1/2.2 Technical Report*.
- Chen, J. et al. (2024). *PixArt-α: Fast Training of Diffusion Transformer*. ICLR.

---

*← [05 — Latent Diffusion](../05-latent-diffusion/README.md) | [07 — Flow Matching →](../07-flow-matching/README.md)*
