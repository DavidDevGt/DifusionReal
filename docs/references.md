# Bibliografía y Referencias

> Fuentes primarias para todas las afirmaciones técnicas del repositorio.

**Convención**: los identificadores arXiv se incluyen **solo cuando están verificados**. Una entrada sin enlace no es un descuido — significa que el identificador no se ha contrastado y es preferible no dar una referencia posiblemente errónea. Las referencias en **negrita** son las más influyentes en la evolución del campo.

---

## Papers fundamentales

### Diffusion Models
- Sohl-Dickstein, J. et al. (2015). *Deep Unsupervised Learning using Nonequilibrium Thermodynamics*. ICML. [arxiv:1503.03585](https://arxiv.org/abs/1503.03585) — *módulos [01](../01-foundations/), [02](../02-ddpm/)*
- Feller, W. (1949). *On the Theory of Stochastic Processes, with Particular Reference to Applications*. Berkeley Symposium. — justifica la gaussianidad del reverse process
- **Ho, J., Jain, A., & Abbeel, P. (2020). *Denoising Diffusion Probabilistic Models*. NeurIPS.** [arxiv:2006.11239](https://arxiv.org/abs/2006.11239)
- Nichol, A. & Dhariwal, P. (2021). *Improved Denoising Diffusion Probabilistic Models*. ICML. [arxiv:2102.09672](https://arxiv.org/abs/2102.09672)
- **Song, J., Meng, C., & Ermon, S. (2021). *Denoising Diffusion Implicit Models*. ICLR.** [arxiv:2010.02502](https://arxiv.org/abs/2010.02502)
- Kingma, D., Salimans, T., Poole, B., & Ho, J. (2021). *Variational Diffusion Models*. NeurIPS. [arxiv:2107.00630](https://arxiv.org/abs/2107.00630) — log-SNR; el ELBO continuo solo depende de `λ_min` y `λ_max`

> ⚠️ **Dos «Song et al. (2021)» distintos**: **Jiaming** Song firma DDIM; **Yang** Song firma Score-SDE. Es la ambigüedad de citación más frecuente en este campo.

### Score Matching
- Hyvärinen, A. (2005). *Estimation of Non-Normalized Statistical Models by Score Matching*. JMLR 6.
- Efron, B. (2011). *Tweedie's Formula and Selection Bias*. JASA. — base de la identidad score ↔ denoiser
- Vincent, P. (2011). *A Connection Between Score Matching and Denoising Autoencoders*. Neural Computation 23(7).
- Anderson, B.D.O. (1982). *Reverse-time Diffusion Equation Models*. Stochastic Processes and their Applications. — el reverse SDE
- Song, Y., Garg, S., Shi, J., & Ermon, S. (2019). *Sliced Score Matching*. UAI. [arxiv:1905.07088](https://arxiv.org/abs/1905.07088)
- **Song, Y. & Ermon, S. (2019). *Generative Modeling by Estimating Gradients of the Data Distribution*. NeurIPS.** [arxiv:1907.05600](https://arxiv.org/abs/1907.05600) — NCSN
- Song, Y. & Ermon, S. (2020). *Improved Techniques for Training Score-Based Generative Models*. NeurIPS. [arxiv:2006.09011](https://arxiv.org/abs/2006.09011) — NCSNv2
- **Song, Y., Sohl-Dickstein, J., Kingma, D., Kumar, A., Ermon, S., & Poole, B. (2021). *Score-Based Generative Modeling through Stochastic Differential Equations*. ICLR.** [arxiv:2011.13456](https://arxiv.org/abs/2011.13456)

### Flow Matching
- **Lipman, Y. et al. (2023). *Flow Matching for Generative Modeling*. ICLR.** [arxiv:2210.02747](https://arxiv.org/abs/2210.02747)
- **Liu, X., Gong, C., & Liu, Q. (2023). *Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow*. ICLR.** [arxiv:2209.03003](https://arxiv.org/abs/2209.03003)
- Albergo, M. & Vanden-Eijnden, E. (2023). *Building Normalizing Flows with Stochastic Interpolants*. ICLR.
- Chen, R.T.Q. et al. (2018). *Neural Ordinary Differential Equations*. NeurIPS. — CNF
- Holderrieth, P. et al. (2025). *An Introduction to Flow Matching and Diffusion Models*.

---

## Arquitecturas

- Ronneberger, O., Fischer, P., & Brox, T. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation*. MICCAI.
- Dhariwal, P. & Nichol, A. (2021). *Diffusion Models Beat GANs on Image Synthesis*. NeurIPS. [arxiv:2105.05233](https://arxiv.org/abs/2105.05233) — ADM; **primer modelo class-conditional** de esta familia
- **Peebles, W. & Xie, S. (2023). *Scalable Diffusion Models with Transformers*. ICCV.** [arxiv:2212.09748](https://arxiv.org/abs/2212.09748) — DiT, AdaLN-Zero, escalado por Gflops
- Chen, J. et al. (2024). *PixArt-α*. ICLR. [arxiv:2310.00426](https://arxiv.org/abs/2310.00426) — primer DiT texto-imagen relevante
- **Esser, P. et al. (2024). *Scaling Rectified Flow Transformers for High-Resolution Image Synthesis*. ICML.** [arxiv:2403.03206](https://arxiv.org/abs/2403.03206) — MMDiT, QK-norm, logit-normal sampling, resolution shift

---

## Schedules, weighting y parametrizaciones

- Salimans, T. & Ho, J. (2022). *Progressive Distillation for Fast Sampling of Diffusion Models*. ICLR. [arxiv:2202.00512](https://arxiv.org/abs/2202.00512) — **origen de v-prediction**
- Hang, T. et al. (2023). *Efficient Diffusion Training via Min-SNR Weighting Strategy*. ICCV. [arxiv:2303.09556](https://arxiv.org/abs/2303.09556)
- Chen, T. (2023). *On the Importance of Noise Scheduling for Diffusion Models*. [arxiv:2301.10972](https://arxiv.org/abs/2301.10972) — sigmoid schedule, dependencia de la resolución
- **Lin, S., Liu, B., Li, J., & Yang, X. (2024). *Common Diffusion Noise Schedules and Sample Steps are Flawed*. WACV.** [arxiv:2305.08891](https://arxiv.org/abs/2305.08891) — **zero-terminal-SNR**, CFG rescaling, leading vs trailing

---

## Samplers y Solvers

- **Lu, C. et al. (2022). *DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps*. NeurIPS.** [arxiv:2206.00927](https://arxiv.org/abs/2206.00927)
- Lu, C. et al. (2023). *DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models*. [arxiv:2211.01095](https://arxiv.org/abs/2211.01095) — estabilidad con guidance alta; variantes singlestep vs multistep
- **Karras, T., Aittala, M., Aila, T., & Laine, S. (2022). *Elucidating the Design Space of Diffusion-Based Generative Models*. NeurIPS.** [arxiv:2206.00364](https://arxiv.org/abs/2206.00364) — EDM; sigma schedule ρ=7, sampler de Heun, `S_churn`
- Zhao, W. et al. (2023). *UniPC: A Unified Predictor-Corrector Framework*. NeurIPS. [arxiv:2302.04867](https://arxiv.org/abs/2302.04867)
- Sabour, A. et al. (2024). *Align Your Steps: Optimizing Sampling Schedules in Diffusion Models*. ICML.

---

## Guidance

- Dhariwal, P. & Nichol, A. (2021). *Diffusion Models Beat GANs on Image Synthesis*. NeurIPS. [arxiv:2105.05233](https://arxiv.org/abs/2105.05233) — **classifier guidance**
- **Ho, J. & Salimans, T. (2022). *Classifier-Free Diffusion Guidance*.** [arxiv:2207.12598](https://arxiv.org/abs/2207.12598)
- Kynkäänniemi, T. et al. (2024). *Applying Guidance in a Limited Interval Improves Sample and Distribution Quality in Diffusion Models*. NeurIPS. — la guidance en ruido alto **perjudica**
- Bradley, A. & Nakkiran, P. (2024). *Classifier-Free Guidance is a Predictor-Corrector*. — **CFG no muestrea de `p(x)·p(c|x)^w`**
- Karras, T. et al. (2024). *Guiding a Diffusion Model with a Bad Version of Itself*. NeurIPS. — autoguidance
- Ahn, D. et al. (2024). *Self-Rectifying Diffusion Sampling with Perturbed-Attention Guidance*. ECCV. — PAG

---

## Conditioning y Control

- **Rombach, R. et al. (2022). *High-Resolution Image Synthesis with Latent Diffusion Models*. CVPR.** [arxiv:2112.10752](https://arxiv.org/abs/2112.10752) — latent space **y** cross-attention conditioning
- Zhang, L., Rao, A., & Agrawala, M. (2023). *Adding Conditional Control to Text-to-Image Diffusion Models*. ICCV. [arxiv:2302.05543](https://arxiv.org/abs/2302.05543) — ControlNet
- Mou, C. et al. (2024). *T2I-Adapter*. AAAI. [arxiv:2302.08453](https://arxiv.org/abs/2302.08453)
- Ye, H. et al. (2023). *IP-Adapter: Text Compatible Image Prompt Adapter*. [arxiv:2308.06721](https://arxiv.org/abs/2308.06721)
- Hu, E.J. et al. (2022). *LoRA: Low-Rank Adaptation of Large Language Models*. ICLR. [arxiv:2106.09685](https://arxiv.org/abs/2106.09685)
- Gal, R. et al. (2023). *An Image is Worth One Word: Textual Inversion*. ICLR. [arxiv:2208.01618](https://arxiv.org/abs/2208.01618)
- Ruiz, N. et al. (2023). *DreamBooth*. CVPR. [arxiv:2208.12242](https://arxiv.org/abs/2208.12242)
- Podell, D. et al. (2024). *SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis*. ICLR. [arxiv:2307.01952](https://arxiv.org/abs/2307.01952) — **micro-conditioning**

---

## Inversión y edición

- Mokady, R. et al. (2023). *Null-text Inversion for Editing Real Images using Guided Diffusion Models*. CVPR. [arxiv:2211.09794](https://arxiv.org/abs/2211.09794)
- Wallace, B., Gokul, A., & Naik, N. (2023). *EDICT: Exact Diffusion Inversion via Coupled Transformations*. CVPR. [arxiv:2211.12446](https://arxiv.org/abs/2211.12446)
- Garibi, D. et al. (2024). *ReNoise: Real Image Inversion Through Iterative Noising*. ECCV.

---

## Distillation y Eficiencia

- Salimans, T. & Ho, J. (2022). *Progressive Distillation*. ICLR. [arxiv:2202.00512](https://arxiv.org/abs/2202.00512)
- Meng, C. et al. (2023). *On Distillation of Guided Diffusion Models*. CVPR. [arxiv:2210.03142](https://arxiv.org/abs/2210.03142) — **guidance distillation**
- **Song, Y., Dhariwal, P., Chen, M., & Sutskever, I. (2023). *Consistency Models*. ICML.** [arxiv:2303.01469](https://arxiv.org/abs/2303.01469)
- Luo, S. et al. (2023). *Latent Consistency Models*. [arxiv:2310.04378](https://arxiv.org/abs/2310.04378) — LCM, LCM-LoRA
- Sauer, A. et al. (2023). *Adversarial Diffusion Distillation*. ECCV 2024. [arxiv:2311.17042](https://arxiv.org/abs/2311.17042) — ADD, SDXL-Turbo
- Sauer, A. et al. (2024). *Fast High-Resolution Image Synthesis with Latent Adversarial Diffusion Distillation*. [arxiv:2403.12015](https://arxiv.org/abs/2403.12015) — LADD
- **Yin, T. et al. (2024). *One-step Diffusion with Distribution Matching Distillation*. CVPR.** [arxiv:2311.18828](https://arxiv.org/abs/2311.18828) — DMD
- **Yin, T. et al. (2024). *Improved Distribution Matching Distillation for Fast Image Synthesis*. NeurIPS.** [arxiv:2405.14867](https://arxiv.org/abs/2405.14867) — DMD2
- Lin, S. et al. (2024). *SDXL-Lightning: Progressive Adversarial Diffusion Distillation*. [arxiv:2402.13929](https://arxiv.org/abs/2402.13929)

---

## VAE y Representación

- Kingma, D.P. & Welling, M. (2014). *Auto-Encoding Variational Bayes*. ICLR. [arxiv:1312.6114](https://arxiv.org/abs/1312.6114)
- Esser, P., Rombach, R., & Ommer, B. (2021). *Taming Transformers for High-Resolution Image Synthesis*. CVPR. — VQGAN; origen del autoencoder de LDM
- Zhang, R. et al. (2018). *The Unreasonable Effectiveness of Deep Features as a Perceptual Metric*. CVPR. [arxiv:1801.03924](https://arxiv.org/abs/1801.03924) — LPIPS
- Isola, P. et al. (2017). *Image-to-Image Translation with Conditional Adversarial Networks*. CVPR. [arxiv:1611.07004](https://arxiv.org/abs/1611.07004) — PatchGAN

---

## Modelos representativos

### Image Generation
- Black Forest Labs (2024). *FLUX.1*. [bfl.ai](https://bfl.ai) 🟡
- Black Forest Labs (2025). *FLUX.2 Technical Report*. [bfl.ai](https://bfl.ai) ⚠️
- Alibaba (2025). *Z-Image Technical Report*. [zimage.run](https://zimage.run) ⚠️
- Alibaba/Qwen (2025-2026). *Qwen-Image Series*. [qwen.ai](https://qwen.ai) ⚠️

### Video Generation
- Ho, J. et al. (2022). *Video Diffusion Models*. [arxiv:2204.03458](https://arxiv.org/abs/2204.03458)
- Blattmann, A. et al. (2023). *Align Your Latents: High-Resolution Video Synthesis with Latent Diffusion Models*. CVPR.
- **Alibaba/Tongyi Lab (2025). *Wan: Open and Advanced Large-Scale Video Generative Models*.** [arxiv:2503.20314](https://arxiv.org/abs/2503.20314) 🟡
- Lightricks (2024-2026). *LTX-Video Series*. [ltx.studio](https://ltx.studio) ⚠️

> Marcadores: ✅ contrastado · 🟡 pendiente de contraste directo · ⚠️ reconstruido de fuentes secundarias. Ver la política completa en [16 — Image Models](../16-image-models/README.md).

---

## Evaluation

- Salimans, T. et al. (2016). *Improved Techniques for Training GANs*. NeurIPS. [arxiv:1606.03498](https://arxiv.org/abs/1606.03498) — **Inception Score**
- Heusel, M. et al. (2017). *GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium*. NeurIPS. — **FID**
- Hessel, J. et al. (2021). *CLIPScore: A Reference-free Evaluation Metric for Image Captioning*.
- Huang, Z. et al. (2024). *VBench: Comprehensive Benchmark Suite for Video Generative Models*. CVPR.
- Huang, Z. et al. (2026). *VBench-2.0: Advancing Video Generation Benchmark Suite for Intrinsic Faithfulness*. ⚠️

---

## Optimización

- Dao, T. et al. (2022). *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*. NeurIPS.
- Dao, T. (2023). *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*.

---

## Índice por módulo

| Módulo | Referencias centrales |
|:-------|:----------------------|
| [01 — Fundamentos](../01-foundations/) | Sohl-Dickstein 2015, Feller 1949, Ho 2020, Kingma VDM 2021, Efron 2011, Lin 2024 |
| [02 — DDPM](../02-ddpm/) | Ho 2020, Nichol & Dhariwal 2021, Song & Ermon 2019 |
| [03 — DDIM](../03-ddim/) | **J.** Song 2021, **Y.** Song 2021, Mokady 2023, Wallace 2023 |
| [04 — Score Models](../04-score-models/) | Hyvärinen 2005, Vincent 2011, Anderson 1982, Song & Ermon 2019/2020, **Y.** Song 2021, Karras 2022 |
| [05 — Latent Diffusion](../05-latent-diffusion/) | Rombach 2022, Esser 2021 (VQGAN), Kingma & Welling 2014, LPIPS, PatchGAN |
| [06 — Arquitecturas](../06-architectures/) | Peebles & Xie 2023, Esser 2024, Chen 2024, Dhariwal & Nichol 2021 |
| [07 — Flow Matching](../07-flow-matching/) | Lipman 2023, Liu 2023, Chen 2018, Esser 2024 |
| [08 — Sampling](../08-sampling/) | Lu 2022/2023, Karras 2022, Lin 2024, Zhao 2023, Sabour 2024 |
| [09 — Guidance](../09-guidance/) | Dhariwal & Nichol 2021, Ho & Salimans 2022, Lin 2024, Kynkäänniemi 2024, Bradley & Nakkiran 2024 |
| [10 — Distillation](../10-distillation/) | Salimans & Ho 2022, Song 2023, Sauer 2023/2024, Yin 2024 ×2, Meng 2023 |
| [11 — Conditioning](../11-conditioning/) | Rombach 2022, Zhang 2023, Ye 2023, Hu 2022, Gal 2023, Ruiz 2023, Podell 2024 |
| [15 — Pipelines](../15-pipelines/) | Todas las anteriores, más los reportes técnicos |
| [16](../16-image-models/) / [17](../17-video-models/) | Reportes técnicos y model cards — ver marcadores de verificación |
| [20 — Taxonomía](../20-taxonomy/) | Índice transversal |

---

*Toda afirmación técnica del repositorio debe poder rastrearse hasta una entrada de esta lista. Si no puede, va marcada como pendiente de verificación en su módulo.*
