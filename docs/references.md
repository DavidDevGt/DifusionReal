# Bibliografía y Referencias

> Fuentes primarias para todas las afirmaciones técnicas del repositorio.

---

## Papers fundamentales

### Diffusion Models
- Sohl-Dickstein, J. et al. (2015). *Deep Unsupervised Learning using Nonequilibrium Thermodynamics*. ICML. [arxiv:1503.03585](https://arxiv.org/abs/1503.03585)
- **Ho, J., Jain, A., & Abbeel, P. (2020). *Denoising Diffusion Probabilistic Models*. NeurIPS.** [arxiv:2006.11239](https://arxiv.org/abs/2006.11239)
- Nichol, A. & Dhariwal, P. (2021). *Improved Denoising Diffusion Probabilistic Models*. ICML. [arxiv:2102.09672](https://arxiv.org/abs/2102.09672)
- **Song, J., Meng, C., & Ermon, S. (2021). *Denoising Diffusion Implicit Models*. ICLR.** [arxiv:2010.02502](https://arxiv.org/abs/2010.02502)

### Score Matching
- Hyvärinen, A. (2005). *Estimation of Non-Normalized Statistical Models by Score Matching*. JMLR.
- Vincent, P. (2011). *A Connection Between Score Matching and Denoising Autoencoders*. Neural Computation.
- **Song, Y. & Ermon, S. (2019). *Generative Modeling by Estimating Gradients of the Data Distribution*. NeurIPS.** [arxiv:1907.05600](https://arxiv.org/abs/1907.05600)
- **Song, Y. et al. (2021). *Score-Based Generative Modeling through Stochastic Differential Equations*. ICLR.** [arxiv:2011.13456](https://arxiv.org/abs/2011.13456)

### Flow Matching
- **Lipman, Y. et al. (2023). *Flow Matching for Generative Modeling*. ICLR.** [arxiv:2210.02747](https://arxiv.org/abs/2210.02747)
- **Liu, X. et al. (2023). *Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow*. ICLR.** [arxiv:2209.03003](https://arxiv.org/abs/2209.03003)
- Albergo, M. & Vanden-Eijnden, E. (2023). *Building Normalizing Flows with Stochastic Interpolants*. ICLR.
- Holderrieth, P. et al. (2025). *An Introduction to Flow Matching and Diffusion Models*.

---

## Arquitecturas

### U-Net
- Ronneberger, O. et al. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation*.

### DiT
- **Peebles, W. & Xie, S. (2023). *Scalable Diffusion Models with Transformers*. ICCV.** [arxiv:2212.09748](https://arxiv.org/abs/2212.09748)

### MMDiT
- **Esser, P. et al. (2024). *Scaling Rectified Flow Transformers for High-Resolution Image Synthesis*. ICML.** [arxiv:2403.03206](https://arxiv.org/abs/2403.03206)

---

## Modelos representativos

### Image Generation
- **Rombach, R. et al. (2022). *High-Resolution Image Synthesis with Latent Diffusion Models*. CVPR.** [arxiv:2112.10752](https://arxiv.org/abs/2112.10752)
- Podell, D. et al. (2024). *SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis*. [arxiv:2307.01952](https://arxiv.org/abs/2307.01952)
- Chen, J. et al. (2024). *PixArt-α: Fast Training of Diffusion Transformer for Photorealistic Text-to-Image Synthesis*. ICLR. [arxiv:2310.00426](https://arxiv.org/abs/2310.00426)
- Black Forest Labs (2024). *FLUX.1 Technical Report*. [bfl.ai](https://bfl.ai)
- Black Forest Labs (2025). *FLUX.2 Technical Report*. [bfl.ai](https://bfl.ai)
- Alibaba (2025). *Z-Image Technical Report*. [zimage.run](https://zimage.run)
- Alibaba/Qwen (2025-2026). *Qwen-Image Series*. [qwen.ai](https://qwen.ai)

### Video Generation
- Ho, J. et al. (2022). *Video Diffusion Models*. [arxiv:2204.03458](https://arxiv.org/abs/2204.03458)
- Blattmann, A. et al. (2023). *Align Your Latents: High-Resolution Video Synthesis with Latent Diffusion Models*. CVPR.
- **Alibaba/Tongyi Lab (2025). *Wan: Open and Advanced Large-Scale Video Generative Models*.** [arxiv:2503.20314](https://arxiv.org/abs/2503.20314), [wan.video](https://wan.video)
- Lightricks (2024-2026). *LTX-Video Series*. [ltx.studio](https://ltx.studio)

---

## Guidance y Conditioning
- **Ho, J. & Salimans, T. (2022). *Classifier-Free Diffusion Guidance*. NeurIPS Workshop.** [arxiv:2207.12598](https://arxiv.org/abs/2207.12598)
- Dhariwal, P. & Nichol, A. (2021). *Diffusion Models Beat GANs on Image Synthesis*. NeurIPS. [arxiv:2105.05233](https://arxiv.org/abs/2105.05233)
- Zhang, L. et al. (2023). *Adding Conditional Control to Text-to-Image Diffusion Models (ControlNet)*. ICCV. [arxiv:2302.05543](https://arxiv.org/abs/2302.05543)
- Ye, H. et al. (2023). *IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models*. [arxiv:2308.06721](https://arxiv.org/abs/2308.06721)
- Hu, E.J. et al. (2022). *LoRA: Low-Rank Adaptation of Large Language Models*. ICLR. [arxiv:2106.09685](https://arxiv.org/abs/2106.09685)

---

## Distillation y Eficiencia
- Salimans, T. & Ho, J. (2022). *Progressive Distillation for Fast Sampling of Diffusion Models*. ICLR. [arxiv:2202.00512](https://arxiv.org/abs/2202.00512)
- **Song, Y. et al. (2023). *Consistency Models*. ICML.** [arxiv:2303.01469](https://arxiv.org/abs/2303.01469)
- Sauer, A. et al. (2023). *Adversarial Diffusion Distillation*. [arxiv:2311.17042](https://arxiv.org/abs/2311.17042)
- Luo, S. et al. (2023). *Latent Consistency Models*. [arxiv:2310.04378](https://arxiv.org/abs/2310.04378)

---

## Samplers y Solvers
- **Lu, C. et al. (2022). *DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling*. NeurIPS.** [arxiv:2206.00927](https://arxiv.org/abs/2206.00927)
- Lu, C. et al. (2023). *DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models*.
- Karras, T. et al. (2022). *Elucidating the Design Space of Diffusion-Based Generative Models*. NeurIPS. [arxiv:2206.00364](https://arxiv.org/abs/2206.00364)

---

## Training
- Hang, T. et al. (2023). *Efficient Diffusion Training via Min-SNR Weighting Strategy*. ICCV.
- Lin, S. et al. (2024). *Common Diffusion Noise Schedules and Sample Steps are Flawed*. WACV.

---

## Evaluation
- Heusel, M. et al. (2017). *GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium (FID)*. NeurIPS.
- Hessel, J. et al. (2021). *CLIPScore: A Reference-free Evaluation Metric for Image Captioning*.
- Huang, Z. et al. (2024). *VBench: Comprehensive Benchmark Suite for Video Generative Models*. CVPR.
- Huang, Z. et al. (2026). *VBench-2.0: Advancing Video Generation Benchmark Suite for Intrinsic Faithfulness*.

---

## VAE y Representación
- Kingma, D.P. & Welling, M. (2014). *Auto-Encoding Variational Bayes*. ICLR. [arxiv:1312.6114](https://arxiv.org/abs/1312.6114)
- Esser, P. et al. (2021). *Taming Transformers for High-Resolution Image Synthesis*. CVPR.
- Chen, R.T.Q. et al. (2018). *Neural Ordinary Differential Equations*. NeurIPS.

---

## Optimización
- Dao, T. et al. (2022). *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*. NeurIPS.
- Dao, T. (2023). *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*.

---

*Las referencias marcadas en **negrita** son las más influyentes en la evolución del campo.*
