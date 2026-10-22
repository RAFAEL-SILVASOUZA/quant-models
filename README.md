# quant-models

Baixa um modelo do HuggingFace e gera GGUFs só com as quants que cabem no
hardware (2x 24 GB + RAM para experts de MoE).

## Uso

```
quant.bat https://huggingface.co/nex-agi/Nex-N2.5-mini
```

Sem argumento, ele pergunta a URL. Aceita também `dono/nome` puro.
`--list-all` mostra todas as quants no menu (a lista curada cobre IQ2_XS..Q8_0
+ MXFP4_MOE para MoE).

Fluxo: download retomavel -> conversao F16 (cacheada) -> menu com tamanho real
de cada quant (`llama-quantize --dry-run`) e veredito de caber -> voce escolhe
por numero -> imatrix (cacheado por modelo) -> quantize. Rodar de novo com a
mesma URL nao rebaixa nem reconverte; o menu reabre com as quants ja geradas
marcadas `[criada]`.

## Layout

- `models/<dono>__<nome>/` — pesos HF (safetensors etc.)
- `gguf/<dono>__<nome>/` — F16, mmproj (se visao), `.imatrix.gguf`, e os GGUFs finais
- `calib/` — corpus de calibracao do imatrix (compartilhado entre modelos).
  Coloque `user-*.txt` aqui para calibrar com texto proprio.
- `config.json` — hardware e tunagem (VRAM por GPU, RAM, folga de KV, chunks
  do imatrix, threads).

## Notas

- Repo protegido (gated): faca login antes — `set HF_TOKEN=hf_...` ou `huggingface-cli login`.
- O build upstream do llama.cpp nao tem as quants XL da Unsloth (Q4_K_XL etc.);
  use Q4_K_M/Q5_K_M como equivalentes mais proximos, e MXFP4_MOE para MoE.
- GGUF final unico; para shardar: `llama-gguf-split --split maxsize:<GiB> arquivo.gguf saida`.
- Se o download Gutenberg falhar (proxy SSL), o imatrix segue so com TinyStories
  (~18 MB) — funciona; para corpus melhor, use `calib/user-*.txt`.
- Modelos com visao geram `mmproj-F16.gguf` junto.
