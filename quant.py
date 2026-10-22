#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
quant.py — baixa um modelo do HuggingFace e gera GGUFs sob medida para o hardware.

Uso:
    quant.bat <url-ou-repo-id> [--list-all]

Fluxo: download retomavel -> conversao F16 (em cache) -> menu de quants com
tamanho real (--dry-run) e veredito de caber -> voce escolhe -> imatrix
(cacheado) -> llama-quantize. Rodar de novo com a mesma URL nao rebaixa nem
reconverte; o menu reabre com as quants ja geradas marcadas.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
MODELS = BASE / "models"
GGUF_DIR = BASE / "gguf"
CALIB = BASE / "calib"

CFG_DEFAULTS = {
    "llama_dir": "C:/Projetos/llama-cpp",
    "bin": "build/bin/Release",
    "convert_script": "convert_hf_to_gguf.py",
    "vram_mib_per_gpu": [24576, 24576],
    "single_gpu_mib": 24576,
    "ram_gb": 64,
    "kv_reserve_mib": 3072,
    "imatrix_chunks": 100,
    "imatrix_ngl": 999,
    "quant_threads": None,
}

# Lista curada (ordem crescente de tamanho aproximado); MXFP4_MOE so para MoE
QUANTS_CURATED = [
    "IQ2_XS", "IQ2_S", "IQ3_XXS", "IQ3_S", "Q2_K", "Q3_K_M", "Q3_K_L",
    "Q4_K_S", "Q4_K_M", "IQ4_XS", "Q5_K_M", "Q6_K", "Q8_0",
]
QUANTS_MOE_EXTRA = ["MXFP4_MOE"]
# Quants que nao usam imatrix (nao ha ganho/nao e suportado)
QUANTS_SEM_IMATRIX = {"Q8_0", "F16", "BF16", "F32", "COPY", "MXFP4_MOE"}

# Corpus de calibracao padrao (baixado uma vez, fica em calib/)
CALIB_TXT = [
    # (nome do arquivo, URL)
    ("tinystories-valid.txt", None),  # via huggingface_hub (dataset abaixo)
    ("pg1342.txt", "https://www.gutenberg.org/cache/epub/1342/pg1342.txt"),
    ("pg27013.txt", "https://www.gutenberg.org/cache/epub/27013/pg27013.txt"),
    ("pg84.txt", "https://www.gutenberg.org/cache/epub/84/pg84.txt"),
    ("pg996.txt", "https://www.gutenberg.org/cache/epub/996/pg996.txt"),
]
CALIB_DATASET = ("roneneldan/TinyStories", "TinyStories-valid.txt")

HF_PATTERNS = [
    "*.safetensors", "*.safetensors.index.json",
    "config.json", "generation_config.json",
    "tokenizer.json", "tokenizer.model", "tokenizer_config.json",
    "vocab.json", "vocab.bin", "merges.txt", "merges.txt.gz",
    "special_tokens_map.json", "added_tokens.json",
    "chat_template.jinja", "chat_template.json",
    "preprocessor_config.json", "video_preprocessor_config.json",
    "processor_config.json",
]


def load_cfg():
    cfg = dict(CFG_DEFAULTS)
    f = BASE / "config.json"
    if f.is_file():
        try:
            user = json.loads(f.read_text(encoding="utf-8"))
            user.pop("_comentario", None)
            cfg.update(user)
        except Exception as e:
            print(f"[aviso] config.json invalido, usando padroes: {e}")
    return cfg


def exe_dir(cfg):
    d = Path(cfg["llama_dir"]) / cfg["bin"]
    if not d.is_dir():
        sys.exit(f"[erro] binarios llama.cpp nao encontrados em {d} (edite config.json)")
    return d


def exe(cfg, name):
    p = exe_dir(cfg) / (name + ".exe")
    if not p.is_file():
        sys.exit(f"[erro] {p} nao existe")
    return str(p)


def repo_from_url(s):
    s = s.strip().rstrip("/")
    m = re.search(r"huggingface\.co/([^/\s]+/[^/\s]+)", s)
    if m:
        return m.group(1)
    if "/" in s and not s.startswith("http"):
        return s
    sys.exit(f"[erro] nao entendi a URL/repo: {s}")


def safe_name(repo):
    return repo.replace("/", "__")


def is_moe(model_dir):
    for cf in ["config.json", "text_config.json"]:
        f = model_dir / cf
        if not f.is_file():
            continue
        try:
            txt = f.read_text(encoding="utf-8")
        except Exception:
            continue
        if re.search(r'"(num_experts|n_routed_experts|num_local_experts|moe_intermediate_size)"\s*:\s*[1-9]', txt):
            return True
    return False


def has_vision(model_dir):
    f = model_dir / "config.json"
    if not f.is_file():
        return False
    txt = f.read_text(encoding="utf-8")
    return bool(re.search(r'"(vision_config|image_token_id)"', txt)) or "ConditionalGeneration" in txt


# ---------------------------------------------------------------- download

def download(repo, model_dir):
    from huggingface_hub import snapshot_download
    MODELS.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Download de {repo} para {model_dir} (retomavel) ===")
    t0 = time.time()
    kw = dict(repo_id=repo, local_dir=str(model_dir), allow_patterns=HF_PATTERNS,
              max_workers=8)
    try:
        snapshot_download(**kw)
    except Exception as e:
        print(f"[aviso] primeira passada falhou ({e}); tentando incluir .bin")
        snapshot_download(repo_id=repo, local_dir=str(model_dir),
                          allow_patterns=HF_PATTERNS + ["*.bin", "*.bin.index.json"],
                          max_workers=8)
    if not list(model_dir.rglob("*.safetensors")) and not list(model_dir.rglob("*.bin")):
        sys.exit("[erro] nenhum peso baixado (repo protegido? defina HF_TOKEN e rode de novo)")
    print(f"[ok] pesos em {model_dir} ({time.time()-t0:.0f}s)")


def fix_tokenizer_config(model_dir):
    """Repos novos declaram tokenizer_class 'TokenizersBackend' (transformers 5.x);
    o transformers local nao conhece essa classe. PreTrainedTokenizerFast le o
    mesmo tokenizer.json e funciona na conversao."""
    f = model_dir / "tokenizer_config.json"
    if not f.is_file():
        return
    txt = f.read_text(encoding="utf-8")
    if "TokenizersBackend" in txt:
        f.write_text(txt.replace("TokenizersBackend", "PreTrainedTokenizerFast"), encoding="utf-8")
        print("[fix] tokenizer_class TokenizersBackend -> PreTrainedTokenizerFast")


# ---------------------------------------------------------------- conversao

def convert(model_dir, gdir, name):
    f16 = gdir / f"{name}-F16.gguf"
    if f16.is_file():
        print(f"\n[skip] F16 ja existe: {f16.name}")
        return f16
    gdir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Conversao para F16 (demora; fica em cache) ===")
    t0 = time.time()
    tmp = f16.with_name(f16.name + ".tmp")  # so vira F16 oficial quando completar
    cmd = [sys.executable, _CONV_SCRIPT, str(model_dir), "--outtype", "f16", "--outfile", str(tmp)]
    r = subprocess.run(cmd)
    if r.returncode != 0:
        tmp.unlink(missing_ok=True)
        sys.exit("[erro] conversao falhou (arquitetura nao suportada? atualize o llama.cpp e rode 'git pull' la)")
    os.replace(tmp, f16)
    print(f"[ok] {f16.name} ({f16.stat().st_size/2**30:.1f} GiB, {time.time()-t0:.0f}s)")
    return f16


_CONV_SCRIPT = None


def convert_mmproj(model_dir, gdir):
    mm = gdir / "mmproj-F16.gguf"
    if mm.is_file():
        print(f"[skip] mmproj ja existe: {mm.name}")
        return mm
    print("\n=== Conversao do projector de visao (mmproj) ===")
    t0 = time.time()
    tmp = mm.with_name(mm.name + ".tmp")
    cmd = [sys.executable, _CONV_SCRIPT, str(model_dir), "--mmproj",
           "--outtype", "f16", "--outfile", str(tmp)]
    r = subprocess.run(cmd)
    if r.returncode != 0:
        tmp.unlink(missing_ok=True)
        print("[aviso] mmproj falhou; seguindo sem visao")
        return None
    os.replace(tmp, mm)
    print(f"[ok] {mm.name} ({time.time()-t0:.0f}s)")
    return mm


# ---------------------------------------------------------------- dry-run

def dry_run_sizes(cfg, f16, quants):
    """Retorna {quant: size_GiB} usando llama-quantize --dry-run (cacheado)."""
    cache = f16.parent / "sizes.json"
    key = f"{f16.stat().st_size}-{int(f16.stat().st_mtime)}"
    data = {}
    if cache.is_file():
        try:
            data = json.loads(cache.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    if data.get("_key") != key:
        data = {"_key": key}
    out = {}
    qe = exe(cfg, "llama-quantize")
    for q in quants:
        if q in data:
            out[q] = data[q]
            continue
        r = subprocess.run([qe, "--dry-run", str(f16), q],
                           capture_output=True, text=True, errors="replace")
        m = re.search(r"quant size\s*=\s*([\d.,]+)\s*MiB", r.stdout + r.stderr)
        if not m:
            print(f"[aviso] dry-run de {q} nao reportou tamanho")
            continue
        gib = float(m.group(1).replace(",", "")) / 1024
        data[q] = round(gib, 2)
        out[q] = data[q]
    cache.write_text(json.dumps(data, indent=1), encoding="utf-8")
    return out


def veredito(cfg, gib, moe):
    mib = gib * 1024
    kv = cfg["kv_reserve_mib"]
    single = cfg["single_gpu_mib"]
    total = sum(cfg["vram_mib_per_gpu"])
    ram_budget = cfg["ram_gb"] * 1024 * 0.8
    if mib + kv <= single:
        return "cabe em 1 GPU"
    if mib + kv <= total:
        return "cabe nas 2 GPUs (split)"
    if moe and mib + kv <= total + ram_budget:
        return "experts na RAM (MoE)"
    return "NAO CABE confortavelmente"


# ---------------------------------------------------------------- imatrix

def ensure_corpus():
    CALIB.mkdir(parents=True, exist_ok=True)
    corpus = CALIB / "corpus.txt"
    partes = []
    # arquivos do usuario tem prioridade
    for f in sorted(CALIB.glob("user-*.txt")):
        partes.append(f)
    need_dl = []
    for nome, url in CALIB_TXT:
        f = CALIB / nome
        if not f.is_file():
            need_dl.append((nome, url))
    if need_dl:
        print("\n=== Baixando corpus de calibracao (uma vez) ===")
        from huggingface_hub import hf_hub_download
        for nome, url in need_dl:
            f = CALIB / nome
            try:
                if url is None:
                    ds, fn = CALIB_DATASET
                    src = Path(hf_hub_download(repo_id=ds, filename=fn, repo_type="dataset",
                                              local_dir=str(CALIB / "_hf")))
                    shutil.move(str(src), f)
                else:
                    import urllib.request
                    urllib.request.urlretrieve(url, f)
                print(f"[ok] {nome} ({f.stat().st_size/2**20:.1f} MiB)")
            except Exception as e:
                print(f"[aviso] {nome}: {e}")
    partes += [CALIB / n for n, _ in CALIB_TXT if (CALIB / n).is_file()]
    if not partes:
        sys.exit("[erro] corpus de calibracao vazio; coloque .txt em calib/ (prefixo user-)")
    if not corpus.is_file() or corpus.stat().st_size < sum(p.stat().st_size for p in partes) * 0.9:
        print("Montando calib/corpus.txt ...")
        with open(corpus, "w", encoding="utf-8", errors="replace") as w:
            for p in partes:
                w.write(p.read_text(encoding="utf-8", errors="replace"))
                w.write("\n\n")
    return corpus


def ensure_imatrix(cfg, f16, name):
    imat = f16.parent / f"{name}.imatrix.gguf"
    if imat.is_file():
        print(f"[skip] imatrix ja existe: {imat.name}")
        return imat
    corpus = ensure_corpus()
    print("\n=== Calculando imatrix (demora uma vez por modelo) ===")
    t0 = time.time()
    cmd = [exe(cfg, "llama-imatrix"), "-m", str(f16), "-f", str(corpus),
           "-o", str(imat), "--chunks", str(cfg["imatrix_chunks"]),
           "-ngl", str(cfg["imatrix_ngl"])]
    r = subprocess.run(cmd)
    if r.returncode != 0 or not imat.is_file():
        print("[aviso] imatrix falhou; quants IQ podem ficar ruins. Tente reduzir imatrix_ngl/imatrix_chunks no config.json")
        return None
    print(f"[ok] {imat.name} ({time.time()-t0:.0f}s)")
    return imat


# ---------------------------------------------------------------- quantize

def quantize(cfg, f16, gdir, name, q, imat):
    out = gdir / f"{name}-{q}.gguf"
    if out.is_file():
        print(f"[skip] {out.name} ja existe")
        return out
    # neste build, flags vem antes dos posicionais; threads e o 4o posicional
    cmd = [exe(cfg, "llama-quantize")]
    if imat and q not in QUANTS_SEM_IMATRIX:
        cmd += ["--imatrix", str(imat)]
    cmd += [str(f16), str(out), q]
    if cfg["quant_threads"]:
        cmd.append(str(cfg["quant_threads"]))
    print(f"\n=== Quantizando {q} ===")
    t0 = time.time()
    r = subprocess.run(cmd)
    if r.returncode != 0 or not out.is_file():
        print(f"[erro] {q} falhou")
        out.unlink(missing_ok=True)
        return None
    print(f"[ok] {out.name}: {out.stat().st_size/2**30:.1f} GiB ({time.time()-t0:.0f}s)")
    return out


# ---------------------------------------------------------------- main

def main():
    global _CONV_SCRIPT
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(add_help=True, description=__doc__)
    ap.add_argument("url", nargs="?", help="URL do HuggingFace ou repo_id (dono/nome)")
    ap.add_argument("--list-all", action="store_true", help="mostrar todas as quants no menu")
    args = ap.parse_args()

    cfg = load_cfg()
    repo = repo_from_url(args.url) if args.url else repo_from_url(
        input("Cole a URL do modelo no HuggingFace: "))
    name = repo.split("/")[-1]
    model_dir = MODELS / safe_name(repo)
    gdir = GGUF_DIR / safe_name(repo)
    gdir.mkdir(parents=True, exist_ok=True)

    conv = Path(cfg["llama_dir"]) / cfg["convert_script"]
    if not conv.is_file():
        sys.exit(f"[erro] {conv} nao existe")
    _CONV_SCRIPT = str(conv)

    print(f"Modelo: {repo}")
    download(repo, model_dir)
    fix_tokenizer_config(model_dir)
    moe = is_moe(model_dir)
    print(f"Tipo: {'MoE' if moe else 'denso'}" + (" + visao" if has_vision(model_dir) else ""))

    f16 = convert(model_dir, gdir, name)
    if has_vision(model_dir):
        convert_mmproj(model_dir, gdir)

    quants = list(QUANTS_CURATED)
    if moe:
        quants += QUANTS_MOE_EXTRA
    if args.list_all:
        quants = ["IQ1_S", "IQ1_M", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ2_M",
                  "Q2_K", "Q2_K_S", "IQ3_XXS", "IQ3_XS", "IQ3_S", "IQ3_M",
                  "Q3_K_S", "Q3_K_M", "Q3_K_L", "IQ4_NL", "IQ4_XS",
                  "Q4_K_S", "Q4_K_M", "Q5_K_S", "Q5_K_M", "Q6_K"]
        if moe:
            quants += ["MXFP4_MOE"]
        quants += ["Q8_0"]

    print("\nCalculando tamanhos reais (--dry-run) ...")
    sizes = dry_run_sizes(cfg, f16, quants)
    if not sizes:
        sys.exit(f"[erro] nenhum dry-run funcionou em {f16.name}. O F16 pode estar "
                 "incompleto/corrompido (conversao interrompida?). Apague o arquivo e rode de novo.")

    rows = []
    for q in quants:
        if q not in sizes:
            continue
        out = gdir / f"{name}-{q}.gguf"
        rows.append((q, sizes[q], veredito(cfg, sizes[q], moe), out.is_file()))
    rows.sort(key=lambda r: r[1])

    print(f"\n{'#':>3}  {'QUANT':<10} {'GiB':>7}  {'veredito':<28} status")
    print("-" * 72)
    for i, (q, gb, v, feito) in enumerate(rows, 1):
        mark = "[criada]" if feito else ""
        print(f"{i:>3}  {q:<10} {gb:>7.2f}  {v:<28} {mark}")
    print("-" * 72)
    print("Escolha numeros separados por virgula (ex: 4,7). 'q' para sair.")

    picks = input("> ").strip().lower()
    if picks in ("", "q", "quit", "sair"):
        print("Nada a fazer.")
        return
    idxs = []
    for tok in re.split(r"[,\s;]+", picks):
        if tok.isdigit() and 1 <= int(tok) <= len(rows):
            idxs.append(int(tok) - 1)
    if not idxs:
        print("Entrada nao entendida.")
        return

    escolhidas = [rows[i][0] for i in idxs if not rows[i][3]]
    ja = [rows[i][0] for i in idxs if rows[i][3]]
    for q in ja:
        print(f"[skip] {q} ja foi gerada")
    if not escolhidas:
        print("Todas as escolhidas ja existem.")
        return

    ruim = [rows[i] for i in idxs if "NAO CABE" in rows[i][2]]
    if ruim:
        for q, gb, v, _ in ruim:
            print(f"[atencao] {q} ({gb:.1f} GiB): {v}")
        if input("Gerar mesmo assim? (s/N) ").strip().lower() not in ("s", "sim", "y"):
            return

    imat = None
    if any(q not in QUANTS_SEM_IMATRIX for q in escolhidas):
        imat = ensure_imatrix(cfg, f16, name)

    ok, fail = [], []
    for q in escolhidas:
        r = quantize(cfg, f16, gdir, name, q, imat)
        (ok if r else fail).append(q)
    print(f"\nResumo: geradas {ok or '[]'}" + (f"; falharam {fail}" if fail else ""))
    print(f"Saida: {gdir}")
    if any((gdir / f"{name}-{q}.gguf").is_file() and (gdir / f"{name}-{q}.gguf").stat().st_size > 48 * 2**30
           for q in ok):
        print("Dica: para shardar em varios arquivos, use llama-gguf-split --split maxsize")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrompido. Rode de novo para retomar de onde parou.")
