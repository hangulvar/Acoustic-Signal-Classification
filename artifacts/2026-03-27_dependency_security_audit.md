# 🔒 Python Dependency Security Audit
**Project:** Acoustic-Signal-Classification  
**Audit Date:** 2026-03-27  
**Tool:** pip-audit 2.10.0 (PyPI Advisory Database)  
**Status:** ✅ Remediated (12/13 CVEs fixed — 1 pending upstream fix)

---

## 📋 requirements.txt — Packages Audited

```
torch>=2.5.0         (index: download.pytorch.org/whl/cu124)
torchvision>=0.20.0
torchaudio>=2.5.0
numpy
pandas
matplotlib
seaborn
scikit-learn
librosa
soundfile
tqdm
tensorboard
ipykernel
notebook
jupyter
pytest
```

---

## ✅ Supply Chain / Poisoned Package Assessment

> The high-profile **LiteLLM supply chain attack (March 24, 2026)** by "TeamPCP" poisoned `litellm`
> versions 1.82.7–1.82.8 with credential-stealing malware. `litellm` is NOT a dependency in this project.

| Attack | Package | Status |
|--------|---------|--------|
| TeamPCP / LiteLLM (Mar 24, 2026) | `litellm` v1.82.7–1.82.8 | ✅ Not installed |
| Colorama typosquat (May 2025) | Fake `colorama` impersonators | ✅ Legitimate v0.4.6 (transitive dep) |
| Fake `numpy`/`pandas` impersonators (Aug 2025) | Typosquat packages | ✅ Official PyPI packages |
| PyTorch Pickle attack (May 2025) | Malicious PyTorch model SDKs | ✅ From official pytorch.org index |

**None of the packages in `requirements.txt` are poisoned or compromised by any known supply chain attack as of March 27, 2026.**

---

## 🛡️ CVE Remediation Summary

### Before (13 CVEs in 8 packages)

| Package | Old Version | CVE / Advisory | Fix Version |
|---------|------------|----------------|-------------|
| `filelock` | 3.13.1 | CVE-2025-68146 | 3.20.1 |
| `filelock` | 3.13.1 | CVE-2026-22701 | 3.20.3 |
| `nbconvert` | 7.16.6 | CVE-2025-53000 | 7.17.0 |
| `pillow` | 11.0.0 | CVE-2026-25990 | 12.1.1 |
| `pip` | 25.2 | CVE-2025-8869 | 25.3 |
| `pip` | 25.2 | CVE-2026-1703 | 26.0 |
| `pygments` | 2.19.2 | CVE-2026-4539 | *(no fix yet)* |
| `requests` | 2.32.5 | CVE-2026-25645 | 2.33.0 |
| `tornado` | 6.5.2 | GHSA-78cv-mqj4-43f7 | 6.5.5 |
| `tornado` | 6.5.2 | CVE-2026-31958 | 6.5.5 |
| `urllib3` | 2.5.0 | CVE-2025-66418 | 2.6.0 |
| `urllib3` | 2.5.0 | CVE-2025-66471 | 2.6.0 |
| `urllib3` | 2.5.0 | CVE-2026-21441 | 2.6.3 |

### Fix Command Run
```bash
pip install --upgrade filelock nbconvert pillow pip requests tornado urllib3
```

### After (1 remaining CVE — no upstream fix available)

| Package | Version | CVE | Status |
|---------|---------|-----|--------|
| `pygments` | 2.19.2 | CVE-2026-4539 | ⏳ Awaiting upstream patch |

> **CVE-2026-4539 in `pygments`** has no fix released yet. Monitor https://github.com/advisories
> and upgrade `pygments` when a patched version is published.

### Packages Successfully Upgraded

| Package | Old → New |
|---------|-----------|
| `filelock` | 3.13.1 → 3.25.2 |
| `nbconvert` | 7.16.6 → 7.17.0 |
| `pillow` | 11.0.0 → 12.1.1 |
| `pip` | 25.2 → 26.0.1 |
| `requests` | 2.32.5 → 2.33.0 |
| `tornado` | 6.5.2 → 6.5.5 |
| `urllib3` | 2.5.0 → 2.6.3 |

---

## 🔁 How to Re-Run This Audit

```bash
# Install pip-audit (if not already installed)
pip install pip-audit

# Audit against requirements.txt
pip-audit -r requirements.txt

# Audit the full installed environment
pip-audit --skip-editable
```

---

## 🛡️ Security Best Practices Going Forward

1. **Pin exact versions** in `requirements.txt` (e.g., `numpy==2.1.2`) to prevent unexpected version pulls.
2. **Add pip-audit to CI** — run `pip-audit -r requirements.txt` before each training run.
3. **Monitor PyPI advisories** for core packages at [osv.dev](https://osv.dev/list?ecosystem=PyPI).
4. **Watch for `pygments` fix** — upgrade once CVE-2026-4539 has a patch.
