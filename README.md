# 🛡️ DeepGuard — RL su IDSGame (SARSA(0) Linear FA vs DDQN)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1A7BLnOVV3CVQXSexXbMQ5sZ8weZ_6wSb)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-DDQN-ee4c2c?logo=pytorch)

## 📌 Overview
Questo progetto confronta due approcci **value-based** di Reinforcement Learning su **IDSGame** (scenario di cyber-defense):

- **SARSA(0) con Function Approximation lineare**
- **Double Deep Q-Network (DDQN)**

L’obiettivo è capire come **reward shaping** e **feature engineering** influenzino stabilità e performance, e perché un algoritmo “più potente” (DDQN) possa performare peggio se alimentato con input troppo semplificati.

---

## ✅ Cosa è stato implementato
### 1) SARSA(0) + Linear Function Approximation
- Approssimazione di Q(s,a) con pesi **W[a, :]** su feature 1D
- Training on-policy con ε-greedy
- Valutazione periodica greedy e visualizzazione dei pesi appresi (heatmap)

### 2) DDQN (Double DQN)
- **Replay Buffer**
- **Policy network** e **Target network**
- Aggiornamento DDQN (azione migliore dalla policy, valore dalla target)
- Rete neurale MLP semplice: **Linear → ReLU → Linear → ReLU → Linear**

### 3) IDSGameWrapper (Gym API compatibility)
Ho introdotto un wrapper dedicato per gestire differenze tra API vecchie/nuove di Gym e normalizzare:
- `reset()` / `step()`
- conversione dello stato in `np.ndarray` coerente

### 4) Reward Shaping
In ambienti con reward sparsi, il reward shaping rende l’apprendimento più stabile:
- penalità forte se `hacked`
- reward positivo se “safe”
- reward denso legato alla variazione di vulnerabilità nello stato

---

## 🧠 Feature engineering
Nel notebook viene usata `extract_state_features(state)` che trasforma lo stato 2D (4×5) in **11 feature**:
- 4 medie per nodo (righe)
- 5 medie per attributo (colonne)
- totale medio + bias

✅ Questo è **molto efficace** per SARSA lineare (feature “buone” e compatte).  
⚠️ Ma per DDQN può essere un collo di bottiglia: la rete neurale riceve input già “pre-digerito” e perde informazione spaziale/topologica.

---

## 📊 Risultati
| Metodo | Reward medio / valutazione | Note |
|---|---:|---|
| Baseline | ~ **-100** | comportamento scarso |
| **SARSA-FA** | **32.70** | miglioramento netto: feature lineari efficaci |
| **DDQN** | ~ **-21** | meglio del baseline, ma molto sotto SARSA-FA |

### Interpretazione
Il problema non sembra l’algoritmo (DDQN è teoricamente superiore), ma **l’implementazione specifica**:
- rete troppo basilare
- input 1D troppo aggregato → la rete non può imparare rappresentazioni complesse

---

## 🔮 Roadmap
### 1) DDQN con input grezzo 2D + CNN (step logico successivo)
- usare lo stato intero `(4,5)` come input
- sostituire MLP con CNN (stile Nature DQN) per catturare struttura spaziale

### 2) Refactor code-quality
- rimuovere dipendenza da variabili globali in DDQN (`LR`, `MEMORY_SIZE`, ecc.)
  - passare hyperparams nel costruttore o in un oggetto `Config` (dataclass)
- spezzare `sarsa_linear_fa_train`:
  - creare classe `SARSALinearFAAgent`
  - separare update pesi, selection policy, evaluation loop

---
