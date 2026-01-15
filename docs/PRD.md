# LyricalLattice — Product Requirements Document (PRD)

**Version**: 2.0
**Status**: Complete
**Last Updated**: January 2026

---

## Executive Summary

LyricalLattice is an AI-powered wordplay discovery platform for songwriters, poets, and lyricists. It detects, analyzes, and suggests phonetic wordplay patterns in real-time, transforming invisible linguistic connections into visible creative opportunities.

**The Core Insight**: Puns, oronyms, and rhymes exist where language compression leaks information. LyricalLattice systematically explores these leaks to surface creative possibilities that humans miss.

---

## Part I: Problem Statement

### The Creative Friction

Artists struggle to **see** their wordplay — it's there, but invisible.

| Problem | Impact | Evidence |
|---------|--------|----------|
| Wordplay is mentally exhausting | Artists spend hours finding one good pun | User interviews |
| Phonetic relationships are invisible | "Ice cream" / "I scream" requires mental effort | Linguistic research |
| Existing tools are primitive | Rhyme dictionaries miss multi-word patterns | Competitive analysis |
| No systematic exploration | Artists rely on serendipity | Creative workflow studies |

### Root Cause Analysis

```
Why can't artists find wordplay easily?
└── Because phonetic relationships are not visually represented
    └── Because language is stored as orthography, not sound
        └── Because human memory can't hold all phoneme mappings
            └── Because the search space is combinatorially large
                └── ROOT: No tool systematically explores phoneme space
```

### The Information-Theoretic Framing

Natural language compresses meaning into sound:
- **Phoneme inventory**: ~44 sounds (English)
- **Meaning space**: Millions of concepts
- **Result**: Collisions, ambiguity, entropy

**Wordplay exploits high conditional entropy**: `H(Meaning | Sound) > 0`

A pun is not a bug — it's a feature of compression.

---

## Part II: Value Proposition

| Before LyricalLattice | After LyricalLattice |
|-----------------------|----------------------|
| "I spend hours brainstorming puns" | "I see 50 options instantly" |
| "I miss obvious connections" | "The system surfaces what I'd never find" |
| "Rhyme dictionaries are shallow" | "Multi-syllable, multi-word patterns revealed" |
| "I don't know why my lyrics work" | "I understand the phonetic mechanics" |

### The Product Promise

> **Words should reveal their hidden music.**

---

## Part III: Vision & Principles

### Vision Statement

**LyricalLattice makes the invisible music of language visible.**

### Product Principles

| Principle | Definition | Trade-off Accepted |
|-----------|------------|-------------------|
| **Scannable over complete** | Results glanceable in <2 seconds | May hide edge cases |
| **Playful over serious** | Wordplay deserves joy | May feel less "professional" |
| **Explainable over magical** | Show why, not just what | More UI complexity |
| **Fast over perfect** | 50ms response > 99% accuracy | Accept near-misses |

---

## Part IV: Target Users

### Primary Persona: Alex, 24 — The Hungry Lyricist

**Context**:
- Writing 2-3 songs per week
- Participates in rap battles / cyphers
- Values authenticity and wordplay density

**Pain Points**:
- "I know there's a better bar, but I can't find it"
- "My rhymes feel predictable"

**Success Metric**: "I found a bar I never would have thought of"

### Secondary Personas

- **Maria, 35** — Professional songwriter (Nashville, needs speed)
- **Professor Chen, 52** — Academic (teaches linguistics/poetics)
- **DJ Syntax, 28** — Battle rapper (memorizes patterns)

---

## Part V: Scope Definition

### In Scope — 21 Wordplay Devices

| Category | Devices | Count |
|----------|---------|-------|
| **Phonetic-Only** | Homophones, Oronyms, Perfect Rhyme, Slant Rhyme, Assonance, Consonance, Alliteration, Internal Rhyme, Multisyllabic Rhyme, Compound Rhyme, Onomatopoeia, Euphony/Cacophony, Stacked Rhyme | 13 |
| **Phonetic + Semantic** | Puns, Double Entendre, Malapropism, Mondegreen | 4 |
| **Music-Specific** | Polyrhythmic Rhyme, Breath Rhyme, Melisma Wordplay, Sample Flip | 4 |

### Supporting Systems

- G2P (Grapheme-to-Phoneme) conversion via ByT5
- ARPAbet reverse index (110K+ phoneme patterns)
- FAISS vector search (644K+ word embeddings)
- Real-time API (< 200ms P95)
- Web frontend (responsive, accessible)

### Out of Scope (V1)

| Feature | Reason | Future Version |
|---------|--------|----------------|
| Audio input (S2P) | Model training required | V2 |
| Mobile native apps | Web-first strategy | V2 |
| Non-English languages | Data requirements | V2 |

---

## Part VI: System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    LYRICALLATTICE PLATFORM                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐       │
│   │   WEB UI    │────▶│   API GW    │────▶│   CACHE     │       │
│   │  (Vanilla)  │     │  (FastAPI)  │     │  (Redis)    │       │
│   └─────────────┘     └──────┬──────┘     └─────────────┘       │
│                              │                                   │
│          ┌───────────────────┼───────────────────┐              │
│          ▼                   ▼                   ▼              │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐       │
│   │  G2P ENGINE │     │  DETECTION  │     │   SEARCH    │       │
│   │   (ByT5)    │     │   ENGINE    │     │   (FAISS)   │       │
│   └─────────────┘     └──────┬──────┘     └─────────────┘       │
│                              │                                   │
│          ┌───────────────────┼───────────────────┐              │
│          ▼                   ▼                   ▼              │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐       │
│   │  PHONETIC   │     │  SEMANTIC   │     │   MUSIC     │       │
│   │ DETECTORS   │     │ DETECTORS   │     │ DETECTORS   │       │
│   │    (13)     │     │    (4)      │     │    (4)      │       │
│   └─────────────┘     └─────────────┘     └─────────────┘       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Part VII: Domain Model

### Device Type Taxonomy

```
DeviceType
├── Phonetic
│   ├── HOMOPHONE       # pair/pear/pare
│   ├── ORONYM          # ice cream / I scream
│   ├── PERFECT_RHYME   # cat/hat/bat
│   ├── SLANT_RHYME     # love/move
│   ├── ASSONANCE       # hear the mellow wedding bells
│   ├── CONSONANCE      # pitter patter
│   ├── ALLITERATION    # Peter Piper picked
│   ├── INTERNAL_RHYME  # fire/retire within line
│   ├── MULTISYLLABIC   # education/dedication
│   ├── COMPOUND_RHYME  # understand it / granite
│   ├── ONOMATOPOEIA    # buzz, crash, sizzle
│   ├── EUPHONY         # cellar door (pleasant)
│   ├── CACOPHONY       # cracked crust (harsh)
│   └── STACKED_RHYME   # real/deal/steel/feel
│
├── Semantic
│   ├── PUN             # homophone + both meanings fit
│   ├── DOUBLE_ENTENDRE # innocent surface + hidden meaning
│   ├── MALAPROPISM     # "intensive purposes"
│   └── MONDEGREEN      # "kiss the sky" → "kiss this guy"
│
└── Music
    ├── POLYRHYTHMIC    # rhymes on varying beats
    ├── BREATH_RHYME    # rhymes at pause points
    ├── MELISMA         # stretched syllables
    └── SAMPLE_FLIP     # repurposed sample phonemes
```

---

## Part VIII: Theoretical Foundations

### The Pun Equation

```
PunStrength = H(M | P) - H(M | P, C)

Where:
  H = Shannon entropy
  M = meaning (random variable)
  P = phoneme sequence
  C = context
```

**Interpretation**:
- Before context: High ambiguity (multiple meanings possible)
- After context: Ambiguity collapses to one interpretation
- The entropy drop = cognitive surprise = humor

### Formal Pun Discovery Algorithm

```python
def discover_puns(phrase: str) -> list[PunMatch]:
    p0 = phonemize(phrase)
    candidates = []

    for segmentation in phonetic_segmentations(p0):
        if phonetic_distance(segmentation, p0) <= epsilon:
            if semantic_distance(segmentation, phrase) >= tau:
                if context_switchable(segmentation, phrase):
                    candidates.append(segmentation)

    return rank_by_pun_strength(candidates)
```

---

## Part IX: API Requirements

### Endpoints

| Endpoint | Method | Latency Target | Description |
|----------|--------|----------------|-------------|
| `/api/v1/detect_wordplay` | POST | < 200ms | Full 21-device detection |
| `/api/v1/suggest_rhymes` | POST | < 100ms | Rhyme suggestions |
| `/api/v1/phonemize` | POST | < 500ms | Text to phoneme conversion |
| `/api/v1/find_oronyms` | POST | < 200ms | Oronym search |
| `/api/v1/health` | GET | < 50ms | Health check |

---

## Part X: Non-Functional Requirements

### Performance

| Metric | Target |
|--------|--------|
| P50 latency | < 100ms |
| P95 latency | < 200ms |
| P99 latency | < 500ms |
| Cold start | < 5s |

### Scalability

| Metric | Target |
|--------|--------|
| Concurrent users | 10,000 |
| Vocabulary size | 500,000+ words |
| Vector index | 644,000+ embeddings |

---

## Part XI: Frontend Design System

### Colors

```css
--color-background:     #0a0a0f;   /* Deep space black */
--color-surface:        #12121a;   /* Elevated surfaces */
--accent-gradient: linear-gradient(135deg, #6366f1, #a855f7, #ec4899);

/* Device-Specific Colors */
--color-homophone:      #22d3ee;   /* Cyan */
--color-oronym:         #a855f7;   /* Purple */
--color-rhyme:          #ec4899;   /* Pink */
```

### Typography

| Element | Font | Weight | Size |
|---------|------|--------|------|
| Logo | Inter | 700 | 2.5rem |
| Body | Inter | 400 | 1rem |
| Phonemes | JetBrains Mono | 400 | 0.9rem |

---

## Part XII: Milestones

| Milestone | Status | Description |
|-----------|--------|-------------|
| M1: Core Infrastructure | ✅ Complete | API, FAISS |
| M2: G2P Engine | ✅ Complete | ByT5 + fallback |
| M3: 13 Phonetic Detectors | ✅ Complete | All implemented |
| M4: 4 Semantic Detectors | ✅ Complete | All implemented |
| M5: 4 Music Detectors | ✅ Complete | All implemented |
| M6: Unified Orchestrator | ✅ Complete | Single API |
| M7: Frontend MVP | ✅ Complete | Vanilla JS |

---

## Appendix: ARPAbet Reference

| Phoneme | Example | IPA |
|---------|---------|-----|
| AA | father | ɑ |
| AE | cat | æ |
| AH | but | ʌ |
| AY | my | aɪ |
| EH | bed | ɛ |
| EY | say | eɪ |
| IY | beat | i |
| OW | go | oʊ |
| UW | boot | u |

---

## Glossary

| Term | Definition |
|------|------------|
| **ARPAbet** | Phonetic alphabet using ASCII (e.g., "P AE1 R") |
| **G2P** | Grapheme-to-Phoneme conversion |
| **Homophone** | Words that sound identical |
| **Oronym** | Phrases with different word boundaries but same sound |
| **Mondegreen** | Misheard phrase |

---

*This PRD represents the complete specification for LyricalLattice V1.*
