# Siddhant's Drop-Year Roadmap: Research/Frontier-Track AI Foundations
**Duration:** 52 weeks | **Track chosen:** Research/frontier (not applied-only) | **Daily commitment:** 3-4 hrs AI + 90 min gym

---

## The Real Problem to Solve First

You already told me the failure mode: motivated → productive → silently stopped → forgot what you learned. The plan below is useless if that repeats. Three non-negotiable systems before Week 1 starts:

1. **Public output, every single day.** A GitHub repo you commit to daily, even if it's 10 lines. Private effort is the easiest thing to quietly abandon — public effort has a witness (even if that witness is just future-you scrolling your own commit graph).
2. **A devlog.** One paragraph a day (Notion, a markdown file, whatever) — what you did, what confused you, what you'll do tomorrow. This is what makes Week 30 remember Week 3.
3. **Weekly review, every Sunday.** Not "what did I learn" (too vague, doesn't stick) — **"what did I ship."** If the answer is nothing, that week's plan was wrong-sized; shrink it, don't grind through it again.

Rule: **every course below must end in a project you built, not a certificate.** Certificates are a side effect, never the goal.

---

## Phase Overview

| Phase | Weeks | Focus | Exit Milestone |
|---|---|---|---|
| 0 | 1 | Systems + Python reset | Env set up, 5 scripts on GitHub |
| 1 | 2-9 | Math + NumPy/Pandas | Linear regression from scratch, real EDA project |
| 2 | 10-15 | Classical ML | Kaggle competition submitted |
| 3 | 16-23 | Deep learning foundations | NN from scratch + CNN in PyTorch + tiny GPT |
| 4 | 24-31 | NLP + Transformers + LLMs | Fine-tuned an LLM, built RAG, understand attention |
| 5 | 32-39 | Agents + applied GenAI | Deployed public AI agent/product |
| 6 | 40-47 | Research skills | 15-20 papers read, 1 reproduced, 1 OSS contribution |
| 7 | 48-52 | Portfolio + pre-college prep | Live portfolio site, plan to sustain pace in college |

---

## Phase 0 — Week 1: Reset

- **Day 1-2:** Set up GitHub, VS Code, Python (venv or conda), Jupyter. Create devlog file. Create progress tracker (spreadsheet: week / hours / shipped / stuck-on — same spirit as your gym tracker).
- **Day 3-7:** Do NOT rewatch the Python-for-AI video passively. Rebuild it from memory: write 5 small scripts covering functions, OOP basics, file I/O, list/dict comprehensions, working with a public API. If you get stuck, look up only that specific gap, then close the tab and write the code yourself.
- **Milestone:** 5 scripts pushed to GitHub. If you can't do this without heavy help, that's fine — it just means repeat this week once more before moving on.

---

## Phase 1 — Weeks 2-9: Math + NumPy/Pandas

| Week | Focus | Resource | Project/Output |
|---|---|---|---|
| 2 | Linear algebra intuition | 3Blue1Brown – *Essence of Linear Algebra* (free, YouTube) | Notes in devlog: vectors, matrices, dot product, what a transformation *is* |
| 3 | Calculus intuition | 3Blue1Brown – *Essence of Calculus* (free, YouTube) | Notes: derivatives, chain rule, gradients — this is what gradient descent runs on |
| 4-5 | Probability & statistics | StatQuest with Josh Starmer – core ML/stats playlist (free, YouTube) + Khan Academy Statistics (free, practice problems) | Explain in your own words: mean/variance, distributions, Bayes' theorem |
| 6 | NumPy | Official NumPy quickstart + freeCodeCamp NumPy tutorial (free) | **Project 1:** Linear regression from scratch — gradient descent, cost function, no sklearn |
| 7 | Pandas + Matplotlib | Kaggle micro-courses: *Pandas*, *Data Visualization* (free, ~4 hrs each) | **Project 2:** Full EDA on a real Kaggle dataset (Titanic or similar), 3 visualizations + written conclusions |
| 8 | Consolidation | — | Redo Projects 1 & 2 with zero references. If you can't, that's your real gap — fix it now, not later |
| 9 | Buffer | — | Catch up on anything behind; do NOT skip ahead with gaps unresolved |

**Exit test:** Can you explain, out loud with no notes, what a gradient descent step actually does mathematically? If no, repeat Week 3+6.

---

## Phase 2 — Weeks 10-15: Classical ML

| Week | Focus | Resource | Project/Output |
|---|---|---|---|
| 10-12 | Supervised ML properly | **Andrew Ng — Machine Learning Specialization** (Coursera/DeepLearning.AI, ~$49/mo, check financial aid) — Course 1 | Implement linear + logistic regression from scratch, then compare to sklearn |
| 13 | Trees, advanced algos | ML Specialization — Course 2 | Small decision tree / random forest project |
| 14 | Unsupervised + intro RL | ML Specialization — Course 3 (lighter week) | Notes only — this is background, not a focus area for your goals |
| 15 | Real competition | — | **Project: Kaggle Titanic or House Prices**, full pipeline, submitted, documented in devlog |

**Companion book (use throughout, not a separate week):** *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow* by Aurélien Géron — the best practical reference for this whole phase.

**Exit test:** A working Kaggle submission with a written explanation of every design choice you made.

---

## Phase 3 — Weeks 16-23: Deep Learning Foundations

| Week | Focus | Resource | Project/Output |
|---|---|---|---|
| 16-17 | Neural nets from true scratch | **Andrej Karpathy — Neural Networks: Zero to Hero** (free, YouTube) — micrograd, makemore | **Project 3:** 2-layer NN in raw NumPy on MNIST, forward + backprop by hand |
| 18 | PyTorch fundamentals | PyTorch official *60-Minute Blitz* (free) | Rebuild Project 3 in PyTorch |
| 19-20 | CNNs | fast.ai *Practical Deep Learning for Coders* Part 1 (free) — or DeepLearning.AI Deep Learning Specialization Course 4 if you prefer more structure (paid) | **Project 4:** Image classifier on CIFAR-10, real accuracy number reported |
| 21 | RNNs/sequences (light) | Brief overview only — StatQuest RNN/LSTM videos (free) | Notes only — transformers replaced most of this, don't over-invest |
| 22-23 | Building a tiny GPT | Karpathy — *Let's Build GPT from Scratch* (free, YouTube) | **Project 5:** Tiny character-level GPT trained on a small text corpus |

**Exit test:** Can you draw and explain backpropagation through a 2-layer network on paper, no help? This is the actual gatekeeper skill for everything after this phase.

---

## Phase 4 — Weeks 24-31: NLP, Transformers, LLMs

| Week | Focus | Resource | Project/Output |
|---|---|---|---|
| 24-25 | Practical NLP | **Hugging Face NLP Course** (free) | Fine-tune a pretrained model (e.g., DistilBERT) on a text classification task |
| 26 | Attention mechanism, deeply | Selected Stanford CS224n lectures on attention/transformers (free, YouTube) + read *Attention Is All You Need* (paraphrase it, don't just skim) | Devlog post: explain self-attention in your own words |
| 27-28 | Fine-tuning a real LLM | Hugging Face `transformers` + `peft` (LoRA) docs | **Project 6:** Fine-tune a small open LLM (GPT-2 or small Llama) on a custom dataset |
| 29 | RAG | DeepLearning.AI short course — *Building and Evaluating Advanced RAG* (free, ~1.5 hrs) | **Project 7:** Basic RAG pipeline over your own document set |
| 30 | Prompt engineering + eval | DeepLearning.AI short course — *ChatGPT Prompt Engineering for Developers* (free) | Write eval criteria for your RAG project, iterate on it |
| 31 | Consolidation | — | Publish a blog post explaining transformers — teaching it is the real test of understanding |

**Exit test:** Could you explain to another beginner why transformers replaced RNNs, using the actual mechanism, not just "they're better"?

---

## Phase 5 — Weeks 32-39: AI Agents + Applied GenAI

| Week | Focus | Resource | Project/Output |
|---|---|---|---|
| 32-33 | Agent frameworks | DeepLearning.AI short courses — *AI Agents in LangGraph*, *Multi AI Agent Systems with crewAI* (free, ~1-2 hrs each) | Build a simple autonomous agent (e.g., a research assistant that searches + summarizes) |
| 34-35 | Multimodal AI (intro) | CLIP paper + Hugging Face multimodal docs (free) | Small project combining text + image (e.g., image captioning or search) |
| 36-38 | **Capstone project** | — | **Project 8:** A real, deployed AI product — pick something you'd actually use. Full pipeline: data/model → backend → simple UI (Streamlit/Gradio) → hosted publicly (Hugging Face Spaces or similar) |
| 39 | Buffer/polish | — | Documentation, README, demo video/GIF for portfolio |

**Exit test:** A link you can send anyone, that works, that they can actually use.

---

## Phase 6 — Weeks 40-47: Research Skills

- **Papers:** Read 15-20 papers with written notes — start foundational (*Attention Is All You Need*, GPT-1/2/3, ResNet, CLIP), then move to 2-3 recent arXiv papers in whatever subarea pulls you (agents, multimodal, efficiency).
- **Reproduce one result.** Pick something small and scoped — not a frontier paper, a simple one — and get its core method working on a toy dataset yourself.
- **Open-source contribution.** Even a small PR (docs fix, small bug, small feature) to a real AI repo (Hugging Face, LangChain, a smaller research repo). This is disproportionately valuable for credibility later.
- **Stay current:** *The Batch* newsletter (DeepLearning.AI, free) — weekly, low time cost, keeps you oriented on what's actually moving in the field.

**Exit test:** Can you summarize a paper's core contribution in 3 sentences, without reading the abstract again?

---

## Phase 7 — Weeks 48-52: Portfolio + Pre-College Prep

- Build a simple personal portfolio site listing all 8 projects with links, demos, and what you learned building each.
- Write one long-form devlog post: the whole year, honestly — what worked, what didn't, what you'd do differently.
- Optional: reach out to a few people doing interesting AI work (researchers, small AI startups) — not for a job necessarily, just to ask questions. This habit compounds a lot over years.
- **Plan explicitly for college:** decide now how you'll protect at least 5-8 hrs/week for continued building once B.Tech starts — this pace is the asset; a mediocre curriculum will not replace it, but it also won't maintain itself once classes start eating your day.

---

## Consolidated Course List (in order of use)

| # | Resource | Cost | Used in |
|---|---|---|---|
| 1 | 3Blue1Brown — Essence of Linear Algebra & Calculus | Free | Weeks 2-3 |
| 2 | Khan Academy — Linear Algebra / Calculus / Statistics | Free | Ongoing, Weeks 2-9 |
| 3 | StatQuest (Josh Starmer) — Stats/ML playlist | Free | Weeks 4-5 |
| 4 | Kaggle Micro-courses — Pandas, Data Visualization | Free | Week 7 |
| 5 | NumPy official quickstart + freeCodeCamp NumPy | Free | Week 6 |
| 6 | Andrew Ng — Machine Learning Specialization (Coursera) | ~$49/mo (financial aid available) | Weeks 10-14 |
| 7 | *Hands-On Machine Learning* (Aurélien Géron) — book | ~$40 | Weeks 10-15 (reference) |
| 8 | Andrej Karpathy — Neural Networks: Zero to Hero | Free | Weeks 16-18, 22-23 |
| 9 | PyTorch official 60-Minute Blitz | Free | Week 18 |
| 10 | fast.ai — Practical Deep Learning for Coders | Free | Weeks 19-20 |
| 11 | Hugging Face NLP Course | Free | Weeks 24-25 |
| 12 | Stanford CS224n (selected lectures) | Free | Week 26 |
| 13 | DeepLearning.AI short courses (RAG, prompting, agents) | Free | Weeks 29-33 |

Only paid item that matters: Andrew Ng's Specialization. Everything else worth doing in this plan is free — don't buy random Udemy "AI certification" bundles; they add nothing your GitHub projects won't already prove.

---

## Suggested Daily Structure

| Block | Activity |
|---|---|
| Morning | Gym (90 min, per your existing plan) |
| Midday/Afternoon | 2-3 hr deep AI block (course + hands-on project work, no passive watching without immediately coding it) |
| Evening | 30-60 min math practice + 10 min devlog entry |
| Sunday | Weekly review: what shipped, what's next week's focus |

---

## Note on the Weight Plateau (unrelated to AI, but you asked)

Track your intake for 3-4 days before touching the training program — you said you're not tracking, and that's very likely the entire reason 55kg hasn't moved despite 2 years of training and a well-built plan.
