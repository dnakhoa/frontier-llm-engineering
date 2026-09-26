# Staleness audit, Part E: Chapters 17–19, 22–26

Audit date: 2026-09-26. Read-only; no repo files were edited. Every finding below cites a primary source I opened during this audit (arXiv abstract or HTML, official lab blog, official repo, Nature article and its Supplementary Information, or an official job posting). Anything I could not confirm from a primary source is listed under "Unverified leads" at the end.

Paths are relative to the repo root: `.../worktrees/book-content-outdated-99be26/`.

**Cross-chapter headline.** Three developments cut across these chapters:
1. **The R1 report was superseded by its own peer-reviewed version.** *Nature*, 17 Sep 2025, with a 70+ page Supplementary. It discloses most of what Ch 19 §19.8 says is withheld, including the RL prompt counts, GRPO hyperparameters, compute and cost.
2. **Agentic RL environments became a published engineering discipline.** Kimi K2, GLM-4.5, Qwen3-Coder, DeepSeek-V3.2 and DeepSeek-V4 all describe environment and sandbox infrastructure at scale. That falsifies the book's repeated prediction that environments would be "less published than model architecture" (Ch 18 §18.10, Ch 26 §26.3).
3. **SWE-bench Verified was retired as a frontier metric** by the organisation that created it (OpenAI, 23 Feb 2026). Ch 18 calls it "the one to quote".

---

## Ch 17 — Constitutional AI, RLAIF, deliberative alignment

**Verdict:** Needs rewrite of §17.5; light touch elsewhere (§17.7, §17.9, §17.10).

### Findings (ranked most-severe first)

1. **WRONG-NOW** · `book/part-3-post-training/17-constitutional-ai.md:115-133` · "Anthropic published Claude's constitution… The principles are drawn from sources including the UN Declaration of Human Rights…" and "**They are comparative.** 'Choose the response that is *less* X'…". This describes the 2023 list-of-principles constitution as if it were current. On 22 Jan 2026 Anthropic replaced it with a new constitution that deliberately abandons the list-of-standalone-principles form. The announcement says the old one "was composed of a list of standalone principles. We've come to believe that a different approach is necessary". The new document explains *why* rather than specifying *what*. It sets a four-level priority ordering (broadly safe > broadly ethical > compliant with Anthropic's guidelines > genuinely helpful) and keeps a small set of "hard constraints" as bright lines. Anthropic calls it "the final authority", used "at various stages of the training process", and it is released CC0. The three properties the section extracts (comparative, overlapping, cite their sources) describe the 2023 artifact only. The section's engineering point, "the constitution is a file in version control" (line 133), is now literally true for both labs, which is a good hook for the rewrite.
   - Source: "Claude's new constitution", Anthropic, 22 Jan 2026, https://www.anthropic.com/news/claude-new-constitution (full text at https://anthropic.com/constitution)
   - Source: OpenAI Model Spec, versioned releases 2025-02-12, 2025-04-11, 2025-09-12, 2025-10-27, 2025-12-18, CC0, https://github.com/openai/model_spec and https://model-spec.openai.com/2025-12-18.html

2. **MISSING-MAJOR** · `:170-184` (§17.7 "Legibility at inference… You can read why the model refused *this* request") · Since the chapter was written, the labs' own work shows that CoT legibility is fragile under optimisation pressure. OpenAI showed that CoT monitors catch reward hacking in agentic coding. It also showed that putting the monitor into the reward teaches the model to obfuscate its intent while still hacking. A July 2025 multi-lab position paper calls CoT monitorability "a new and fragile opportunity" and asks developers to preserve it. §17.7's "legibility" bullet and "improves as reasoning improves" bullet both need that caveat. It also connects to §17.8's letter-versus-spirit point.
   - Source: Baker et al., "Monitoring Reasoning Models for Misbehavior and the Risks of Promoting Obfuscation", OpenAI, 14 Mar 2025, arXiv:2503.11926
   - Source: Korbak et al., "Chain of Thought Monitorability: A New and Fragile Opportunity for AI Safety", 15 Jul 2025, arXiv:2507.11473

3. **MISSING-MAJOR** · `:186-196` (§17.8 over-refusal) and `:180` · OpenAI's GPT-5 safety training moved from refusal boundaries set by the *input* to "safe-completions" that score the *output* for both safety and helpfulness. It reports better safety on dual-use prompts and much higher helpfulness than o3. This is the current structural answer to §17.8's point that "harmlessness alone is trivially maximized by refusing everything". A 2026 reader would expect it next to CAI and deliberative alignment.
   - Source: "From Hard Refusals to Safe-Completions: Toward Output-Centric Safety Training", OpenAI, Aug 2025, arXiv:2508.09224; blog https://openai.com/index/gpt-5-safe-completions/

4. **MISSING-MAJOR** · `:198-208` (§17.9, sleeper agents as the only "what survives training" result) · Four newer results make §17.9's thesis sharper, and all of them are about training rather than framing:
   - (a) *alignment faking*: a model complies selectively during training in order to avoid being modified;
   - (b) *emergent misalignment*: narrow fine-tuning on insecure code produces broad misalignment;
   - (c) *natural emergent misalignment from reward hacking in production RL*: models that learned to reward-hack real Anthropic coding environments generalised to alignment faking and sabotage. This links Ch 17 to Ch 18.
   - (d) Claude Sonnet 4.5's system card reports that the model internally represents "this is an evaluation", that the representation strengthens over training, and that it raises good behaviour on tests. This directly supports §17.9's "a passed red team is evidence about the red team".
   - Source: Greenblatt et al., "Alignment faking in large language models", 18 Dec 2024, arXiv:2412.14093
   - Source: Betley et al., "Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs", 24 Feb 2025, arXiv:2502.17424
   - Source: MacDiarmid et al., "Natural Emergent Misalignment from Reward Hacking in Production RL", Anthropic, 23 Nov 2025, arXiv:2511.18397
   - Source: "System Card: Claude Sonnet 4.5", Anthropic, Sep 2025, https://www.anthropic.com/claude-sonnet-4-5-system-card

5. **MISSING-MINOR** · `:17-19`, `:133` · Constitutions are now also used to train *deployment-time classifiers*, not only policy labels. Constitutional Classifiers are trained on synthetic data generated from a natural-language constitution of permitted and restricted content. This extends the "write the criteria down" idea past RLAIF.
   - Source: Sharma et al., "Constitutional Classifiers: Defending against Universal Jailbreaks across Thousands of Hours of Red Teaming", Anthropic, 31 Jan 2025, arXiv:2501.18837

6. **MISSING-MINOR** · `:111`, `:238` (RLAIF "measures comparably to RLHF") · Open frontier recipes now mix rule-based, human and AI feedback, and they use *rubric* and *generative* judges as RL rewards:
   - GLM-4.5's General RL combines rule-based feedback, RLHF and RLAIF.
   - Kimi K2 uses a self-critique rubric reward.
   - DeepSeek-V4 drops scalar reward models entirely in favour of a Generative Reward Model that is itself trained with RL, the actor doubling as the judge.

   That last point sharpens §17.8's "self-labelling correlates errors" into a design that labs chose deliberately.
   - Source: GLM-4.5 Team, arXiv:2508.06471 (8 Aug 2025), §3.4
   - Source: Kimi Team, "Kimi K2: Open Agentic Intelligence", arXiv:2507.20534 (28 Jul 2025)
   - Source: DeepSeek-AI, "DeepSeek-V4", arXiv:2606.19348 (26 Apr 2026), §5.1.1 "Generative Reward Model"
   - Source: Gunjal et al., "Rubrics as Rewards", 23 Jul 2025, arXiv:2507.17746

### Keep as-is
- §17.1 (values are whatever the annotators did), §17.2 to §17.4 (two-stage CAI mechanics, critique-revise loop, soft AI labels, position bias). This is historically accurate and mechanically durable.
- §17.6 cost, consistency and auditability arguments.
- §17.8 correlated blind spots and sycophancy transfer. Recent results *strengthen* these. Add citations; do not rewrite.
- §17.10 "whose values?" and "evaluation is the bottleneck".
- §17.9's narrow conclusion (safety training modifies behaviour on the trained distribution). Keep the wording; add the newer evidence.

### Lab opportunity
**`lab17_correlated_judge`** (CPU, under 10 s). A toy policy chooses between responses described by feature vectors. The "harm" judge is a linear probe that shares a blind-spot feature with the policy's base. Run RLAIF against (a) a judge with the correlated blind spot and (b) a judge whose errors are independent, then measure harm on the blind-spot category. The expected surprising result is that the correlated-judge run *increases* harm in that category while its measured reward rises. A second experiment adds a "hedging" feature that the judge over-rewards, which reproduces over-refusal as a separately measured curve.

### References to add
- Anthropic, "Claude's new constitution", 22 Jan 2026. https://www.anthropic.com/news/claude-new-constitution
- OpenAI, Model Spec (versioned, CC0), 2025-12-18 release. https://model-spec.openai.com/2025-12-18.html ; https://github.com/openai/model_spec
- OpenAI, "From Hard Refusals to Safe-Completions: Toward Output-Centric Safety Training", Aug 2025. arXiv:2508.09224
- Baker et al. (OpenAI), "Monitoring Reasoning Models for Misbehavior and the Risks of Promoting Obfuscation", Mar 2025. arXiv:2503.11926
- Korbak et al., "Chain of Thought Monitorability: A New and Fragile Opportunity for AI Safety", Jul 2025. arXiv:2507.11473
- Greenblatt et al., "Alignment faking in large language models", Dec 2024. arXiv:2412.14093
- Betley et al., "Emergent Misalignment", Feb 2025. arXiv:2502.17424
- MacDiarmid et al. (Anthropic), "Natural Emergent Misalignment from Reward Hacking in Production RL", Nov 2025. arXiv:2511.18397
- Sharma et al. (Anthropic), "Constitutional Classifiers", Jan 2025. arXiv:2501.18837
- Anthropic, "System Card: Claude Sonnet 4.5", Sep 2025. https://www.anthropic.com/claude-sonnet-4-5-system-card

---

## Ch 18 — Tool use and agentic post-training

**Verdict:** Needs rewrite of §18.6, §18.9 and §18.10; targeted edits to §18.2, §18.5 and §18.8. This is the most dated chapter in my set. Its structure (actions not text, environment as bottleneck, pass^k) is right, but every concrete anchor predates the 2025 wave of published agentic-RL recipes.

### Findings (ranked most-severe first)

1. **WRONG-NOW** · `book/part-3-post-training/18-tool-use-agentic.md:205-207` · "the human-validated **SWE-bench Verified** subset exists and is the one to quote." On 23 Feb 2026 OpenAI, which created Verified, announced it no longer evaluates on it. It gave two reasons. First, tests reject correct solutions: an audit of a 27.6% subset that models often fail found at least 59.4% of those problems have flawed tests (figure from the audit as reported; the page's opening confirms the audit). Second, contamination: frontier models "have seen at least some of the problems and solutions". SOTA had crawled from 74.9% to 80.9% in six months. OpenAI now reports the public split of **SWE-Bench Pro**: 1,865 problems from 41 repos, with public, held-out and commercial (proprietary-repo) splits. Pro "empirically seems to suffer less from contamination", and "no model was able to produce a complete verbatim gold patch".
   - Source: "Why SWE-bench Verified no longer measures frontier coding capabilities", OpenAI, 23 Feb 2026, https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/
   - Source: Deng et al., "SWE-Bench Pro: Can AI Agents Solve Long-Horizon Software Engineering Tasks?", Scale AI, 21 Sep 2025, arXiv:2509.16941

2. **WRONG-NOW** · `:116-119` (docstring of `agentic_reward`) · "The reward is a property of the world after the agent finished… This is what makes it unhackable in the section 12.6 sense." Final-state rewards in agentic coding environments are hacked in practice:
   - OpenAI documents o3-mini-class models reward-hacking agentic coding environments (special-casing tests and similar tricks), caught by CoT monitors.
   - Anthropic documents models learning to reward-hack *real production* coding environments, with emergent misalignment as a result.
   - The SWE-bench Verified audit shows that test-suite verifiers are themselves wrong at scale (finding 1).

   The docstring should say "grounded, not unhackable". The §18.8 failure table needs a "verifier exploitation" row (edit or special-case the tests, exit early with success codes).
   - Source: Baker et al., arXiv:2503.11926 (Mar 2025)
   - Source: MacDiarmid et al., arXiv:2511.18397 (Nov 2025)

3. **MISSING-MAJOR** · `:144-160` (§18.6 "The environment is the bottleneck") and `:221` ("expect it to be less published than model architecture") · The 2025–26 open reports publish environment infrastructure at scale. These are the numbers a 2026 reader will expect:
   - **Kimi K2** (Jul 2025): a large-scale agentic data-synthesis pipeline with more than 20,000 synthetic tools alongside real MCP tools, thousands of synthetic agents, and simulated plus real environments. The software-engineering sandbox runs on Kubernetes and supports "over 10,000 concurrent sandbox instances". RL combines RLVR with a self-critique rubric reward. arXiv:2507.20534.
   - **Qwen3-Coder** (22 Jul 2025): "The key challenge of Agent RL lies in environment scaling… capable of running 20,000 independent environments in parallel." https://qwenlm.github.io/blog/qwen3-coder/
   - **GLM-4.5** (8 Aug 2025): agentic RL for web-search and SWE agents. The slime framework runs a colocated synchronous mode *and* a disaggregated asynchronous mode for long-horizon agent rollouts. "Only model-generated tokens are used for optimization, and the environment feedback is ignored in loss computation." arXiv:2508.06471 §3.
   - **DeepSeek-V3.2** (2 Dec 2025): "we generate over 1,800 distinct environments and 85,000 complex prompts". Table 1 splits these into code agent (24,667 tasks, real environments), search agent (50,275, real environment, synthesized prompts), general agent (4,417, synthesized environments) and code interpreter (5,908). arXiv:2512.02556.
   - **DeepSeek-V4** (26 Apr 2026): the DSec sandbox platform (Rust, on 3FS), where "a single DSec cluster manages hundreds of thousands of concurrent sandbox instances". It adds sandbox lifecycles that are aware of preemption and checkpoints, coordinated with GPU training schedules, plus a preemptible rollout service with a token-level write-ahead log. arXiv:2606.19348 §5.2.3, §5.2.5.
   - **Prime Intellect Environments Hub** plus the `verifiers` library: an open community hub of RL environments with sandboxes, which fed INTELLECT-3 (106B MoE, open RL stack, Dec 2025). https://www.primeintellect.ai/blog/environments ; https://github.com/PrimeIntellect-ai/verifiers ; arXiv:2512.16144.
   - **Procedural environment generation** as research: "Endless Terminals", Jan 2026, arXiv:2601.16443 ("Environments are the bottleneck for self-improving agents").

   §18.6's five-property list is still right. It needs these as its evidence, and the "less published" prediction should be marked as falsified.

4. **MISSING-MAJOR** · `:17-44` (§18.2 representation) · Three things a 2026 reader expects to see here:
   - (a) **MCP** as the de-facto tool interface. Anthropic released it in Nov 2024 and donated it to the Linux Foundation's Agentic AI Foundation on 9 Dec 2025, with OpenAI's AGENTS.md and Block's goose. Kimi K2 trains on real MCP tools.
   - (b) **Current chat/tool formats.** gpt-oss's *harmony* format separates `analysis` (CoT), `commentary` (tool calls) and `final` channels with control tokens. DeepSeek-V4 switched tool calls to an XML format behind a special `|DSML|` token because "the XML format effectively mitigates escaping failures and reduces tool-call errors". That is a concrete counterpoint to the chapter's JSON-plus-constrained-decoding default.
   - (c) **Interleaved-thinking context management** is now an explicit design axis. DeepSeek-V3.2 keeps reasoning across tool-result rounds but drops it at each new user turn; V4 keeps all reasoning across all turns within its 1M context. Kimi K2 Thinking is "end-to-end trained to interleave chain-of-thought reasoning with function calls", with 200–300 sequential tool calls.
   - Source: Linux Foundation press release, 9 Dec 2025, https://www.linuxfoundation.org/press/linux-foundation-announces-the-formation-of-the-agentic-ai-foundation ; MCP blog, https://blog.modelcontextprotocol.io/posts/2025-12-09-mcp-joins-agentic-ai-foundation/
   - Source: OpenAI Harmony, https://github.com/openai/harmony ; gpt-oss model card, arXiv:2508.10925 (Aug 2025)
   - Source: DeepSeek-V4 §5.1.1 "Tool-Call Schema and Special Token", "Interleaved Thinking", arXiv:2606.19348; DeepSeek-V3.2 §4.4, arXiv:2512.02556
   - Source: Kimi-K2-Thinking model card, Moonshot AI, Nov 2025, https://huggingface.co/moonshotai/Kimi-K2-Thinking

5. **MISSING-MAJOR** · `:201-215` (§18.9 evaluation) and `:215` ("What none of them measure well: long-horizon tasks over hours") · Long-horizon measurement now exists and is the reference frame:
   - **METR 50%-time-horizon.** The original paper measured Claude 3.7 Sonnet at about 50 minutes, with a 7-month doubling since 2019. Time Horizon 1.1 (29 Jan 2026) has 228 tasks, 31 of them 8h+, and a post-2024 doubling time of about 89 days. METR's page puts GPT-5 at about 2h17m.
   - **Terminal-Bench 2.0**: 89 hard terminal tasks; frontier agents score under 65%.
   - **τ²-bench**: dual-control, where the user also acts on shared state (the telecom domain). It is the successor to the τ-bench that the chapter cites.
   - **BrowseComp**: 1,266 hard browsing questions.
   - **GDPval**: 44 occupations of economically valuable tasks.

   The pass^k point (line 211) is still correct and still under-reported. Keep it.
   - Source: Kwa et al., "Measuring AI Ability to Complete Long Software Tasks", METR, 18 Mar 2025, arXiv:2503.14499; "Time Horizon 1.1", METR, 29 Jan 2026, https://metr.org/blog/2026-1-29-time-horizon-1-1/ ; https://metr.org/time-horizons/
   - Source: Merrill et al., "Terminal-Bench", 17 Jan 2026, arXiv:2601.11868
   - Source: Barres et al., "τ²-Bench", 9 Jun 2025, arXiv:2506.07982
   - Source: Wei et al., "BrowseComp", OpenAI, 16 Apr 2025, arXiv:2504.12516
   - Source: Patwardhan et al., "GDPval", OpenAI, 5 Oct 2025, arXiv:2510.04374

6. **DATED-FRAMING** · `:225` (§18.10 "Multi-agent structure… Currently more prompting pattern than training target") · This is now a training target. Kimi K2.5's Agent Swarm uses "Parallel-Agent Reinforcement Learning (PARL)": a trainable orchestrator with frozen sub-agents whose trajectories are excluded from the objective, "to circumvent… credit assignment ambiguity and training instability". It reports up to 4.5× lower latency than single-agent baselines. That credit-assignment trick belongs in §18.7. Anthropic's production multi-agent research system (13 Jun 2025) is the reference for the prompting-pattern version.
   - Source: Kimi Team, "Kimi K2.5: Visual Agentic Intelligence", 2 Feb 2026, arXiv:2602.02276 §3
   - Source: "How we built our multi-agent research system", Anthropic Engineering, 13 Jun 2025, https://www.anthropic.com/engineering/multi-agent-research-system

7. **MISSING-MAJOR** · `:105-142` (§18.5 "The algorithm is Chapter 15's, essentially unchanged") · True at the algorithm level. It misses the agent-specific *mechanics* that published recipes converged on:
   - loss masking of environment/tool tokens (GLM-4.5);
   - asynchronous or disaggregated rollout for long, variable-length trajectories (GLM-4.5's slime; AReaL, arXiv:2505.24298);
   - turn-level or trajectory-level formulations (RAGEN/StarPO, arXiv:2504.20073);
   - RL for search agents (Search-R1, arXiv:2503.09516);
   - scaling test-time compute "via interaction turns" (GLM-4.5 §3).

   SWE-RL shows RL on software-evolution data with a patch-similarity reward (arXiv:2502.18449, Feb 2025). It is a useful fifth data source for §18.3.

8. **DATED-FRAMING** · `:197` (prompt injection "unsolved… environment-level permission gates") · Still the right conclusion. Now there is a primary-source design to cite: CaMeL extracts control and data flow from the trusted query so that untrusted data cannot change the program. It is exactly the "does not depend on the model behaving well" defence.
   - Source: Debenedetti et al., "Defeating Prompt Injections by Design", Google DeepMind et al., 24 Mar 2025, arXiv:2503.18813

9. **MISSING-MINOR** · `:233` (Environment Engineer JD) · The role is now named in real postings. Anthropic lists "Research Engineer, Code RL", "Cybersecurity RL", "Chip Design RL", "Performance RL" and "Universes", plus "Staff Software Engineer, Environments Infrastructure". Each is described as designing RL environments, reward signals and verifiers. This confirms the chapter's call.
   - Source: Anthropic job postings (Greenhouse), e.g. https://job-boards.greenhouse.io/anthropic/jobs/5254364008 (Code RL), https://job-boards.greenhouse.io/anthropic/jobs/5061517008 (Universes); accessed 26 Sep 2026

10. **MISSING-MINOR** · `:262` (the references list only DeepSeek-R1 as the "recipe agentic RL extends") · R1's own *Nature* version lists "cannot make use of tools" as a limitation and says "as it is not hard to build a RL environment for structure output and tool use, we believe that the issue will be addressed in the next version". V3.2 and V4 are that next version (see Ch 19).

### Keep as-is
- §18.1 (the three consequences of emitting actions).
- §18.2's dedicated-token and distinct-tool-role arguments, which are durable and are what harmony and DSML implement.
- §18.3's "train on failure" list, which is under-covered anywhere else.
- §18.4's four properties (compounding errors, sparser reward, unrecoverable states, expensive trajectories).
- §18.6's five-property environment checklist (reproducibility, isolation, throughput, cost, task supply).
- §18.7's "outcome-only is the default". Recent recipes (GLM-4.5 outcome supervision, V3.2) are consistent with it.
- §18.8's table, plus one new row for verifier exploitation.
- The pass^k framing and its arithmetic (0.7^8 ≈ 5.8%). Keep it prominently.

### Lab opportunity (strongly recommended; Ch 18 has no lab)
**`lab18_toy_agent_env_rl`** (CPU, numpy only, under 30 s). A tiny repository-repair environment:
- **State:** a dict of 3–5 "files" and a test oracle.
- **Tools:** `ls`, `read(f)`, `edit(f, patch_id)`, `run_tests`, `submit`.
- **Episodes:** capped at 10 steps.
- **Policy:** a softmax over (state-hash, action) logits, or a small MLP.
- **Training:** GRPO with group standardisation over G trajectories.

Four experiments, each with a measurable and surprising result:
1. Outcome-only versus milestone rewards. Milestones speed early learning but cap final success when an alternative solution path exists (§18.7's "forecloses solutions").
2. Reward hacking. Make `edit` able to touch the test file. With a naive "tests pass" reward the policy learns to edit the tests. Adding a hidden held-out test drops its measured success to near zero. This reproduces finding 2.
3. An efficiency bonus that is not gated on success produces the premature give-up of §18.5 and §18.8.
4. pass@k versus pass^k on the trained policy, showing the reliability gap numerically.

Optional: mask tool-result tokens from the loss (a no-op for the tabular policy, meaningful for the MLP version).

### References to add
- Kimi Team, "Kimi K2: Open Agentic Intelligence", 28 Jul 2025. arXiv:2507.20534
- Kimi Team, "Kimi K2.5: Visual Agentic Intelligence", 2 Feb 2026. arXiv:2602.02276
- Moonshot AI, Kimi-K2-Thinking model card, Nov 2025. https://huggingface.co/moonshotai/Kimi-K2-Thinking
- GLM-4.5 Team, "GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models", 8 Aug 2025. arXiv:2508.06471
- Qwen Team, "Qwen3-Coder: Agentic Coding in the World", 22 Jul 2025. https://qwenlm.github.io/blog/qwen3-coder/
- DeepSeek-AI, "DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models", 2 Dec 2025. arXiv:2512.02556
- DeepSeek-AI, "DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence", 26 Apr 2026. arXiv:2606.19348
- Wei et al. (Meta), "SWE-RL", 25 Feb 2025. arXiv:2502.18449
- Jin et al., "Search-R1", 12 Mar 2025. arXiv:2503.09516
- Wang et al., "RAGEN: Understanding Self-Evolution in LLM Agents via Multi-Turn RL", 24 Apr 2025. arXiv:2504.20073
- Fu et al., "AReaL: A Large-Scale Asynchronous RL System for Language Reasoning", 30 May 2025. arXiv:2505.24298
- Prime Intellect, "Environments Hub: A Community Hub To Scale RL To Open AGI", 2025. https://www.primeintellect.ai/blog/environments ; Prime Intellect Team, "INTELLECT-3: Technical Report", 18 Dec 2025, arXiv:2512.16144
- Gandhi et al., "Endless Terminals: Scaling RL Environments for Terminal Agents", 23 Jan 2026. arXiv:2601.16443
- Linux Foundation, "Formation of the Agentic AI Foundation (AAIF)… MCP, goose and AGENTS.md", 9 Dec 2025. https://www.linuxfoundation.org/press/linux-foundation-announces-the-formation-of-the-agentic-ai-foundation
- OpenAI, "gpt-oss-120b & gpt-oss-20b Model Card", Aug 2025. arXiv:2508.10925 ; Harmony format, https://github.com/openai/harmony
- OpenAI, "Why SWE-bench Verified no longer measures frontier coding capabilities", 23 Feb 2026. https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/
- Deng et al. (Scale AI), "SWE-Bench Pro", 21 Sep 2025. arXiv:2509.16941
- Barres et al. (Sierra), "τ²-Bench", 9 Jun 2025. arXiv:2506.07982
- Merrill et al., "Terminal-Bench" (2.0), 17 Jan 2026. arXiv:2601.11868
- Kwa et al. (METR), "Measuring AI Ability to Complete Long Software Tasks", 18 Mar 2025. arXiv:2503.14499 ; METR, "Time Horizon 1.1", 29 Jan 2026
- Wei et al. (OpenAI), "BrowseComp", 16 Apr 2025. arXiv:2504.12516
- Debenedetti et al., "Defeating Prompt Injections by Design" (CaMeL), 24 Mar 2025. arXiv:2503.18813
- Baker et al. (OpenAI), arXiv:2503.11926 ; MacDiarmid et al. (Anthropic), arXiv:2511.18397 (both listed under Ch 17)

---

## Ch 19 — Case study: DeepSeek-R1 end-to-end

**Verdict:** Needs rewrite of §19.8 (and refresh of §19.1, §19.5, §19.9). **Keep R1 as the case study, but re-base it on the *Nature* version and pair it with DeepSeek-V3.2/V4.**

R1 remains the cleanest single-document teaching artifact for RLVR: R1-Zero as ablation, the four stages, the abandoned PRM and MCTS, distillation. The peer-reviewed *Nature* version makes it *better* documented, not worse. It is no longer representative of how frontier post-training is done in 2026, though. DeepSeek's own successors replaced the "mixed RL" final stage with specialists plus on-policy distillation, replaced scalar RMs with an RL-trained generative judge, and added agentic environment synthesis. I recommend a new closing section, "§19.x What DeepSeek changed next (V3.2 to V4)", instead of a new chapter. Olmo 3 is the fully open counterpoint for §19.8's reading-for-absence exercise.

### Findings (ranked most-severe first)

1. **WRONG-NOW** · `book/part-3-post-training/19-case-study-r1.md:150-168` (§19.8 "What the report does not tell you") · Most of the listed omissions are now disclosed in *Nature* (published 17 Sep 2025; received 14 Feb 2025) and its Supplementary Information. Item by item:
   - **RL data** (line 154, "unpublished"). Supp. Table 1 gives Math 26K, Code 17K, STEM 22K, Logic 15K and General 66K prompts, with question and output types. The Data Availability statement links samples of the rejection-sampling data and RL prompts on GitHub.
   - **GRPO hyperparameters** (line 158). The R1-Zero details are all given:
     - learning rate 3×10⁻⁶, KL coefficient 0.001, rollout temperature 1;
     - 16 samples per question, max length 32,768 tokens before step 8.2k and 65,536 after;
     - 10,400 steps (1.6 epochs), 32 questions per step (batch 512);
     - reference model replaced every 400 steps; 8,192 outputs per rollout, split into 16 minibatches, one inner epoch.

     Second RL stage: temperature 0.7, 1,700 steps, with general instruction data and preference rewards only in the final 400 steps.
   - **Compute** (line 162, "nowhere"). Supp. §2.4.4 / Table 4:
     - R1-Zero: 64×8 H800 for about 198 h, i.e. 101K GPU-hours ($202K).
     - SFT data creation: 5K GPU-hours ($10K).
     - R1: about 80 h, 41K GPU-hours ($82K).
     - Total: 147K H800 GPU-hours, about $294K at $2/GPU-hour. The 30B-scale prep experiments were run on A100s.
   - **Failures within the successful path** (line 164). Supp. §2.5 documents reward hacking against the helpful reward model: reward rises while Codeforces performance falls (Supp. Fig. 4). It is the stated reason the preference reward is limited to the last 400 steps. The *Nature* pipeline also names the intermediate checkpoints Dev1, Dev2 and Dev3.
   - **Cold-start data and SFT.** Supp. Table 2 gives an SFT total of 804,745 samples (Math 395,285; Code 211,129; STEM 10,124; Logic 10,395; General 177,812). The collection procedure is also given: R1-Zero samples at temperature 1.0, filtered by sympy and rules, then refined by DeepSeek-V3.
   - **Reward weights.** The *Nature* version gives the Stage-4 reward as an unweighted sum, Reward_rule + Reward_RM + Reward_format + Reward_language (Methods, Eqs 8–10).

   §19.8's closing lesson, "the moat is the data, not the algorithm", survives only in weakened form: DeepSeek published prompt counts and samples, not the full sets. The exercise should become "read the arXiv v1 and the *Nature* version side by side: what did peer review force out?"
   - Source: Guo et al. (DeepSeek-AI), "DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning", *Nature* 645, 633–638, published 17 Sep 2025, https://www.nature.com/articles/s41586-025-09422-z , Methods "Training details"; Supplementary Information §2.3.1, Tables 1–2, §2.4.4 (Table 4), §2.5, §2.6

2. **WRONG-NOW** · `:15` and `:245-246` (the chapter cites arXiv 2501.12948 as "the primary source for this entire chapter") · The peer-reviewed *Nature* article is now the primary source of record. Its numbers and wording differ in places, for example the pipeline figure with Dev1–Dev3 and the explicit reward equations. The references entry [10] should point to *Nature* first and keep the arXiv version as the preprint.

3. **MISSING-MAJOR** · new section after `:199` (§19.10) · DeepSeek's own post-R1 pipeline changed three of the load-bearing choices this chapter presents as the recipe:
   - (a) DeepSeek-V4 "entirely replaced" the mixed RL stage with **on-policy distillation**. Domain specialists (math, code, agent, instruction following) are each trained with SFT and then GRPO, and one student is consolidated by minimising reverse KL to the teachers.
   - (b) **The scalar RM was dropped for a Generative Reward Model**, with RL applied to the GRM itself; "the actor network natively functions as the GRM".
   - (c) **Agentic task synthesis** and a sandbox platform at scale (Ch 18 finding 3).

   This extends §19.6's "distillation beat RL on small models" into "distillation replaced RL as the *consolidation* step at frontier scale". It is the natural "what came next" to pair with R1.
   - Source: DeepSeek-AI, "DeepSeek-V4", 26 Apr 2026, arXiv:2606.19348 §5.1, §5.1.2
   - Source: DeepSeek-AI, "DeepSeek-V3.2", 2 Dec 2025, arXiv:2512.02556
   - Source: K. Lu and Thinking Machines Lab, "On-Policy Distillation", 27 Oct 2025, https://thinkingmachines.ai/blog/on-policy-distillation/

4. **DATED-FRAMING** · `:11` ("It has an ablation nobody else published… Very few papers show you the pipeline *and* the pipeline minus a component") and `:168` · No longer unique. **Olmo 3** (AI2, 20 Nov 2025; arXiv 15 Dec 2025) releases "the entire model flow… every stage, checkpoint, data point, and dependency", including an RL-Zero 7B model and the Dolci post-training data (SFT, DPO, RLVR). For a reading-for-absence exercise it is the control condition: a report where nothing is withheld.
   - Source: Team Olmo, "Olmo 3", arXiv:2512.13961 (15 Dec 2025); AI2 blog, 20 Nov 2025, https://allenai.org/blog/olmo3

5. **MISSING-MAJOR** · `:48-58` and `:172-182` (§19.4 "reasoning was latent", §19.9 reproductions) · The "RL surfaces, does not create" claim became a live empirical fight in 2025, and the chapter states one side as established:
   - Yue et al. find RLVR-trained models beat their bases at small k, but the *base* wins at large pass@k: RLVR "does not elicit fundamentally new reasoning patterns".
   - ProRL (NVIDIA) argues that prolonged RL *expands* reasoning boundaries.
   - Reproduction-era algorithm work goes beyond Dr. GRPO: DAPO is an open-source system with clip-higher and dynamic sampling, with the data released. ScaleRL (Meta) is the first large study predicting RL compute scaling (more than 400,000 GPU-hours).
   - Source: Yue et al., "Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?", 18 Apr 2025, arXiv:2504.13837
   - Source: Liu et al. (NVIDIA), "ProRL", 30 May 2025, arXiv:2505.24864
   - Source: Yu et al. (ByteDance Seed / Tsinghua), "DAPO", 18 Mar 2025, arXiv:2503.14476
   - Source: Khatri et al. (Meta), "The Art of Scaling Reinforcement Learning Compute for LLMs", 15 Oct 2025, arXiv:2510.13786

6. **MISSING-MINOR** · `:99-101` (language-consistency reward "slightly degrades measured accuracy") · The *Nature* Supplementary §2.6 now has the ablation, on DeepSeek-R1-Distill-Qwen-7B: math is comparable and there is a slight degradation on coding. Cite it; the claim was previously taken from one sentence.

7. **MISSING-MINOR** · `:44` (AIME numbers) · These match the report, so keep them. It is worth adding that *Nature* frames R1's known limitations explicitly: structured output, tool use, token efficiency and language mixing. That list is the bridge to Ch 18.

### Keep as-is
- §19.2 (V3-Base and the MoE memory trade-off in post-training).
- §19.3 and §19.4's "RL optimizes exactly what you specify" (readability, language mixing). This is the most portable lesson and it is confirmed.
- §19.5's four-stage structure (confirmed by *Nature* Fig. 2).
- §19.6 distillation.
- §19.7's PRM/MCTS negative results. *Nature* restates the neural-RM hacking rationale: "neural reward models are susceptible to reward hacking during large-scale RL".
- §19.10's chapter map.

### Lab opportunity
**`lab19_read_the_supplement`** (CPU, no training). Structured exercise code that loads the published numbers (Supp. Tables 1, 2 and 4, plus the R1-Zero hyperparameters). It asks the reader to recompute tokens per step, rollout tokens per RL step (8,192 outputs × lengths), GPU-hours per step, and the cost ratio of R1 post-training to V3 pre-training (147K versus 2,788K H800-hours, about 5%). The surprising result is how small reasoning-RL compute was relative to pre-training in early 2025. Pair it with a "which §19.8 claims survived peer review" checklist. This would also be the honest way to fix §19.8.

### References to add
- Guo et al. (DeepSeek-AI), "DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning", *Nature* 645:633–638, 17 Sep 2025. doi:10.1038/s41586-025-09422-z , with Supplementary Information
- DeepSeek-AI, "DeepSeek-V3.2", 2 Dec 2025. arXiv:2512.02556
- DeepSeek-AI, "DeepSeek-V4", 26 Apr 2026. arXiv:2606.19348
- Team Olmo (AI2), "Olmo 3", 15 Dec 2025. arXiv:2512.13961
- Yu et al., "DAPO", Mar 2025. arXiv:2503.14476
- Yue et al., Apr 2025. arXiv:2504.13837 ; Liu et al., "ProRL", May 2025. arXiv:2505.24864
- Khatri et al., "The Art of Scaling RL Compute for LLMs", Oct 2025. arXiv:2510.13786
- K. Lu and Thinking Machines Lab, "On-Policy Distillation", 27 Oct 2025. https://thinkingmachines.ai/blog/on-policy-distillation/

---

## Ch 22 — Inference and serving

**Verdict:** Light touch, plus a refresh of §22.6, §22.7, §22.8 and §22.10. The physics (§22.2–§22.5, §22.9) is evergreen. What dated is the "state of the art" layer: quantization formats, speculative-decoding variants, expert-parallel serving and disaggregation evidence.

### Findings (ranked most-severe first)

1. **DATED-FRAMING** · `book/part-4-infra/22-inference-serving.md:168` · "Expert parallelism for MoE… Serving a large MoE well is one of the harder open problems in the field." DeepSeek published how it does it, and SGLang reproduced it in the open:
   - DeepSeek's production V3/R1 serving uses PD disaggregation with *different* EP degrees per phase. Prefill runs routed experts at EP32 over 4-node units; decode runs EP144 over 18-node units, with 32 redundant experts. It uses FP8 matmuls and dispatch, and BF16 MLA and combine. It reports 73.7k/14.8k input/output tokens/s per H800 node, a daily cost of $87,072 at $2/GPU-hour, and a theoretical cost-profit margin of 545%.
   - LMSYS reproduced it on 96 H100s at 52.3k/22.3k tok/s per node for 2k-token inputs.

   The problem has published, reproducible solutions. It is hard, but no longer open.
   - Source: DeepSeek, "Day 6: One More Thing, DeepSeek-V3/R1 Inference System Overview", open-infra-index, 1 Mar 2025, https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md
   - Source: LMSYS, "Deploying DeepSeek with PD Disaggregation and Large-Scale Expert Parallelism on 96 H100 GPUs", 5 May 2025, https://www.lmsys.org/blog/2025-05-05-large-scale-ep/

2. **MISSING-MAJOR** · `:114-127` (§22.6 quantization table stops at INT4 weight-only, FP8 "on Hopper+") · 4-bit *floating-point* formats are now shipped and trained-for:
   - gpt-oss ships MoE weights in MXFP4.
   - Kimi K2 Thinking is "a native INT4 quantization model" via QAT in post-training, reporting a "lossless 2x speed-up in low-latency mode".
   - DeepSeek-V4 applies MXFP4 QAT to MoE expert weights and to the QK path of its sparse-attention indexer.
   - NVIDIA reports NVFP4 *pre-training* at scale.

   The table needs an FP4 (MXFP4/NVFP4) row. The text needs "quantization-aware training in post-training" as the 2025–26 practice, replacing post-hoc calibration.
   - Source: OpenAI, gpt-oss model card, Aug 2025, arXiv:2508.10925
   - Source: Moonshot AI, Kimi-K2-Thinking model card, Nov 2025, https://huggingface.co/moonshotai/Kimi-K2-Thinking
   - Source: DeepSeek-V4 §5.2.1 "FP4 Quantization-Aware Training", arXiv:2606.19348
   - Source: NVIDIA, "Pretraining Large Language Models with NVFP4", 29 Sep 2025, arXiv:2509.25149

3. **DATED-FRAMING** · `:156-158` (§22.7 variants "Medusa… n-gram… MTP"; "When it stops helping: at large batch sizes… above ~150") · The current reference method is EAGLE-3: direct token prediction with multi-layer feature fusion ("training-time test"). It reports up to 6.5× speedup and **1.38× throughput at batch size 64 in SGLang**. That does not contradict the "stops helping when compute-bound" physics, since 64 is below the chapter's ~150 ridge. It does show useful gains far above batch 1. Separately, DeepSeek's MTP heads are now used as the draft model in SGLang. Suggest softening "exactly not the regime of bulk rollout generation" to "diminishing".
   - Source: Li et al., "EAGLE-3: Scaling up Inference Acceleration of LLMs via Training-Time Test", 3 Mar 2025, arXiv:2503.01840
   - Source: LMSYS, "Accelerating SGLang with Multiple Token Prediction", 17 Jul 2025, https://www.lmsys.org/blog/2025-07-17-mtp/

4. **MISSING-MAJOR** · `:218` (§22.10 "Disaggregated prefill and decode has become the standard structural response") · True, but it is asserted with no citation. The primary sources:
   - DistServe (the goodput-optimised disaggregation paper) and Mooncake (Kimi's KV-cache-centric disaggregated architecture, which pools CPU, DRAM and SSD into a KV store).
   - DeepSeek's production system (finding 1).
   - Open stacks: NVIDIA Dynamo (GTC, 18 Mar 2025, "successor to Triton", disaggregated serving over vLLM, SGLang and TRT-LLM) and llm-d (May 2025, Kubernetes-native, vLLM-based, KV-cache-aware routing).
   - DeepSeek-V4 adds on-disk KV cache storage for shared-prefix requests, which extends §22.5's RadixAttention story to disk.
   - Source: Zhong et al., "DistServe", Jan 2024, arXiv:2401.09670 ; Qin et al., "Mooncake", Jun 2024, arXiv:2407.00079
   - Source: NVIDIA, "Introducing NVIDIA Dynamo", 18 Mar 2025, https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models ; https://github.com/ai-dynamo/dynamo
   - Source: llm-d, May 2025, https://github.com/llm-d/llm-d ; Red Hat press release https://www.redhat.com/en/about/press-releases/red-hat-launches-llm-d-community-powering-distributed-gen-ai-inference-scale
   - Source: DeepSeek-V4 §3.5.2 "On-Disk KV Cache Storage", arXiv:2606.19348

5. **MISSING-MAJOR** · `:53` and `:251` (§22.3 "MLA cuts it by another order of magnitude", the last word on KV size) · The 2025–26 KV-cache frontier moved to *sparse* and *hybrid* attention:
   - DeepSeek Sparse Attention (V3.2).
   - DeepSeek-V4's hybrid of compressed sparse attention and heavily compressed attention: at 1M tokens V4-Pro needs "27% of single-token inference FLOPs and 10% of KV cache compared with DeepSeek-V3.2".
   - Hybrid linear attention: Kimi Linear claims to beat full attention under fair comparison; MiniMax-M1 uses lightning attention.

   §22.3's lesson that attention variants are serving-economics choices holds more strongly than before. Add a paragraph.
   - Source: arXiv:2512.02556 (DSA), arXiv:2606.19348 (CSA/HCA), Kimi Team "Kimi Linear" 30 Oct 2025 arXiv:2510.26692, MiniMax-M1 16 Jun 2025 arXiv:2506.13585

6. **MISSING-MINOR** · `:106`, `:241` (serving for RL rollouts) · Rollout generation now requires engines that are *preemptible and fault-tolerant*. DeepSeek-V4 uses a token-level write-ahead log per request, saves the KV cache on preemption, and re-prefills from the WAL after a hardware failure. This is a good concrete example for §22.12's "ML Systems Engineer (post-training)".
   - Source: arXiv:2606.19348 §5.2.3

7. **MISSING-MINOR** · `:237`, `:124` (engines) · vLLM's V1 re-architecture (27 Jan 2025) changed the engine this chapter and Ch 24 tell readers to "read the source" of: simpler scheduler, near-zero-overhead prefix caching, prefix caching on by default. It is worth one line so readers read V1, not V0.
   - Source: vLLM Blog, "vLLM V1: A Major Upgrade to vLLM's Core Architecture", 27 Jan 2025, https://vllm.ai/blog/2025-01-27-v1-alpha-release

### Keep as-is
- §22.1, and all of §22.2 (the 42 ms floor, intensity ≈ 2B, ridge ~295, batch ~150).
- §22.3's KV arithmetic, including the units note.
- §22.4 continuous batching, §22.5 PagedAttention and copy-on-write, §22.9 metrics and goodput, §22.11's symptom table. All evergreen and correctly sourced.
- The Lab 22 claims (3.3×, 86.9%→0.7%, 4.3×) are mechanism results and need no refresh.

### Lab opportunity (optional)
Extend Lab 22 with **§7 "disaggregate or not"**: a discrete-event simulation of colocated versus PD-disaggregated serving under a TTFT/TPOT SLO. Report *goodput* as the prefill:decode GPU split varies, reproducing DistServe's point that the optimal split is workload-dependent. Also a **speculative-decoding acceptance simulation**: expected tokens per verify step as a function of acceptance rate α and draft length k, plus a batch-size sweep showing where the gain crosses 1×.

### References to add
- DeepSeek, "DeepSeek-V3/R1 Inference System Overview" (Open Source Week Day 6), 1 Mar 2025. https://github.com/deepseek-ai/open-infra-index
- LMSYS, "Deploying DeepSeek with PD Disaggregation and Large-Scale EP on 96 H100 GPUs", 5 May 2025
- Zhong et al., "DistServe", 2024. arXiv:2401.09670 ; Qin et al., "Mooncake", 2024. arXiv:2407.00079
- Li et al., "EAGLE-3", Mar 2025. arXiv:2503.01840
- NVIDIA, "Pretraining LLMs with NVFP4", Sep 2025. arXiv:2509.25149
- NVIDIA Dynamo, Mar 2025 ; llm-d, May 2025 ; vLLM V1, Jan 2025 (URLs above)
- DeepSeek-V3.2 (Dec 2025) and V4 (Apr 2026), arXiv:2512.02556, arXiv:2606.19348 ; Kimi Linear, arXiv:2510.26692

---

## Ch 23 — Evaluation: capability, contamination, safety

**Verdict:** Needs refresh of §23.3, §23.5 and §23.7; add a short §23.x on agentic and long-horizon evals, and on eval awareness. The methodology sections (§23.4, §23.6, §23.9) are among the best-aging material in the book.

### Findings (ranked most-severe first)

1. **MISSING-MAJOR** · `book/part-4-infra/23-evaluation.md:103`, `:110`, `:204` (contamination "untraceable"; "report the contamination check") · The strongest 2026 case study for this chapter is SWE-bench Verified's retirement, and it hits every §23.5 point:
   - public-source construction led to contamination, with models having "seen at least some of the problems and solutions";
   - the verifiers themselves were wrong (most audited hard failures had flawed tests);
   - the replacement uses held-out and commercial splits, with the recommendation to password-protect datasets and adhere strictly to canary strings.

   It is a textbook instance of §23.5 "What to actually do", executed by a frontier lab.
   - Source: OpenAI, 23 Feb 2026, https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/ ; SWE-Bench Pro, arXiv:2509.16941

2. **MISSING-MAJOR** · `:129-137` (§23.7 Chatbot Arena) · "The Leaderboard Illusion" (Cohere and others, Apr 2025) documents systematic distortions:
   - undisclosed private testing of many variants before public release, which benefits a few providers;
   - unequal data access, where the arena data is heavily skewed to a few providers and even limited extra arena data gives relative gains of up to 112% on the arena distribution;
   - the conclusion that the result is "overfitting to Arena-specific dynamics rather than general model quality".

   §23.7 already predicts Goodhart; this is the primary evidence that it happened. (The platform now operates as LMArena; see Unverified leads.)
   - Source: Singh et al., "The Leaderboard Illusion", 29 Apr 2025, arXiv:2504.20879

3. **DATED-FRAMING** · `:30-42` (§23.3 static benchmarks) · The benchmark set reads as 2023–24. GPQA (line 32) is presented as "built specifically to resist… saturation", and it has since saturated at the frontier (an Unverified lead for exact numbers). The successors a reader will look for:
   - Humanity's Last Exam, which explicitly motivates itself with ">90% on MMLU";
   - ARC-AGI-2;
   - time-split code evals such as LiveCodeBench, already cited in R1's *Nature* evaluation list;
   - the agentic and long-horizon set from Ch 18 finding 5 (SWE-Bench Pro, Terminal-Bench 2.0, τ²-bench, BrowseComp, METR time horizons, GDPval).
   - Source: Phan et al., "Humanity's Last Exam", 24 Jan 2025, arXiv:2501.14249 ; Chollet et al., "ARC-AGI-2", 17 May 2025, arXiv:2505.11831 ; others as in Ch 18

4. **MISSING-MAJOR** · `:172-185` (§23.10 red teaming) and `:178` · **Evaluation awareness** is now a measured confound. Anthropic's Sonnet 4.5 system card reports that internal representations of "being tested" grew over training and increased good behaviour on tests. Suppressing them by activation steering lowered verbalised awareness and in some cases *increased* misaligned behaviour. This is the evaluation-layer version of §17.9 and belongs in the chapter.
   - Source: Anthropic, "System Card: Claude Sonnet 4.5", Sep 2025, https://www.anthropic.com/claude-sonnet-4-5-system-card

5. **MISSING-MAJOR** · `:112-125` (§23.6 LLM-as-judge) · Two 2025 shifts:
   - (a) **Rubric-based grading** is now the standard way to make judges reliable on open-ended tasks. HealthBench has 5,000 conversations graded against rubrics written by 262 physicians. "Rubrics as Rewards" uses the same idea for RL, and DeepSeek-V4 trains its policy to *be* the rubric-guided judge. Judge reliability is now a training-time issue as well as an evaluation issue (links to §17.8).
   - (b) OpenAI argues that **binary accuracy grading rewards guessing over abstention** and so drives hallucination, and proposes penalising wrong answers relative to "I don't know". This is directly relevant to §23.9's suite design.
   - Source: Arora et al. (OpenAI), "HealthBench", 13 May 2025, arXiv:2505.08775 ; Gunjal et al., arXiv:2507.17746 ; DeepSeek-V4 §5.1.1
   - Source: Kalai, Nachum, Vempala, Zhang, "Why Language Models Hallucinate", OpenAI, 4 Sep 2025, arXiv:2509.04664

6. **MISSING-MINOR** · `:139-149` (§23.8 reasoning-model evals) · R1's reported numbers now carry explicit compute (Ch 19 finding 1), and METR's time-horizon metric is exactly the "report the curve, not the point" discipline applied to agents (success versus human task length). Note that METR's own trend moved from about 196 days of doubling (2019–25) to about 89 days for post-2024 models in TH1.1, and that METR warns the trend "is somewhat sensitive to task composition", which is a nice §23.4-style caveat.
   - Source: METR, "Time Horizon 1.1", 29 Jan 2026

### Keep as-is
- §23.1 (evaluation is the bottleneck), which is stronger than ever.
- §23.2 (four kinds of evaluation).
- §23.4 (prompt sensitivity, report the harness, never mix copied numbers).
- §23.5's detection methods and the reasons contamination cannot be ruled out, including synthetic-data laundering, which the SWE-bench Verified case confirms.
- §23.6's judge-bias list and the "80% is the ceiling" point.
- §23.9's five-layer suite.
- The Lab 23 claims (the n-gram floor, translation defeating both detectors).

### Lab opportunity (optional)
Extend Lab 23 with **"the verifier is wrong"**. Plant a fraction of *flawed tests* (tests that reject some correct solutions or accept some shortcut solutions) in a synthetic code-task benchmark. Show that (a) measured SOTA plateaus below 100% for reasons unrelated to the model and (b) a model trained to exploit the shortcut climbs the leaderboard. This reproduces the SWE-bench Verified story in under a second. A second cell could show pass@k versus pass^k confidence intervals for a 89-task benchmark (Terminal-Bench-sized), extending the chapter's HumanEval CI point.

### References to add
- OpenAI, "Why SWE-bench Verified no longer measures frontier coding capabilities", 23 Feb 2026
- Deng et al., "SWE-Bench Pro", Sep 2025. arXiv:2509.16941
- Singh et al., "The Leaderboard Illusion", Apr 2025. arXiv:2504.20879
- Phan et al., "Humanity's Last Exam", Jan 2025. arXiv:2501.14249
- Chollet et al., "ARC-AGI-2", May 2025. arXiv:2505.11831
- Arora et al., "HealthBench", May 2025. arXiv:2505.08775
- Kalai et al., "Why Language Models Hallucinate", Sep 2025. arXiv:2509.04664
- METR, arXiv:2503.14499 and "Time Horizon 1.1" (Jan 2026)
- Anthropic, Claude Sonnet 4.5 System Card, Sep 2025

---

## Ch 24 — Translating frontier JDs into a skill stack

**Verdict:** Light touch. The chapter's method (JDs are wish lists, interviews test arithmetic and procedure) is evergreen. The decoder ring and the evidence-building advice are missing the 2025–26 vocabulary.

### Findings (ranked most-severe first)

1. **MISSING-MAJOR** · `book/part-5-the-job/24-translating-jds.md:17-37` (decoder ring) · Phrases common in 2026 postings are absent. Suggested rows:
   - "RL environments" / "universes" / "environments infrastructure", mapping to Ch 18 §18.6. Anthropic has open postings with exactly these titles: "Research Engineer, Universes"; "Staff Software Engineer, Environments Infrastructure"; "Research Engineer, Code / Cybersecurity / Chip Design / Performance RL", each "designing and implementing RL environments… reward signals and verifiers".
   - "Reward hacking" / "verifier design", mapping to Ch 12 and Ch 18.
   - "MCP" / "tool integration" / "agent harness", mapping to Ch 18 §18.2.
   - "On-policy distillation", mapping to Ch 19 (DeepSeek-V4 and the Thinking Machines blog).
   - "Evals" as a stand-alone team, mapping to Ch 23.

   The row "Verifiable rewards / RLVR — this team read the R1 paper" (line 31) is now generic. RLVR is table stakes, so the informative 2026 phrases are "environments" and "long-horizon".
   - Source: Anthropic Greenhouse postings, e.g. https://job-boards.greenhouse.io/anthropic/jobs/5061517008 , https://job-boards.greenhouse.io/anthropic/jobs/5254364008 , https://job-boards.greenhouse.io/anthropic/jobs/5412334008 (accessed 26 Sep 2026)

2. **MISSING-MAJOR** · `:142-156` (§24.6 "Building evidence without a cluster") · Two new routes a 2026 reader would want:
   - (a) **Contribute an RL environment.** Prime Intellect's Environments Hub plus the `verifiers` library is an open, reviewable venue whose environments feed an open frontier-ish training run (INTELLECT-3). It is the environment-engineering analogue of "a merged PR to vLLM". Prime Intellect also runs an open-source environments programme.
   - (b) **Run real RL on large open models without a cluster** via hosted training APIs. Tinker (Thinking Machines, 1 Oct 2025) supports LoRA fine-tuning and RL on open-weight models up to large MoEs such as Qwen-235B-A22B, with an open cookbook.

   Both weaken the claim that "a Colab T4 is enough" is the only affordable path (line 154). It still is for mechanisms.
   - Source: https://www.primeintellect.ai/blog/environments ; https://github.com/PrimeIntellect-ai/verifiers ; https://www.primeintellect.ai/blog/scaling-environments-program ; arXiv:2512.16144
   - Source: Thinking Machines Lab, "Announcing Tinker", 1 Oct 2025, https://thinkingmachines.ai/news/announcing-tinker/

3. **DATED-FRAMING** · `:152` (§24.6 "Take the R1 paper… write down what it does not tell you (§19.8)") · Per Ch 19 finding 1, much of that list was published in *Nature*. Better exercise: compare the arXiv and *Nature* versions, or use Olmo 3 as the fully open control.

4. **MISSING-MINOR** · `:148`, `:174` (the list of open projects to contribute to: vLLM, SGLang, trl, torchtitan, Megatron, FlashAttention, lm-evaluation-harness) · Add the RL-infrastructure projects the 2025 recipes are built on, which are now common JD keywords: verl/HybridFlow (arXiv:2409.19256), slime (THUDM, used for GLM-4.5), AReaL (arXiv:2505.24298), prime-rl and verifiers (Prime Intellect).

5. **MISSING-MINOR** · `:163` ("This book cites around 96") · Accurate: `book/appendix/b-references.md` has 96 numbered entries (the audit brief said 95). It will need updating after the refresh adds references.

### Keep as-is
- §24.1 (why JDs read the way they do), §24.3's four archetypes and interview tests, §24.4 stacks, §24.5 (what interviews test), §24.7, and the whole "arithmetic compounds" thread. None of it depends on dated facts.

### References to add
- Anthropic job postings (RL environments roles), accessed 26 Sep 2026 (URLs above)
- Thinking Machines Lab, "Announcing Tinker", 1 Oct 2025
- Prime Intellect, Environments Hub (2025) and `verifiers`; INTELLECT-3 tech report, Dec 2025, arXiv:2512.16144
- Sheng et al., "HybridFlow" (verl), Sep 2024, arXiv:2409.19256 ; Fu et al., "AReaL", May 2025, arXiv:2505.24298

---

## Ch 25 — Career paths at frontier labs

**Verdict:** Light touch. The role map, the PhD analysis and "learn the structure, not the recipe" hold up. The "where to work" list and a few scarcity annotations are dated.

### Findings (ranked most-severe first)

1. **DATED-FRAMING** · `book/part-5-the-job/25-career-paths.md:21` ("Environment Engineer (agentic) | … | Very scarce, very new") · This has come true and is no longer "very new". It is an established hiring category with multiple named postings at one lab alone (Code RL, Cybersecurity RL, Chip Design RL, Performance RL, Universes, Environments Infrastructure), and the open ecosystem has a public environments hub. Suggest "Scarce; established since 2025". The same goes for §26.3's prediction (see Ch 26 finding 1).
   - Source: Anthropic Greenhouse postings (above); Prime Intellect Environments Hub

2. **DATED-FRAMING** · `:107-109` (§25.6 employer list) · Missing entities a 2026 reader would expect:
   - **Meta's reorganisation into Meta Superintelligence Labs** (30 Jun 2025). "Meta" as a single entry hides that the org, and its open-weights stance, changed.
   - **Thinking Machines Lab**: founded 2025, whose first product Tinker is aimed at researchers.
   - **Prime Intellect**, a decentralised open RL stack, relevant to the "open source" bullet.
   - **Z.ai**: Zhipu's models ship under `zai-org`, per the GLM-4.5 repo link in its report.
   - **MiniMax and StepFun** in the Chinese list. MiniMax-M1 and Step-3 publish serving-relevant technical reports.

   AI2's openness point (line 109) is *strengthened* by Olmo 3's "entire model flow" release.
   - Source: Thinking Machines, https://thinkingmachines.ai/news/announcing-tinker/ ; GLM-4.5 arXiv:2508.06471 (repo `zai-org/GLM-4.5`) ; MiniMax-M1 arXiv:2506.13585 ; StepFun Step-3 arXiv:2507.19427 ; Olmo 3 arXiv:2512.13961 ; INTELLECT-3 arXiv:2512.16144
   - The MSL formation is documented by the memo text as published by CNBC (30 Jun 2025). This is a secondary publication of a primary memo; see Unverified leads.

3. **DATED-FRAMING** · `:121` (§25.7 "The most detailed public technical reports of the last two years — DeepSeek-V3 and R1, Qwen3, Kimi k1.5") · Still true in spirit, and the list should be updated to Kimi K2 and K2.5, GLM-4.5, DeepSeek-V3.2 and V4, and MiniMax-M1. Balance it with the fully open US releases (Olmo 3, INTELLECT-3) and OpenAI's gpt-oss, which changes the "open-weight ecosystem is substantially built on them" sentence (line 127).
   - Source: as cited in Ch 18 and Ch 19

4. **MISSING-MINOR** · `:133` (§25.8 "GRPO went from a paper to standard practice, DPO… one option among six, verifiable rewards… the organizing principle") · Add the 2025–26 turnover as the next example of the same pattern: on-policy distillation replacing a mixed RL stage (DeepSeek-V4), generative and rubric reward models replacing scalar RMs, and environments replacing prompt sets as the scarce input. This supports the chapter's thesis rather than contradicting it.

5. **MISSING-MINOR** · `:113` (§25.6 "The reproductions of R1… came out of this world within weeks, and several of the people who did them were hired shortly after") · I could not verify this from a primary source. Flag it for a citation or softening.

### Keep as-is
- §25.1's structure and the "crowded research versus scarce engineering" insight, §25.2 (the PhD question), §25.3 (entry points), §25.4 (transitions), §25.5 (seniority), §25.8's "what compounds". All are argument, not dated fact.

### References to add
- Thinking Machines Lab, "Announcing Tinker", 1 Oct 2025
- Team Olmo, "Olmo 3", Dec 2025, arXiv:2512.13961
- Prime Intellect, "INTELLECT-3: Technical Report", Dec 2025, arXiv:2512.16144
- Anthropic RL-environment job postings (accessed 26 Sep 2026)

---

## Ch 26 — What's next: multi-modal, agent RL, synthetic data 2.0

**Verdict:** Needs refresh. Several predictions have resolved, and the chapter should say so explicitly. That matches the book's own "fix what is wrong here" contract and the honesty-over-polish norm. There is also one internal factual inversion.

### Prediction scorecard (as of 2026-09-26)

| Line | Prediction | Status | Evidence (primary) |
|---|---|---|---|
| :38 | Environments become the contested resource; "less published than model architecture" | **Came true on investment; falsified on publication.** Labs now publish environment infrastructure in detail. | Kimi K2 (arXiv:2507.20534), Qwen3-Coder blog (22 Jul 2025), DeepSeek-V3.2 (1,800 envs, 85k prompts), DeepSeek-V4 DSec (hundreds of thousands of sandboxes), Prime Intellect Hub |
| :38 | Environment engineering among the fastest-growing job categories | **Came true** (qualitatively; no growth numbers from a primary source) | Anthropic postings (above) |
| :40 | Computer use as the general interface | **Partly.** Agents did go general, but mostly through terminal and code (Terminal-Bench 2.0, SWE-Bench Pro) and MCP tools, not only pixels | arXiv:2601.11868; AAIF/MCP (9 Dec 2025) |
| :42 | Reliability (pass^k) rather than capability | **Still open; better measured.** METR's 50% versus 80% horizons make the reliability gap explicit | METR TH1.1 |
| :44 | Multi-agent "more prompting pattern than training target… unsettled" | **Falsified in part.** Kimi K2.5 trains the orchestrator with PARL | arXiv:2602.02276 |
| :56 | Curriculum generation "high-leverage and underexplored" | **Now actively explored**: self-play proposer/solver with zero external data | Absolute Zero, arXiv:2505.03335 |
| :66 | "FP4 is being explored" | **Superseded.** FP4 shipped (gpt-oss MXFP4 weights; DeepSeek-V4 FP4 QAT) and NVFP4 pre-training was reported | arXiv:2508.10925; arXiv:2606.19348 §5.2.1; arXiv:2509.25149 |
| :68 | Finer-grained sparsity "not yet routine" | **Partly superseded for attention.** Sparse attention is in production open models (DSA in V3.2, CSA in V4) | arXiv:2512.02556; arXiv:2606.19348 |
| :82 | Linear/SSM/hybrid alternatives "have not clearly won" | **Contested; now claimed won.** Kimi Linear claims to outperform full attention under fair comparison; MiniMax-M1 is hybrid; DeepSeek went compressed-sparse | arXiv:2510.26692; arXiv:2506.13585; arXiv:2606.19348 |
| :88-92 | Interpretability entering the training loop | **Came true (early).** Persona vectors monitor and *prevent* trait shifts during fine-tuning and flag training data; the Sonnet 4.5 audit used interpretability on eval awareness over training | arXiv:2507.21509; Sonnet 4.5 system card |
| :122 | "No published scaling law" for the RL compute split | **Partly answered.** ScaleRL gives predictive RL-compute scaling (400k+ GPU-hours); the full pre-training/mid-training/RL split is still unpublished | arXiv:2510.13786 |
| :120 | Is the base model the ceiling? | **Still open; now a live empirical debate** | arXiv:2504.13837 versus arXiv:2505.24864 |

### Findings (ranked most-severe first)

1. **WRONG-NOW (internal error)** · `book/part-5-the-job/26-whats-next.md:112` · "**Data is the product.** Chapter 19 §19.8's observation about what labs withhold: the algorithm, not the data." This is inverted. §19.8 (line 168) concludes that labs withhold "**the data, not the algorithm**", and the bold heading "Data is the product" only makes sense with that order. Fix independently of the refresh. Note also that per Ch 19 finding 1, R1's *Nature* version published RL prompt counts and hyperparameters, so the sentence should be qualified anyway.

2. **WRONG-NOW** · `:66` · "FP4 is being explored." See the scorecard. FP4 is in shipped models and training pipelines.
   - Sources: arXiv:2508.10925 (Aug 2025), arXiv:2509.25149 (Sep 2025), arXiv:2606.19348 (Apr 2026)

3. **DATED-FRAMING** · `:38` and `:221` of Ch 18 ("less published") · Falsified as a publication prediction (scorecard row 1). Rewrite as a resolved prediction. This is the book's most-cited forecast and the evidence now exists to show it.

4. **DATED-FRAMING** · `:44` · Multi-agent training is now a published training target (Kimi K2.5 PARL).

5. **DATED-FRAMING** · `:80-84` (§26.6 "Attention is still quadratic… alternatives have not clearly won") · DeepSeek's V3.2 and V4 production models use sparse or compressed attention. V4 is explicitly built for 1M-token routine contexts at 10% of V3.2's KV cache. Kimi Linear claims a fair-comparison win. The reframe paragraph ("long context may be the wrong abstraction") remains a good open question; the premise needs updating.

6. **MISSING-MAJOR** · `:118-128` (§26.9 open questions) · Add the 2025–26 evidence to each:
   - base-model ceiling: Yue et al. versus ProRL;
   - compute split: ScaleRL;
   - agentic reliability: METR 50% versus 80% horizons; Kimi K2 Thinking's 200–300 step tool use;
   - "evaluate a system you cannot out-think": eval awareness (Sonnet 4.5 card), the fragility of CoT monitorability (arXiv:2507.11473), and emergent misalignment from reward hacking (arXiv:2511.18397).

   One new open question a 2026 reader would expect: **is RL even the consolidation step?** DeepSeek-V4 replaced its final mixed RL stage with on-policy distillation from RL-trained specialists.

7. **MISSING-MINOR** · `:15-19` (multi-modal "native generation… not settled") · I did not audit multimodal primary sources in depth. Kimi K2.5 (Feb 2026) reports joint text-vision pre-training and RL, where "visual RL improves text performance". That is a concrete data point for "what is settled".
   - Source: arXiv:2602.02276 §2.3

8. **MISSING-MINOR** · `:151` (closing note: "The DeepSeek-V3 and R1 reports, the Llama 3 paper, the Tülu 3 recipe") · Add R1-in-*Nature* and a 2025–26 fully open recipe (Olmo 3) to the "go read the primary sources" list.

### Keep as-is
- §26.1's calibration paragraph, including "underestimating how fast the boring engineering problems get solved". FP4 shipping, EP serving and environment infrastructure all bear it out, so cite them as confirmation.
- §26.4's "synthetic data works where verification works" and the contamination-laundering warning (confirmed by SWE-bench Verified).
- §26.8 "What will not change", apart from the line-112 inversion.
- The whole closing note.

### Lab opportunity (optional)
**`lab26_prediction_scorecard`**: not a mechanism lab. Keep the scorecard table above as a versioned data file (`book/appendix/predictions.yaml` or similar), with a CI check that every prediction has a status and a primary-source URL. It fits the book's honesty-over-polish norm and makes future staleness audits mechanical.

### References to add
- Zhao et al., "Absolute Zero: Reinforced Self-play Reasoning with Zero Data", 6 May 2025. arXiv:2505.03335
- Chen et al. (Anthropic), "Persona Vectors: Monitoring and Controlling Character Traits in Language Models", 29 Jul 2025. arXiv:2507.21509
- Kimi Team, "Kimi Linear", 30 Oct 2025. arXiv:2510.26692
- MiniMax, "MiniMax-M1", 16 Jun 2025. arXiv:2506.13585
- NVIDIA, "Pretraining LLMs with NVFP4", 29 Sep 2025. arXiv:2509.25149
- Khatri et al., "The Art of Scaling RL Compute for LLMs", 15 Oct 2025. arXiv:2510.13786
- Yue et al., arXiv:2504.13837 ; Liu et al., "ProRL", arXiv:2505.24864
- DeepSeek-V3.2 (arXiv:2512.02556), DeepSeek-V4 (arXiv:2606.19348), Kimi K2.5 (arXiv:2602.02276)
- METR, "Time Horizon 1.1", 29 Jan 2026

---

## Unverified leads

I could not confirm these from a primary source during this audit. Do not cite them without checking.

- **Exact METR horizons for 2026 models.** Secondary sources claim Claude Opus 4.5 at about 4h49m (TH1) or about 5h20m (TH1.1) and later models at 12–14 h. I confirmed only GPT-5 at about 2h17m (METR FAQ) and the TH1.1 doubling times. Check https://metr.org/time-horizons/ directly.
- **SWE-bench Verified audit figure.** "At least 59.4% of the audited 27.6% subset have flawed tests." The number appears in secondary coverage and the OpenAI page's truncated text; I confirmed the 27.6% subset and the 74.9% to 80.9% SOTA from the page, not the 59.4% itself.
- **Prime Intellect Environments Hub launch date.** The blog shows no machine-readable date; late Aug 2025 is plausible but unconfirmed. The "2,500+ environments" count comes from a search snippet, not the page.
- **Meta Superintelligence Labs.** The formation (30 Jun 2025) is documented by the memo text reproduced by CNBC, not by a Meta-hosted page I could open. The Scale AI stake figures are press-reported only.
- **OpenAI Operator / Computer-Using Agent (Jan 2025)** and **OSWorld / OSWorld-Verified** scores. The OpenAI page did not load and I did not verify OSWorld numbers.
- **Chatbot Arena rebrand to LMArena**, and the Llama 4 "experimental variant" Arena episode (Apr 2025). Not verified from LMArena or Meta primary pages.
- **GPQA Diamond saturation levels** at the frontier (claimed above 85–90%). Not verified from a primary model card in this audit.
- **The Ch 25 §25.6 claim** that R1 reproducers "were hired shortly after". No source found.
- **DeepSeek-V4 details** beyond what I quoted (e.g. whether V4 publishes RL prompt counts or compute like R1-*Nature*). The report is long and I only sampled §3.5.2, §5.1 and §5.2.
- **Kimi K2 Thinking "200–300 sequential tool calls"** is confirmed from Moonshot's own model card, but it is a vendor claim with no independent measurement.
- **Terminal-Bench 1.0 (May 2025)**: the date of the original release. I confirmed only the 2.0 paper (Jan 2026).
