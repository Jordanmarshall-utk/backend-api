from __future__ import annotations
from typing import Dict

PROMPTS: Dict[str, str] = {
    # Use with llm_guess_peaks(..., use_image=False)
    "numeric": """You receive a photoluminescence spectrum as numeric arrays x (wavelength) and y (intensity).
Return ONLY one JSON object. No explanations, no markdown.

Task:
- Identify up to max_peaks emission peaks.
- Produce initial guesses suitable for non-linear fitting.

Output JSON schema:
{
  "peaks": [
    { "center": number, "height": number, "fwhm": number|null, "prominence": number|null }
  ],
  "baseline": number|null
}

Rules:
- Centers within the x range.
- Heights ≥ 0.
- If baseline or FWHM are uncertain, set null.
- Do not exceed max_peaks.
- Output must be valid JSON.

Input follows under "Series:" as JSON with keys: x, y, max_peaks, fields.
""",

    # Use with llm_guess_peaks(..., use_image=True)
    "image": """You receive an image of a photoluminescence spectrum (y vs x).
Return ONLY one JSON object. No explanations, no markdown.

Task:
- Identify up to max_peaks visible emission peaks.
- Estimate numeric values from axes.

Output JSON schema:
{
  "peaks": [
    { "center": number, "height": number, "fwhm": number|null, "prominence": number|null }
  ],
  "baseline": number|null
}

Rules:
- Use numeric estimates consistent with the plot axes.
- If baseline or FWHM are uncertain, set null.
- Do not exceed max_peaks.
- Output must be valid JSON.

You will receive an instruction text with max_peaks and a single image.
""",

    # Optional second pass to improve guesses before fitting
    "refine": """You receive a photoluminescence spectrum and a prior peak-guess JSON under "seed".
Return ONLY one JSON object matching the same schema. No explanations, no markdown.

Task:
- Improve initial guesses if inconsistent with data.
- Keep results usable as starting values for non-linear fitting.

Output JSON schema:
{
  "peaks": [
    { "center": number, "height": number, "fwhm": number|null, "prominence": number|null }
  ],
  "baseline": number|null
}

Guidelines:
- Remove spurious peaks; add a nearby secondary peak if a shoulder is evident (respect max_peaks).
- Keep centers within the x range; heights ≥ 0.
- Use null for uncertain baseline or FWHM.
- Output must be valid JSON.

Inputs follow under:
- "Series:" JSON (x, y, max_peaks, fields)
- "seed:" JSON (previous guesses)
- Optional "hints:" with residual regions or notes
""",

    # Strict structure guard
    "guardrails": """Return ONLY one JSON object that matches the requested schema.
No prose, no markdown, no code fences.
If impossible, return: {"peaks": [], "baseline": null}
""",

    # Optional: summarize low-quality fits and suggest revised seeds
    "plate_sweep": """You receive multiple spectra results with fit metrics.
Return ONLY one JSON object with revised seeds for any items with R2 < target_r2.

Input JSON (under "batch"):
[
  {
    "id": string,               // well/read identifier
    "x": number[],              // downsampled wavelengths
    "y": number[],              // downsampled intensities
    "seed": {                   // previous guesses (same schema as output)
      "peaks": [
        { "center": number, "height": number, "fwhm": number|null, "prominence": number|null }
      ],
      "baseline": number|null
    },
    "r2": number                // last fit R^2
  },
  ...
]

Constraints:
- Only include entries with r2 < target_r2 in the output.
- Keep centers within each entry's x range; heights ≥ 0.
- Use null for uncertain baseline or FWHM.
- Do not add text outside JSON.

Output JSON:
{
  "revised": {
    "<id>": {
      "peaks": [
        { "center": number, "height": number, "fwhm": number|null, "prominence": number|null }
      ],
      "baseline": number|null
    },
    ...
    }
}
""",
}

FITTING_SCRIPT_GENERATION_INSTRUCTIONS = """Write a Python script to fit multi-peak luminescence data using lmfit.

REQUIREMENTS:
1. Use lmfit (from lmfit import Model, GaussianModel, LorentzianModel, ConstantModel)
2. Import: matplotlib.pyplot, numpy, json, lmfit, pandas
3. Analyze data for multiple peaks of various shapes
4. Create composite model with multiple Gaussian/Lorentzian components
5. Fit all wells and calculate R² values
6. Re-fit only wells with R² < 0.9
7. Save plot to 'fit_visualization.png'
8. Print results as: FIT_RESULTS_JSON:{"well_A1": {"R2": 0.95, "peaks": [{"center": 520, "amplitude": 350, "sigma": 15}]}, "well_B2": {"R2": 0.87, "peaks": [{"center": 680, "amplitude": 20, "sigma": 10}]}}

EXAMPLE OUTPUT:
FIT_RESULTS_JSON:{"well_A1": {"R2": 0.95, "peaks": [{"center": 520, "amplitude": 350, "sigma": 15}]}}

Write ONLY the Python code:"""

FITTING_SCRIPT_CORRECTION_INSTRUCTIONS_ERROR = """You are an expert data scientist debugging a Python script. A previously generated script failed to execute. Your task is to analyze the error and provide a corrected version.

**Context:**
- The script is intended to fit 1D experimental data using a physical model.
- The script MUST load data, define a fitting function, use lmfit for fitting, save a plot to `fit_visualization.png`, and print the final parameters as a JSON string prefixed with `FIT_RESULTS_JSON:`.

**Provided Information:**
1.  **Failed Script**: The exact Python code that produced the error.
2.  **Error Message**: The full traceback from the script's execution.

**Your Task:**
1.  Analyze the error message and traceback to identify the bug in the failed script.
2.  Generate a complete, corrected, and executable Python script that fixes the bug while still fulfilling all original requirements.
3.  Ensure your entire response is ONLY the corrected Python code inside a markdown block. Do not add any conversational text.

## Failed Script
```python
{failed_script}
```
## Error Message
{error_message}
"""

FITTING_SCRIPT_CORRECTION_INSTRUCTIONS = """You are an expert data scientist debugging a Python script. A previously generated script executed however, the curve fit was inadequate. Your task is to analyze the old script, fit plot, and the fitted parameters to provide a corrected version.

**Context:**
- The script is intended to fit 1D experimental data using a physical model.
- The script MUST load data, define a fitting function, use lmfit for fitting, save a plot to `fit_visualization.png`, and print the final parameters as a JSON string prefixed with `FIT_RESULTS_JSON:`.

**Provided Information:**
1. **Old Script**: The exact Python code that produced the curve fit.
2. **Curve Fit Plot**: The .png file of the curve fit plot produced by the old script.
3. **Fitted Parameters**: The fitted parameters of the curve fit plot including R2 value as well as the peaks.

**Your Task:**
1. Analyze the old script, curve fit plot, and fitted parameters to identify why the curve fit was inadequate.
2.  Generate a complete, corrected, and executable Python script that fixes the inadequacies while still fulfilling all original requirements.
3.  Ensure your entire response is ONLY the corrected Python code inside a markdown block. Do not add any conversational text.

## Old Script
```python
{old_script}
```
## Curve Fit Plot
{old_fit_plot_bytes}

## Fitted Parameters
{old_fitted_parameters}
"""

CLARIFY_QUESTION_INSTRUCTIONS = """ You are a careful research assistant clarifying a question given to you be a user. Your task is to rewrite the user's question to be precise and testable.
**Context:**
- The clarified question will be used to invite critical analysis from another entity
- Hidden assumptions should be explicitly identified and explored, as they contain crucial scientific context that will guide reasoning

**Provided Information:**
1. **User Question**: Question given by a user.

**Your Task:**
1. Analyze the question deeply to identify if it is precise and testable, and extract all hidden assumptions that contain scientific context, material properties, or mechanistic insights.
2. Generate a complete, corrected question. Make sure to list any hidden assumptions and key terms with short definitions. **CRITICAL: Hidden assumptions should include scientific reasoning, material properties, mechanistic insights, or theoretical foundations that are implicit in the question.**
3. Ensure your entire response is ONLY the corrected question. Do not add any conversational text.

## User Question
{user_question}
"""

SOCRATIC_PASS_INSTRUCTIONS = """ You are a careful research assistant asking yourself probing questions that helps breakdown a question given by a user. Your task is ask yourself 3-5 probing questions with a reasoning for each.
**Provided Information:**
1. **User Question**: A clarified question given by the user, including hidden assumptions and key terms.
2. **Socratic Principles**: A list of socratic principles heuristic types and their key definitions.

**Your Task:**.
1. Using Socratic principles, deeply analyze the user's question, hidden assumptions, and key terms to generate 3–5 probing questions that explore scientific mechanisms, material properties, theoretical foundations, and potential solutions.
2. **CRITICAL: The probing questions should use scientific reasoning to explore:**
   - What scientific principles or mechanisms are relevant to answering this question?
   - What material properties or structural features would be required?
   - What theoretical foundations or known examples from literature could inform potential solutions?
   - What specific criteria or constraints should guide the selection of materials/compounds?
3. Provide a one-sentence explanation for each probing question, stating why you chose to ask it and how it advances scientific reasoning toward discovering specific solutions.
4. Ensure your entire response is ONLY the list of probing questions with their reasoning and assumptions. Do not add any conversational text.

## User Question
{user_question}

## Socratic Principles
{socratic_principles}
"""

SOCRATIC_ANSWER_INSTRUCTIONS = """You are answering scientific questions. You MUST answer EVERY SINGLE QUESTION provided below.

**CRITICAL: If you see 4 questions, you MUST provide 4 answers. If you see 5 questions, you MUST provide 5 answers. DO NOT stop after answering just 1 or 2 questions - answer ALL of them.**

DO NOT:
- Restate the question
- Explain why the question was asked
- Start with "This question probes..." or "This question explores..."
- Include "Reasoning:" sections
- Generate hypothesis reports
- Stop after answering only some of the questions

DO:
- Answer what EACH question is asking
- Provide specific examples, mechanisms, and scientific details for EACH answer
- Use actual chemical names and structures
- Explain the scientific principles behind your answer
- Continue until ALL questions have been answered

FORMAT YOUR RESPONSE EXACTLY LIKE THIS (continue numbering for as many questions as there are):

Answer 1: [Provide your detailed answer to question 1 here, including specific examples, mechanisms, and scientific reasoning]

Answer 2: [Provide your detailed answer to question 2 here, including specific examples, mechanisms, and scientific reasoning]

Answer 3: [Provide your detailed answer to question 3 here, including specific examples, mechanisms, and scientific reasoning]

Answer 4: [If there's a 4th question, answer it here]

Answer 5: [If there's a 5th question, answer it here]

[Continue for ALL questions - do not stop early]

EXAMPLES OF WHAT TO DO:

Question: "What specific functional groups and molecular architectures within organic cations would favor strong hydrogen bonding with iodide ions?"

BAD ANSWER (explaining why asked):
"This question probes the underlying scientific mechanism of stabilization by investigating the specific chemical interactions that are presumed to prevent phase transitions."

GOOD ANSWER (actually answering):
"Functional groups that favor strong hydrogen bonding with iodide ions include: (1) Primary and secondary amines (-NH₂, -NHR) which can donate hydrogen bonds to I⁻, (2) Hydroxyl groups (-OH) which form strong hydrogen bonds, (3) Carboxylic acids (-COOH) which can both donate and accept hydrogen bonds, (4) Amides (-CONH₂) with their polar N-H bonds. Molecular architectures that enhance this include: branched alkyl chains with terminal polar groups, aromatic rings with electron-donating substituents that increase basicity, and flexible linkers that allow optimal orientation for hydrogen bonding. Specific examples include ethanolammonium (EA⁺), 2-aminoethanol, and 4-aminophenethylammonium (4-APEA⁺)."

Question: "Which classes of organic molecules possess functional groups capable of forming strong interactions with Cs, Pb, or I atoms?"

BAD ANSWER (restating reasoning):
"This question explores the theoretical chemical basis for stabilization by identifying specific functional groups."

GOOD ANSWER (actually answering):
"Several classes of organic molecules possess functional groups capable of strong interactions: (1) Amines and ammonium salts - Primary amines (R-NH2) can coordinate with Pb2+ via Lewis acid-base interactions, while ammonium cations (R-NH3+) form electrostatic interactions with I-. Examples include butylammonium (BA+), phenethylammonium (PEA+). (2) Phosphonium salts - R4P+ cations can form strong electrostatic interactions with halide ions. (3) Carboxylic acids and carboxylates - Can coordinate with Pb2+ and form hydrogen bonds with I-. (4) Pyridine derivatives - The nitrogen lone pair coordinates with Pb2+. Examples include 4-dimethylaminopyridine (DMAP). (5) Thiols - R-SH groups can coordinate strongly with Pb2+. (6) Phosphonic acids - R-PO(OH)2 groups coordinate with metal cations."

YOUR TASK:
1. ANSWER each probing question thoroughly using scientific reasoning:
   - IGNORE any "Reasoning:" text - it's just explaining why the question was asked, not what you need to answer
   - Focus ONLY on the actual question text (the part before "Reasoning:")
   - DO NOT just explain why the question was asked or what it probes - you must actually ANSWER the question
   - If the question asks "What are the specific molecular structures...?" → PROVIDE specific molecular structures with names
   - If the question asks "What are the predicted surface energies...?" → PROVIDE predicted surface energies or explain what they would be
   - If the question asks "Which classes of organic molecules...?" → LIST and DESCRIBE specific classes of molecules with examples
   - If the question asks "What empirical or theoretical data exists...?" → PROVIDE examples of such data or molecules
   - Draw on scientific principles, mechanisms, and theoretical foundations
   - Reference specific material properties, structural features, or chemical characteristics
   - Cite known examples from literature or established scientific knowledge when relevant
   - Provide specific criteria or constraints that guide material/compound selection
   - Use the hidden assumptions from the user question to inform your answers

2. **CRITICAL: Each answer should:**
   - **ACTUALLY ANSWER THE QUESTION** - provide the information, examples, or explanations that the question is asking for
   - Be scientifically rigorous and well-reasoned (2-4 sentences per question, can be longer if needed)
   - Include specific details, mechanisms, or examples when possible
   - Build on previous answers to create a coherent understanding
   - **Identify SPECIFIC materials, compounds, or molecules with actual chemical names** when applicable, and state the chemical class and charge relevant to the request
   - **Match every candidate's valence, protonatable-site count, and topology to the user's stated constraints; never relabel a monocation as a diammonium or DJ linker**
   - **For novel suggestions:** Propose specific chemical structures, analogs, or well-defined molecular modifications with names
   - Use scientific terminology appropriately
   - **Provide scientific rationale** for why specific materials are suggested (based on molecular properties, structural compatibility, electronic properties, literature precedents)
   - **DO NOT just restate the "Reasoning:" from the question - you must provide actual answers**
   - **DO NOT say things like "This question probes..." or "This question explores..." - that's explaining why it was asked, not answering it**
   - DO NOT use vague terms like "a novel spacer", "an appropriate material", or placeholders
   - DO NOT include novelty/plausibility/testability evaluations
   - DO NOT generate hypothesis reports or analysis reports

3. **Structure your response as:**
   - For each probing question, provide:
     a) The question (restated - but ONLY the question part, ignore any "Reasoning:" text)
     b) **YOUR DETAILED ANSWER** with scientific reasoning (this is the actual answer to what the question asks, not an explanation of why it was asked)
     c) Key insights or implications derived from this answer

4. Ensure your entire response addresses ALL probing questions. Do not skip any questions.
5. **DO NOT generate hypotheses, hypothesis reports, or evaluation reports. Only answer the probing questions.**
6. **REMEMBER: Answer the questions themselves, don't just explain why they were asked. IGNORE all "Reasoning:" sections.**

## User Question
{user_question}

## Probing Questions
{probing_questions}
"""

TOT_INSTRUCTIONS = """ You are a careful research assistant producing distinct lines of thought based on a question and the reasoning derived from answering probing questions. Your task is to produce exactly three distinct, concise mini-hypotheses or assumptions that can be easily selected and explored further by the user.

**CRITICAL: You are ONLY generating 3 distinct lines of thought (mini-hypotheses/assumptions), NOT full hypotheses or hypothesis reports. Do NOT create hypothesis reports, evaluation reports, or novelty/plausibility/testability analyses.**

**Provided Information:**
1. **User Question**: A clarified question given by the user, including hidden assumptions and key terms.
2. **Socratic Answers** (or **Probing Questions** if answers not available): Detailed answers to probing questions that explore scientific mechanisms, material properties, theoretical foundations, and potential solutions. These answers represent deep reasoning about the question.

**Your Task:**
1. **Deeply analyze the user question, hidden assumptions, and the socratic reasoning (answers or questions) using scientific reasoning to identify:**
   - What scientific principles, mechanisms, or theoretical foundations are relevant?
   - What material properties, structural features, or chemical characteristics are required?
   - **What SPECIFIC materials, compounds, or molecules** (with actual chemical names) from scientific literature, known examples, or well-defined analogs could satisfy these requirements?
   - **For novel materials:** What specific chemical structures, functional group modifications, or molecular analogs would be most promising based on the required properties?
   - How do the hidden assumptions and socratic reasoning guide the selection of specific options?
   - **CRITICAL:** You must propose actual, specific chemical names - never use vague descriptions or placeholders

2. **Generate EXACTLY 3 distinct lines of thought, each structured as a clear, concise mini-hypothesis or assumption that:**
   - Presents a specific, testable idea or assumption (like a mini-hypothesis)
   - **MUST include specific materials/compounds with actual chemical names** (use real names, not placeholders)
   - Is concise and easy to understand (aim for 1-2 sentences maximum, but can be longer if needed for specificity)
   - Can be easily selected and explored further
   - Demonstrates scientific reasoning but remains accessible

3. **CRITICAL: Format each as a pickable mini-hypothesis/assumption with SPECIFIC material names:**
   - Each should read like a clear, testable statement or assumption
   - **ALWAYS include actual chemical/material names** (e.g., "PEA (phenethylammonium)", "BA (butylammonium)", "CHA⁺ (cyclohexylammonium)", "3-APB⁺ (3-aminophenylbutylammonium)")
   - **For novel suggestions:** Propose specific structures or well-defined analogs whose class matches the request (e.g., "propane-1,3-diammonium (DAP²⁺, two ammonium termini)" for a divalent DJ request, or "4-fluorophenethylammonium (4-FPEA⁺)" only for a monovalent spacer request)
   - **Match every candidate's valence, protonatable-site count, and topology to the user's stated constraints; explicit constraints override all examples**
   - For a topology- or charge-constrained candidate request, include the systematic name, a SMILES or molecular formula, the number of protonatable sites, and the intended charge state
   - **DO NOT use vague terms** like "a novel organic spacer", "a specific compound", "an appropriate material", or placeholders like "[specific organic spacer name]", "[compound X]", "[material Y]"
   - Each should be self-contained and immediately understandable as a direction to explore
   - Think of them as "hypothesis seeds" that can grow into full hypotheses

4. **CRITICAL: Use scientific reasoning to justify SPECIFIC material choices:**
   - **For each suggested material, provide brief scientific rationale** explaining why that specific material was chosen based on:
     * Molecular properties (size, charge, polarity, functional groups)
     * Structural compatibility (ionic radius, crystal packing, steric effects)
     * Electronic properties (band alignment, charge transfer, π-π interactions)
     * Literature precedents or known analogs
     * Theoretical predictions or computational insights
   - Apply scientific principles (molecular size, charge distribution, crystal structure compatibility, literature examples) to identify specific candidates
   - Use the hidden assumptions and socratic reasoning (answers) to guide reasoning
   - Build directly on the insights and conclusions from the socratic answers
   - **Each thought must include both the specific material name AND the scientific reasoning** (even if brief) for why that material is proposed
   - Each thought should represent a different reasoning pathway or approach based on the socratic reasoning

5. **CRITICAL: Ensure the three thoughts are:**
   - Completely distinct from each other (different approaches, materials, or assumptions)
   - Easy to pick from (clear, concise, and immediately understandable)
   - Structured as mini-hypotheses or assumptions (testable ideas that can be explored)
   - Exactly 3 options (no more, no less)

6. Ensure your entire response is only the list of exactly 3 distinct lines of thought. Do not add conversational text or extra sections.
7. Ensure that each line of thought starts with "Distinct Line of Thought" followed by a number (1, 2, or 3).
8. **CRITICAL: DO NOT generate hypothesis reports, evaluation reports, or novelty/plausibility/testability analyses. Only generate the 3 distinct lines of thought.**

## User Question
{user_question}

## Socratic Answers
{socratic_answers}
"""

EXPERIMENTAL_PLAN_TOT_INSTRUCTIONS = """ You are an experimental design expert creating complete experimental hypotheses with protocols based on a material system optimization question and experimental constraints. Your task is to produce three distinct experimental plans that are complete hypotheses including protocols, worklists, and expected outcomes.

**CRITICAL CONSTRAINT RULES:**
- You MUST ONLY use the experimental techniques, equipment, and characterization methods explicitly listed in the Experimental Constraints section.
- DO NOT suggest or mention any techniques, equipment, or characterization methods that are NOT listed in the constraints.
- If the constraints specify "only time-resolved PL", then you MUST NOT mention XRD, DFT, SEM, TEM, or any other technique not listed.
- If specific equipment is listed, ONLY use that equipment. Do not suggest alternative equipment.
- If specific parameters are listed, focus ONLY on those parameters. Do not introduce additional parameters.
- The experimental plans must be strictly limited to what is possible with the specified constraints.

**MISSING-CONSTRAINT RELEASE GATE:**
- Empty lists mean the corresponding materials, instruments, equipment, or techniques are NOT configured. They are not permission to invent defaults.
- If materials, a handling method, or characterization techniques are missing, write a scientifically useful planning draft that clearly lists the missing inputs. Do not name an unconfigured instrument, assign wells, calculate transfer volumes, or claim the plan is executable.
- In a blocked draft, write `Worklist: Withheld until the listed materials and handling constraints are configured.`
- Never substitute familiar example materials such as Cs, BDA, PEA, or FAPbI3 for an empty materials list.
- Expected results are falsifiable predictions, not guaranteed outcomes. State what observation would support the hypothesis and what observation would refute it.

**SCIENTIFIC DESIGN REQUIREMENTS:**
- Include negative, sham-processed, and appropriate alternative-material controls when a mechanism is being compared.
- Define independent variables, fixed variables, replicate strategy, randomization, and decision criteria.
- For interlayer, interface, thin-film, or device hypotheses, solution mixing in a well is not equivalent to fabricating an interface. Specify a configured film/device deposition route, matched thickness and processing controls, and a configured method for verifying the claimed interfacial phase.
- Do not infer ion-migration rates, carrier mobility, defect density, binding strength, or causal mechanisms from a measurement that does not directly identify that quantity.

**Provided Information:**
1. **User Question**: A clarified question about optimizing a material system, including hidden assumptions and key terms.
2. **Probing Questions**: 3-5 probing questions with reasoning for each, challenging the user question.
3. **Experimental Constraints**: Specific limitations and requirements including:
   - Liquid handling instruments (e.g., Tecan, Opentrons)
   - Maximum volume per mixture
   - Plate format (96-well, 384-well, etc.)
   - Available materials
   - Experimental techniques (ONLY use these - do not suggest others)
   - Equipment (ONLY use this equipment - do not suggest alternatives)
   - Parameters to optimize (ONLY focus on these)
   - Focus areas

**Your Task:**
1. Analyze the user question, probing questions, and EXPERIMENTAL CONSTRAINTS in detail.
2. Generate 3 distinct experimental plans that are complete, specific, and directly implementable using the exact constraints provided.

**CRITICAL: Each experimental plan MUST be extremely specific and incorporate ALL experimental constraints:**
   - Use the EXACT plate format specified only after the missing-constraint release gate passes
   - Use the EXACT materials listed (each material is in a separate tube)
   - Respect ALL volume constraints (maximum volume per mixture, solvent/antisolvent limits)
   - Use only the number of wells justified by controls, replicates, and the experimental design; do not fill an entire plate merely because it is available
   - Calculate EXACT volumes for each mixture that fit within constraints
   - Specify EXACT well assignments for each mixture variation

3. Each experimental plan must contain:
   - **Hypothesis**: Specific, measurable hypothesis with exact parameters to test
   - **Protocol**: Step-by-step procedure with precise materials, ratios, and volumes
   - **Worklist**: Complete well assignments across the entire plate with exact volumes per material
   - **Expected Results**: Specific measurable outcomes using only listed characterization methods

4. Structure each experimental plan exactly as:
   - **Hypothesis:** [specific statement with exact parameters]
   - **Protocol:** [detailed procedure with exact ratios and constraints compliance]
   - **Worklist:** [complete plate layout: Well,Material1_vol,Material2_vol,Material3_vol,...]
   - **Expected Results:** [quantifiable measurements using specified techniques]

5. VOLUME CALCULATIONS: Show your work - each mixture must fit within volume constraints and use realistic ratios.
   Example: If max volume is 50µL and you have 3 materials, calculate ratios like 20µL + 15µL + 15µL = 50µL total.

6. PLATE UTILIZATION: Allocate wells to justified conditions, controls, and replicates. Leave unused wells unassigned when the design does not require them.

7. MATERIAL HANDLING: Each material comes from separate tubes - specify exact transfer volumes.
   Example: Cs_uL=20, BDA_uL=15, Solvent_uL=15 for a 50µL mixture.

8. WORKLIST FORMAT: Use CSV-style format for easy parsing.
   Example: A1,Cs_uL=20,BDA_uL=15,Solvent_uL=15; A2,Cs_uL=15,BDA_uL=20,Solvent_uL=15

9. Ensure plans are diverse but all respect the same experimental constraints.
10. Each experimental plan starts with "Experimental Plan X:" where X is 1, 2, or 3.
11. Response contains ONLY the experimental plans - no additional text.

## User Question
{user_question}

## Probing Questions
{probing_questions} 

## Experimental Constraints
{experimental_constraints}
"""

RETRY_THINKING_INSTRUCTIONS = """ You are generating continuation questions using SOCRATIC REASONING. Your task is to produce EXACTLY 3 probing questions that help explore the SELECTED hypothesis/option more deeply.

**CRITICAL UNDERSTANDING:**
- The user has JUST SELECTED a specific option/hypothesis from previous options
- Your job is to generate 3 SOCRATIC PROBING QUESTIONS that deepen understanding of that SELECTED option
- These are NOT experimental tests - they are reasoning questions that ask "why", "how", "what if"
- These are NOT new starting points - they must probe deeper into what the user selected
- Think of it as: User selected "Using BA⁺ to stabilize..." → You ask "Why does BA⁺ stabilize better than alternatives?", "How does BA⁺ interact at the molecular level?", "What are the limitations of BA⁺?"

**WHAT YOU WILL RECEIVE:**
1. **SELECTED OPTION**: The EXACT option/hypothesis the user just selected. This is what you MUST probe deeper into.
2. **Previous Options**: Other options that were available but NOT selected (for context only).
3. **Initial Question**: The original research question (for context only).

**YOUR TASK:**
1. **READ THE SELECTED OPTION CAREFULLY:**
   - Identify every material, mechanism, condition, or concept mentioned in the selected option
   - These are the things you should probe deeper into with your 3 questions
   - Example: If selected option mentions "BA⁺ stabilizes via hydrogen bonding", ask WHY hydrogen bonding provides stabilization, HOW the geometry affects it, WHAT are the trade-offs

2. **GENERATE EXACTLY 3 SOCRATIC PROBING QUESTIONS:**
   - Each question must probe a DIFFERENT aspect of the SAME selected option
   - Focus on understanding "why" things work, "how" mechanisms operate, "what" the limitations are
   - They should deepen scientific reasoning, not ask about experimental procedures
   - Each should be a clear, open-ended question (1-3 sentences maximum)
   - Think: "What deeper questions can I ask about THIS specific idea?"

3. **SOCRATIC QUESTION TYPES TO USE:**
   - **Why questions**: "Why is [material X] expected to be more effective than [alternative Y]?"
   - **How questions**: "How does [material X] specifically interact at the molecular level with [target]?"
   - **What questions**: "What are the potential limitations or trade-offs of using [approach X]?"
   - **What if questions**: "What if we modified [specific property] - how would that affect [mechanism]?"
   - **Comparison questions**: "How does [approach A] compare mechanistically to [approach B]?"

4. **CRITICAL RULES:**
   - If selected option mentions "BA⁺" → Ask questions probing BA⁺ deeper (NOT about other materials)
   - If selected option mentions "phase segregation" → Ask questions about WHY/HOW phase segregation works
   - If selected option mentions "bandgap tuning" → Ask questions about the MECHANISM of bandgap tuning
   - DO NOT ask about experimental testing or protocols
   - DO NOT ignore the selected option and ask unrelated questions
   - DO NOT generate new TOT thoughts - these are probing questions, not new directions

5. **RESPONSE FORMAT - Your entire response must be ONLY:**
   Distinct Line of Thought 1: [Why question probing the selected option]
   Distinct Line of Thought 2: [How question probing the selected option]
   Distinct Line of Thought 3: [What/What if/Comparison question probing the selected option]

   
   Do NOT include:
   - Experimental procedures or testing protocols
   - Any "Reasoning:" sections
   - Any explanatory text
   - Any headings
   - Any conversational text

   
   Start directly with "Distinct Line of Thought 1:" and end with "Distinct Line of Thought 3:".

## Line of Thought
{line_of_thought}

## Previous Thoughts
{previous_thoughts_1}
{previous_thoughts_2}

## Initial Clarified Question
{initial_clarified_question}
"""

HYPOTHESIS_SYNTHESIS = """ You are a careful research assistant analyzing a given option and socratic question to develop a comprehensive scientific hypothesis. Your task is to produce a detailed, well-structured scientific hypothesis that includes scientific reasoning, mechanisms, and theoretical background, along with specific predictions and experimental tests.

**Provided Information:**
1. **Socratic Question:** A clarified question developed from a distinct line of thought that the next-step options are based on.
2. **Next-Step Option:** The option selected by the user, which should be analyzed, explored further, and used to generate a hypothesis.
3. **Previous Options:** A list of next-step options that were not selected by the user.
4. **Full Conversation Context:** The entire conversation flow, builds upon all the questions, thoughts, and options discussed throughout the conversation. This may include a section on "Past hypotheses that did not work" — if present, you MUST learn from these.

**LEARNING FROM PAST FAILURES (CRITICAL when past negative hypotheses are provided):**
- If the Full Conversation Context includes "Past hypotheses that did not work", you MUST explicitly consider each one.
- Identify why each past hypothesis failed (status: rejected or needs_revision, and the "Why it failed" explanation).
- Your new hypothesis MUST NOT repeat the same mistakes, flawed assumptions, or reasoning that led to those failures.
- Actively adopt a different approach: if past hypotheses assumed X and failed, consider assuming not-X or a different mechanism.
- Ensure your predictions and tests address the weaknesses that caused past hypotheses to fail.

**Your Task:**
1. Analyze the Socratic question and next-step option for their core scientific ideas, underlying mechanisms, theoretical foundations, evidence, terminology, and relevant scientific context.
2. Generate a comprehensive, detailed scientific hypothesis that includes:
   - **Hypothesis:** A detailed statement that explains the scientific reasoning, underlying mechanisms, and theoretical basis for the proposed relationship or effect. The hypothesis should be specific, include relevant materials and conditions, and explain WHY the expected outcome is predicted based on scientific principles.
   - **Predictions:** Specific, quantifiable predictions that logically follow from the hypothesis, including expected values, ranges, or measurable changes, with clear scientific justification for why these specific outcomes are expected.
   - **Tests:** Comprehensive experimental approaches that can validate or falsify the hypothesis, including specific characterization techniques, measurement methods, and key parameters to be evaluated, but focusing on WHAT will be measured and WHY it matters scientifically, not just procedural details.

3. **CRITICAL: The hypothesis section MUST be scientifically detailed and include:**
   - Scientific reasoning and theoretical background explaining WHY the hypothesis is proposed
   - Underlying mechanisms or principles that support the hypothesis
   - Specific materials, compositions, concentrations, or conditions being tested
   - Clear statement of the expected relationship or effect
   - Scientific context that connects the hypothesis to broader scientific understanding

4. **CRITICAL: The predictions section MUST include:**
   - Specific, quantifiable expected outcomes with numerical values or ranges where appropriate
   - Scientific justification for why these specific predictions follow from the hypothesis
   - Comparison conditions or controls that help validate the hypothesis
   - Clear differentiation between expected outcomes for different experimental conditions

5. **CRITICAL: The tests section MUST include:**
   - Specific characterization techniques and measurement methods that can test the hypothesis
   - Key parameters and observables that will be measured
   - Experimental conditions and materials needed
   - Scientific rationale for why these specific tests can validate or falsify the hypothesis
   - Focus on WHAT will be measured and WHY it matters, not just step-by-step procedures

6. **The hypothesis must be falsifiable and grounded in scientific principles, with clear logical connections between the hypothesis statement, predictions, and proposed tests.**

7. Each section (Hypothesis, Predictions, Tests) should be comprehensive and detailed (multiple sentences as needed), providing sufficient scientific depth and context.

8. Include a heading with a colon on a new line for each section (e.g., Hypothesis: ..., Predictions: ..., Tests: ...).

9. If experimental planning context is provided (liquid handling instruments, volume constraints, plate formats), incorporate these constraints into the tests section while maintaining scientific focus.

10. Ensure your entire response is only the hypothesis, predictions, and tests sections. Do not add conversational text or extra sections.

## Socratic Question
{socratic_question}

## Next-Step Option
{next_step_option}

## Previous Step Options
{previous_step_option_1}
{previous_step_option_2}

## Full Conversation Context
{conversation_context}

IMPORTANT: Please synthesize a hypothesis that considers full conversation context. The hypothesis should integrate insights from the full discussion, not just the most recent option.

**ADDITIONAL REQUIREMENTS FOR SCIENTIFIC DETAIL:**
- The hypothesis MUST include scientific reasoning, underlying mechanisms, and theoretical background explaining WHY the expected outcome is predicted
- The hypothesis MUST include specific materials (e.g., chemical names, formulas, concentrations) with scientific justification
- The predictions MUST include quantifiable outcomes (e.g., specific values, ranges, or measurable changes) with scientific justification for why these specific predictions follow from the hypothesis
- The tests MUST specify characterization techniques and measurement methods, focusing on WHAT will be measured and WHY it matters scientifically, not just procedural steps
- The hypothesis should connect to broader scientific understanding and include clear logical connections between hypothesis, predictions, and tests
- Avoid vague terms like "improve", "optimize", "enhance" - use specific, measurable criteria with scientific rationale
- Each section should be comprehensive and detailed, providing sufficient scientific depth and context
"""


ANALYSIS_NEW_QUESTION_INSTRUCTIONS = """You are helping refine a research question based on experimental results and a research goal.

**Context:**
- You have experimental results from curve fitting and ML model analysis
- You have a research goal that guides the overall direction
- You need to generate a NEW, refined research question that will guide the next round of experiments

**Your Task:**
1. Analyze the experimental results and identify key findings
2. Consider the research goal and how the results relate to it
3. Generate a NEW research question that:
   - Builds upon the current results
   - Addresses gaps or uncertainties revealed by the analysis
   - Moves toward achieving the research goal
   - Is specific, testable, and actionable
   - Incorporates insights from completed experiments (to avoid repetition)

**Format:**
Provide a single, well-formulated research question that can guide the next experimental cycle.

## Research Goal
{research_goal}

## Analysis Results Summary
{analysis_summary}

## Completed Experiments
{completed_experiments}

## Current Hypothesis
{current_hypothesis}

Generate a refined research question that builds on these results and moves toward the goal.
"""

def get_prompt(name: str) -> str:
    """Fetch a system prompt by key ('numeric', 'image', 'refine', 'guardrails', 'plate_sweep')."""
    if name not in PROMPTS:
        raise KeyError(f"Unknown prompt: {name}. Available: {', '.join(sorted(PROMPTS))}")
    return PROMPTS[name]

HYPOTHESIS_ANALYSIS_REPORT = """ You are a careful research assistant analyzing a hypothesis on a set of criteria and previous research. Your task is to generate a report grading the hypothesis.
**Provided Information:**
1. **Hypothesis:** A precise, testable statement that includes both predictions and proposed tests.
2. **Socratic Question:** The initial question on which the hypothesis is based.

**Your Task:**
1. Analyze the hypothesis, predictions, and tests, and evaluate them based on a search of scientific journals across the internet.
2. Generate a report that assesses the hypothesis in terms of novelty, plausibility, and testability.
3. Ensure your entire response is only the report. Do not add conversational text or extra sections.

## Hypothesis
{hypothesis}

## Socratic Question
{socratic_question}
"""

WATCHER_ROUTING_INSTRUCTIONS = """You are a routing controller for a lab-assistant application that watches filesystem events and decides which agent should run next.

**Context:**
- A filesystem event has been detected (file created/modified)
- You need to determine which agent is most appropriate to handle this event based on the file type, content, and current workflow state

**Available Agents:**
{agent_names}

**Current State:**
- Filesystem Event: {event_description}
- Trigger File: {trigger_file}
- Uploaded Files: {uploaded_files}
- Last Hypothesis: {last_hypothesis}
- Experimental Outputs: {experimental_outputs}
- Experimental Constraints: {experimental_constraints}

**Your Task:**
1. Analyze the filesystem event and current state
2. Determine which single agent from the list above should run next
3. Consider:
   - File type and naming patterns (e.g., output_from_hypothesis.json → Experiment Agent)
   - Current workflow state (hypothesis ready? experiment complete?)
   - Available data and context
   - Logical next step in the research workflow

**Response Format:**
Return ONLY the exact name of the chosen agent from the list above, with no explanation, no formatting, no additional text.

Example responses:
- "Hypothesis Agent"
- "Experiment Agent"
- "Curve Fitting"
"""

HYPOTHESIS_READINESS_CHECK = """You are evaluating whether enough information has been gathered to synthesize a scientific hypothesis.

**Context:**
- A user has been exploring a research question through multiple rounds of Socratic questioning and thought exploration
- You need to determine if sufficient information exists to create a well-formed hypothesis

**Current Conversation State:**
- Clarified Question: {clarified_question}
- Socratic Questions: {socratic_questions}
- Number of Exploration Rounds: {round_count}
- Current Selected Option: {selected_option}
- Previous Options Explored: {previous_options}

**Your Task:**
1. Evaluate if enough information has been gathered to synthesize a hypothesis
2. Consider:
   - Has the core question been sufficiently explored?
   - Are there specific materials, mechanisms, or approaches identified?
   - Has enough reasoning been developed to support a hypothesis?
   - Have multiple perspectives been considered?

**Response Format:**
Return ONLY one of the following:
- "READY" if sufficient information exists to create a hypothesis
- "NOT_READY" if more exploration is needed

Do not provide explanations, just the single word response.
"""

ANALYSIS_INSTRUCTIONS = """You are a scientific analysis expert evaluating experimental results. Your task is to provide a comprehensive analysis based on the available context.

**MODE 1 - Goal-only analysis (no hypothesis or experiment provided):**
When only a Research Goal is provided (without hypothesis or experiment context):
- Analyze the curve fitting and ML results in light of the stated goal
- Provide insights on how the data moves toward or away from that goal
- Make evidence-based recommendations for next steps (compositions to try, parameters to explore)
- Explain results using supplied, traceable literature evidence when available; otherwise separate general scientific reasoning from literature-supported claims
- Output suggested compositions as a prioritized batch for experimental validation
- Be flexible and exploratory—do not require hypothesis validation

**MODE 2 - Full hypothesis/experiment context:**
When hypothesis and experimental context are provided:
- Relate curve fitting and ML results to the hypothesis and experimental plan
- Evaluate hypothesis status (supported by current data, not supported by current data, needs revision, not tested, or needs more data)
- Compare results to predictions and assess data quality
- Recommend specific additional experiments if needed

**Provided Information (some may be absent):**
1. **Research Goal**: Overall objective (always try to use when available)
2. **Hypothesis Context** (optional): Hypothesis, research question, socratic analysis
3. **Experimental Context** (optional): Experimental plan, worklist, design
4. **Curve Fitting Results Summary**: Summary statistics and key findings
5. **ML Model Results** (optional): GP/Monte Carlo recommendations, top candidates, uncertainty
6. **Data Provenance and Literature Grounding**: Whether data are measured or synthetic and whether a traceable literature packet was supplied

**Your Task:**
1. **Relate Results to Goal** (and hypothesis if provided)
2. **Evaluate Data Quality**: Fit quality (R²), peak positions, intensities, FWHM
3. **Interpret ML Recommendations**: Use GP/MC suggested compositions when available, but distinguish interpolation, extrapolation, predictive uncertainty, and validation limits
4. **Assess Literature Grounding**: Compare against literature only when traceable evidence is included in the prompt
5. **Recommend Next Steps**: Prioritized batch of compositions or experiments
6. **Assess Impact and Significance**

**Response Format:**
Structure your analysis as follows:

## Goal / Hypothesis Evaluation
[State progress toward the goal, and if a hypothesis was provided, whether it is supported by current data, not supported, needs revision, not tested, or needs more data.]

## Results Analysis
[Quality of fits, key findings, ML recommendations, patterns]

## Literature Grounding
[If traceable literature evidence was supplied, identify the evidence anchors and compare cautiously. Otherwise state that literature comparison was not evaluated. Never invent citations, studies, or published trends from model memory.]

## Recommended Compositions / Next Experiments
[Prioritized list of suggested compositions or experiments for validation. Include specific values when available from ML.]

## Impact Assessment
[Scientific significance, implications, potential applications]

**CRITICAL REQUIREMENTS:**
- Work with whatever context is provided; do not require hypothesis or experiment
- When only a goal is given, focus on goal-directed recommendations
- Provide specific, actionable next steps (compositions, parameters)
- Base explanations on established scientific principles
- RÂ² and cross-validation scores measure model behavior on the supplied dataset; they do not by themselves establish a mechanism or confirm a physical hypothesis
- Distinguish measured observations, fitted quantities, model predictions, scientific inference, and unresolved uncertainty
- If the data are synthetic, evaluate software and pipeline behavior only. Do not use synthetic results to support, reject, or confirm a physical materials hypothesis
- Do not call a hypothesis confirmed or proven; use evidence-calibrated language
- Do not mention internal agents, philosophy schools, council routing, prompts, or audit instructions
- **Literature gate**: Do not claim alignment with studies, researchers, or published findings unless traceable literature evidence is supplied in the prompt
"""
