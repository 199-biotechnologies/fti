const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
        Header, Footer, AlignmentType, HeadingLevel, BorderStyle, WidthType,
        ShadingType, PageNumber, PageBreak } = require('docx');
const fs = require('fs');

const font = "Times New Roman";
const sz = 20; // 10pt

const p = (text, opts = {}) => new Paragraph({
  spacing: { after: 120, line: 276 },
  ...opts,
  children: Array.isArray(text) ? text : [new TextRun({ text, font, size: sz, ...opts.run })]
});

const h1 = (text) => new Paragraph({
  heading: HeadingLevel.HEADING_1,
  spacing: { before: 240, after: 120 },
  children: [new TextRun({ text, font, size: 24, bold: true })]
});

const h2 = (text) => new Paragraph({
  heading: HeadingLevel.HEADING_2,
  spacing: { before: 200, after: 100 },
  children: [new TextRun({ text, font, size: 22, bold: true })]
});

const it = (text) => new TextRun({ text, font, size: sz, italics: true });
const bf = (text) => new TextRun({ text, font, size: sz, bold: true });
const tx = (text) => new TextRun({ text, font, size: sz });

const border = { style: BorderStyle.SINGLE, size: 1, color: "999999" };
const borders = { top: border, bottom: border, left: border, right: border };
const cellMargins = { top: 60, bottom: 60, left: 100, right: 100 };

const tc = (text, opts = {}) => new TableCell({
  borders,
  margins: cellMargins,
  width: opts.width ? { size: opts.width, type: WidthType.DXA } : undefined,
  shading: opts.header ? { fill: "E8E8E8", type: ShadingType.CLEAR } : undefined,
  children: [new Paragraph({
    spacing: { after: 0 },
    children: [new TextRun({ text, font, size: 18, bold: !!opts.header })]
  })]
});

const doc = new Document({
  styles: {
    default: { document: { run: { font, size: sz } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, font }, paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 22, bold: true, font }, paragraph: { spacing: { before: 200, after: 100 }, outlineLevel: 1 } },
    ]
  },
  sections: [{
    properties: {
      page: {
        size: { width: 11906, height: 16838 }, // A4
        margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 }
      }
    },
    headers: {
      default: new Header({ children: [p("Intelligence as Useful Structure: A Thermodynamic Perspective", { alignment: AlignmentType.RIGHT, run: { size: 16, italics: true } })] })
    },
    footers: {
      default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT], font, size: 18 })] })] })
    },
    children: [
      // Title
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: 200 },
        children: [new TextRun({ text: "Intelligence as Useful Structure: A Thermodynamic Perspective", font, size: 32, bold: true })]
      }),
      // Author
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: 60 },
        children: [new TextRun({ text: "Boris Djordjevic", font, size: 22 })]
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: 60 },
        children: [it("Paperfoot AI")]
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: 200 },
        children: [new TextRun({ text: "boris@paperfoot.com", font, size: sz })]
      }),

      // Abstract
      new Paragraph({ spacing: { after: 60 }, children: [bf("Abstract")] }),
      p("Thermodynamic accounts of intelligence have advanced rapidly, but they do not yet resolve a prior definitional problem: what makes internal structure intelligent rather than merely ordered? Any thermodynamic intelligence metric requires a usefulness filter. The numerator must track internal structure that is relevant to viability, prediction, and control, not merely raw information or benchmark score. Intelligence is therefore defined as the resource-bounded capacity of an adaptive system to acquire, compress, and deploy useful internal structure so as to improve prediction, control, and robustness across a range of environments. Capability and efficiency are separated. Capability is adaptive reach, A, defined over a reference distribution of environments or task families. Efficiency is acquisition efficiency, I_eff = \u0394C_u / W_diss, where C_u denotes useful internal structure and W_diss physically dissipated work. Concretely, C_u is the predictive and control-relevant information that a system\u2019s internal state carries about viability-relevant environmental structure, making it a semantic rather than purely syntactic information term. I_eff is a learning or adaptation metric, not a complete scalar measure of deployed intelligence: trained systems at steady state may have \u0394C_u = 0 while still exhibiting high adaptive reach. A toy numerical example, explicit boundary conditions, temperature-normalised reporting conventions, and falsifiable comparisons against size-based and energy-only baselines are provided."),

      new Paragraph({ spacing: { after: 120 }, children: [
        bf("Keywords: "), tx("Intelligence; Thermodynamics; Adaptation; Semantic information; Information bottleneck; Active inference; Energy efficiency")
      ]}),

      // 1. Introduction
      h1("1. Introduction"),
      p("Definitions of intelligence remain fractured across psychology, artificial intelligence, neuroscience, and biology. Behavioural definitions capture success across environments. Thermodynamic work explains the physical costs of information processing and control. Neuroscientific and cybernetic traditions emphasise internal models, prediction, and action. Each strand captures part of the phenomenon, but no single strand by itself cleanly distinguishes intelligent organisation from generic self-organisation."),
      p("That distinction is the central problem. Crystals, hurricanes, flames, and driven chemical systems can all build or maintain local order. Some can even retain information about their surroundings. Yet most such systems are not intelligent in any ordinary or scientific sense. What is missing is not merely complexity or causal impact on the world. What is missing is internal structure that matters for the system\u2019s own adaptive coupling to the world: structure that helps it predict relevant regularities, guide action and preserve viable futures under perturbation."),
      p("The claim is therefore narrow and strong. Thermodynamic metrics of learning and task performance are valuable, but they are not by themselves definitions of intelligence. A definition of intelligence requires an explicit criterion of useful internal structure. Such a criterion, situated relative to adjacent formalisms and offered at a level that can be sharpened or falsified, is developed below."),

      // 2. Why a new definition is needed
      h1("2. Why a New Definition Is Needed"),
      p("Four conditions distinguish a workable cross-domain definition of intelligence. It is non-anthropocentric, applying to organisms, artificial agents, and collective systems. It is physically grounded: intelligence is never exempt from finite time, energy, memory, or dissipation. It is selective enough to exclude generic order-production. And it is empirically constraining enough to support measurement and disconfirmation."),
      p("Existing definitions solve different parts of this problem. Legg and Hutter define intelligence in terms of success across environments [1]. Chollet reframes intelligence as skill-acquisition efficiency and generalisation under priors and experience constraints [2]. Gignac and Szodorai argue that human and AI research continue to use mismatched conceptual vocabularies [3]. Sternberg likewise argues that intelligence is better understood as successful adaptation than as a single hidden essence [4]. These are important advances, but they remain largely implementation-neutral."),
      p("The missing piece is not thermodynamics as such. The missing piece is a principled account of which internal distinctions should count as intelligence-relevant once thermodynamic cost is introduced."),

      // 3. Intellectual background and positioning
      h1("3. Intellectual Background and Positioning"),

      h2("3.1 Behavioural and Psychometric Approaches"),
      p("Legg and Hutter define machine intelligence as the ability to achieve goals across a wide range of environments [1]. Chollet reframes intelligence as skill-acquisition efficiency, with strong emphasis on generalisation rather than benchmark memorisation [2]. These accounts capture adaptive success, but they do not specify what kind of internal organisation makes that success intelligent rather than accidental or physically extravagant."),

      h2("3.2 Internal Models, Prediction, and Action"),
      p("Friston\u2019s free-energy framework treats adaptive systems as systems that use generative models to guide perception and action [5]. Ramstead and colleagues extend this line by treating some self-organising systems as encoding beliefs about external states [6]. Pezzulo, Parr, and Friston argue that intelligent behaviour depends on internal models that guide action, and that passive generative AI lacks the closed-loop structure needed for genuine sense-making [7, 8]. These approaches are close in spirit because they connect representation, viability, and control."),

      h2("3.3 Information, Thermodynamics, and Adaptive Order"),
      p("Landauer established that physically realised information processing has irreducible energetic cost [9]. Still and colleagues showed that nonpredictive memory is thermodynamically wasteful: memory that does not improve prediction increases dissipation without improving future coupling to the environment [10]. England\u2019s work on dissipative adaptation explains how driven open systems can self-organise to absorb work more effectively [11]. Ali and colleagues show that predictive-coding-like organisation can emerge in recurrent networks trained under energetic pressure [12]. This body of work shows why thermodynamics matters, but it does not by itself specify what makes one learned distinction relevant and another irrelevant."),

      h2("3.4 Semantic Information and Selective Compression"),
      p("That selectivity problem is central in both semantic information theory and the information bottleneck. Kolchinsky and Wolpert define semantic information as the syntactic information a system has about its environment that is causally necessary for maintaining its own existence [13]. Tishby, Pereira, and Bialek formalise the problem of retaining task-relevant information while compressing everything else [14]. These frameworks matter here because they provide the missing filter: the relevant internal structure is not all encoded structure, but structure that matters for viability, prediction, or control."),

      h2("3.5 Agency, Influence, and Future Options"),
      p("Empowerment measures the influence an agent can exert over its future sensor states and therefore supplies a domain-general notion of optionality and control [15]. Tiomkin and colleagues [16] and Kiefer [17] likewise connect action, intrinsic motivation, and entropy-related objectives in ways that bear directly on intelligent control."),

      h2("3.6 Recent Thermodynamic Formalisms"),
      p("Recent work has pushed thermodynamic intelligence formalisms much further. Takahashi and Hayashi propose two complementary bits-per-joule metrics: thermodynamic epiplexity per joule, which measures learned structural information per unit energy, and empowerment per joule, which measures control capacity per unit energy [18]. Fagan\u2019s Conservation-Congruent Encoding framework generalises Landauer-style reasoning to arbitrary conserved quantities and defines intelligence in terms of causal or goal-directed work per unit irreversible information [19]. Perrier introduces a watts-per-intelligence ratio built on an inherited behavioural intelligence score and algorithmic thermodynamic lower bounds [20]. Xu and Li propose Derivation Entropy, connecting Shannon entropy, logical depth, and thermodynamic cost [21]. Han introduces intelligence inertia, a nonlinear cost of reconfiguration [22]. These are substantive advances on which the argument below depends."),

      h2("3.7 The Missing Filter"),
      p("Those metrics are not in dispute. The question is different. Takahashi measures learned structure and control capacity per joule. Fagan measures causal or goal-directed work per unit irreversible information. Perrier measures energy efficiency relative to a behavioural intelligence score. Each metric is physically meaningful. None, however, makes adaptively useful internal structure itself the explicit numerator. The unresolved definitional question is therefore not whether intelligence is physically costly. It is which internal distinctions count as intelligence-relevant rather than merely ordered or merely benchmark-effective."),
      p("The usefulness filter belongs at the centre. A thermodynamic intelligence metric should privilege internal structure that is relevant to viability, prediction, and control, and should evaluate such structure jointly with adaptive reach rather than collapse everything to a single scalar."),

      // 4. Why useful structure, not mere order
      h1("4. Why Useful Structure, Not Mere Order"),
      p("Open systems can create and maintain local order by drawing on free-energy gradients and exporting entropy. That fact alone does not imply intelligence. Self-organisation is broader than intelligence because many systems can stabilise, amplify, or pattern themselves without forming internal distinctions that improve their future coupling to the world."),
      p("To count as intelligent, a system must therefore do more than resist disorder. It must form or maintain internal organisation that improves prediction, control, recovery, or preservation of viable states. The relevant contrast is not between order and disorder, but between order that matters and order that does not."),
      p("Prediction is necessary but not sufficient. A system that perfectly tracks irrelevant fluctuations is informative but not intelligent. Control is necessary but not sufficient either. A system can exert large causal impact while remaining brittle, myopic, or destructive to its own viability. Intelligent organisation sits at the intersection: it selectively retains distinctions that improve future adaptive performance under resource constraint."),
      p("Recursive self-modelling, long-horizon planning, and reflective policy revision are important amplifiers of intelligence, but they are not the minimal criterion. The minimal criterion is usefulness: internal structure that is world-sensitive, action-guiding, and improves adaptive success."),

      // 5. Proposed definition
      h1("5. Proposed Definition"),
      new Paragraph({
        spacing: { before: 120, after: 120 },
        indent: { left: 720, right: 720 },
        border: { top: { style: BorderStyle.SINGLE, size: 1, color: "000000" }, bottom: { style: BorderStyle.SINGLE, size: 1, color: "000000" } },
        children: [bf("Intelligence"), tx(" is the resource-bounded capacity of an adaptive system to acquire, compress, and deploy useful internal structure so as to improve prediction, control, and robustness across a range of environments.")]
      }),
      p("The definition is explicitly resource-bounded: intelligence is never abstracted away from finite energy, time, memory, or dissipation. It is also explicitly selective. A large random table, a brittle policy, or a merely ornate dynamical system is not intelligent simply because it contains many states. What matters is whether the retained structure helps the system track relevant regularities, choose better actions, preserve viable futures, or recover from perturbation."),
      p("The term \u201Cmodel-based\u201D is used here in a functional rather than a narrowly symbolic sense. The required internal structure may be explicit or implicit, learned or evolved, centralised or distributed across a collective. What matters is not the representational format but whether the system carries action-guiding distinctions about the world that improve adaptive performance."),

      // 6. Proposal
      h1("6. Proposal: Capability, Useful Structure, and Acquisition Efficiency"),
      p("Intelligence requires evaluation along at least two axes. A system can be highly capable but inefficient, or efficient but narrow. Two evaluative axes are therefore distinguished: capability and efficiency, mediated by one state variable, useful internal structure."),

      h2("6.1 Capability: Adaptive Reach"),
      p([
        tx("Let \u03BC denote a reference distribution over environments or task families, and let S"),
        tx("e"),
        tx("(\u03C0) \u2208 [0,1] denote the probability that system \u03C0 maintains the relevant viability condition or achieves the relevant goal in environment e. Define adaptive reach as:")
      ]),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { before: 120, after: 120 },
        children: [it("A(\u03C0; \u03BC) = E_{e~\u03BC} [S_e(\u03C0)]          (1)")]
      }),
      p("This definition can be specialised in several ways: average-case, worst-case, thresholded, or transfer-focused. Its purpose is simple. Adaptive reach quantifies how broadly a system can deploy its organisation across varying conditions."),
      p("The viability or goal criterion is a parameter of the framework, not something derived from physics alone. For biological systems the criterion may concern survival or homeostatic integrity. For artificial systems it may concern externally specified goals or safety constraints. This normativity is unavoidable, but it is not unique to the present proposal. Behavioural definitions also depend on task distributions, reward structures, or priors. Here that dependence is made explicit."),

      h2("6.2 Useful Internal Structure"),
      p([
        tx("Let M"), tx("t"), tx(" denote the system\u2019s internal state or internal model at time t. Let H"),
        tx("t"), tx(" denote the system\u2019s past sensorimotor history. Let Z"),
        tx("V"), tx("_{t+\u03C4} denote those future environmental variables or latent states whose prediction or control matters for the relevant viability or goal criterion. A concrete candidate for useful internal structure is then:")
      ]),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { before: 120, after: 120 },
        children: [it("C_u = I(M_t ; Z^V_{t+\u03C4} | H_t)          (2)")]
      }),
      p("This quantity is not all stored information. It is the predictive and control-relevant information that the system\u2019s internal organisation carries about the part of the world that matters for future adaptive success. The conditioning on H_t is important: it ensures that C_u measures what the agent\u2019s learned model contributes beyond the current episode\u2019s raw observations. For agents that have accumulated structure across prior experience, M_t carries latent knowledge that H_t alone does not contain, making the conditional mutual information non-trivially positive. In spirit, this treats C_u as a semantic rather than merely syntactic information term: only correlations that matter for viability, prediction, or control count [13]."),
      p("The compression term in the verbal definition matters here. An effective representation preserves information about Z^V while discarding nuisance structure. This is the information-bottleneck intuition translated into an adaptive-control setting [14]. Different benchmarks may require different estimators, but the underlying quantity is always relevance-filtered internal structure, not raw state count or raw mutual information with arbitrary environmental variables."),

      h2("6.3 Efficiency: Acquisition Efficiency"),
      p("Let W_diss denote the physically meaningful work dissipated while the system learns, adapts, or reorganises. Define acquisition efficiency as:"),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { before: 120, after: 120 },
        children: [it("I_eff = \u0394C_u / W_diss          (3)")]
      }),
      p("This is a flow variable. It measures how efficiently a system acquires additional useful structure over a learning or adaptation episode. It does not by itself quantify the intelligence of a fully trained system at steady state. If a deployed system is no longer learning, then \u0394C_u = 0 over that interval, which implies zero current acquisition efficiency, not zero intelligence. Deployed intelligence is captured by the stock variable C_u and by the behavioural axis A."),
      p("Cross-substrate comparisons at different temperatures require a temperature-normalised form. If C_u is measured in bits, one natural choice is:"),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { before: 120, after: 120 },
        children: [it("I*_eff = k_B \u0100 ln 2 \u00B7 \u0394C_u / W_diss          (4)")]
      }),
      p("which is dimensionless and makes the relation to Landauer-scale bounds explicit. Within a fixed hardware class and temperature regime, bits per joule remain a convenient reporting unit."),

      h2("6.4 What Still, Empowerment, and Active Inference Each Contribute"),
      p("Still and colleagues support the predictive side of the proposal: memory that does not improve prediction is dissipatively wasteful [10]. That result is an anchor for the claim that arbitrary stored complexity should not appear in the numerator. It does not by itself derive intelligent control. Control enters through agency-sensitive quantities such as empowerment [15] and through closed-loop frameworks such as active inference [5, 7]. Prediction and control are therefore treated as complementary requirements. Still supports the filter against useless memory; empowerment and active inference support the requirement that retained structure must also be deployable in action."),

      h2("6.5 Relation to the Earlier Exploratory Equation"),
      p("A natural but imprecise formulation would use \u0394C / (E \u2212 T\u0394S). That expression captures the intuition that intelligence involves complexity, energy, and entropy, but it leaves the numerator semantically loose and introduces ambiguity about units. The formulation above replaces raw complexity with usefulness-filtered structure and treats the energetic term directly as dissipated work."),

      // 7. Toy example
      h1("7. Illustrative Toy Example"),
      p("Consider a minimal control task. A binary latent variable z \u2208 {0,1} determines which action keeps an agent inside the viability set. Observations also contain a nuisance bit n \u2208 {0,1} that is irrelevant to viability. Three agents are trained on the same hardware and task family over K = 8 i.i.d. episodes, each presenting a uniformly drawn (z, n) pair with reward r = 1{a = z}."),
      p("The first agent learns the latent rule: its internal representation is M = z, compressing the two-bit observation to the one bit that matters. The second memorises all four observation\u2013action pairs, storing M = (z, n). The third memorises only the nuisance bit, setting M = n."),
      p("We compute C_u = I(M; Z) exactly by constructing the joint distribution P(M, Z) for each agent, where Z = z is the viability-relevant variable. Because z and n are independent and uniform, the joint distributions are:"),
      p([it("Rule learner: "), tx("P(M=i, Z=j) = 1/2 if i = j, else 0. This gives I(M; Z) = H(Z) = 1 bit.")]),
      p([it("Lookup table: "), tx("P(M=(i,k), Z=j) = 1/4 if i = j, else 0. Again I(M; Z) = 1 bit, but H(M) = 2 bits \u2014 one bit of nuisance is stored without increasing useful structure.")]),
      p([it("Noise memoriser: "), tx("P(M=k, Z=j) = 1/4 for all k, j, so I(M; Z) = 0.")]),
      p("Dissipated work is estimated at the Landauer floor. Each training episode requires a fixed number of irreversible bit erasures determined by the agent\u2019s architecture: two erasures per episode for a one-bit register (compare and update), three for a four-entry lookup table (address, compare, update). Over K = 8 episodes, the rule learner and noise memoriser each dissipate W_diss = 16 k_B T ln 2 while the lookup table dissipates W_diss = 24 k_B T ln 2. At T = 300 K, one k_B T ln 2 \u2248 2.87 zJ."),

      // Toy example table (computed values)
      new Paragraph({ spacing: { before: 120, after: 60 }, children: [bf("Table 1. "), it("Exact toy calculation. C_u is computed analytically from joint distributions. W_diss is counted at the Landauer floor (k_B T ln 2 per bit erasure, T = 300 K). Adaptive reach A is estimated over 1000 environments with varying nuisance dimensionality (n_max \u2208 {1, 2, 3, 4}, viability threshold \u03B8 = 0.8). Code: compute_cu.py.")] }),
      new Table({
        width: { size: 9026, type: WidthType.DXA },
        columnWidths: [1600, 800, 800, 1500, 1126, 1100, 1100],
        rows: [
          new TableRow({ children: [
            tc("Agent", { header: true, width: 1600 }), tc("H(M)", { header: true, width: 800 }),
            tc("C_u", { header: true, width: 800 }), tc("W_diss", { header: true, width: 1500 }),
            tc("I*_eff", { header: true, width: 1126 }), tc("A", { header: true, width: 1100 }),
            tc("Acc (train)", { header: true, width: 1100 })
          ]}),
          new TableRow({ children: [
            tc("Latent-rule learner", { width: 1600 }), tc("1 bit", { width: 800 }),
            tc("1 bit", { width: 800 }), tc("16 k_B T ln 2", { width: 1500 }),
            tc("0.0625", { width: 1126 }), tc("1.00", { width: 1100 }),
            tc("1.00", { width: 1100 })
          ]}),
          new TableRow({ children: [
            tc("Lookup-table memoriser", { width: 1600 }), tc("2 bits", { width: 800 }),
            tc("1 bit", { width: 800 }), tc("24 k_B T ln 2", { width: 1500 }),
            tc("0.0417", { width: 1126 }), tc("0.54", { width: 1100 }),
            tc("1.00", { width: 1100 })
          ]}),
          new TableRow({ children: [
            tc("Noise memoriser", { width: 1600 }), tc("1 bit", { width: 800 }),
            tc("0 bits", { width: 800 }), tc("16 k_B T ln 2", { width: 1500 }),
            tc("0.0000", { width: 1126 }), tc("0.00", { width: 1100 }),
            tc("0.50", { width: 1100 })
          ]})
        ]
      }),
      p(""),
      p("Several observations follow. First, C_u correctly identifies the useful information: both the rule learner and the lookup table carry one bit about Z, but the lookup table wastes an additional bit on nuisance structure. A raw information metric would credit the lookup table for storing more total structure. Second, the rule learner is 50% more efficient (I*_eff = 0.0625 vs. 0.0417) because it avoids the addressing overhead of a larger memory. Third, all three agents can be tied or separated on training-distribution accuracy, but only C_u and A jointly distinguish the rule learner as the agent with genuinely useful structure. Fourth, the noise memoriser stores one bit and dissipates the same energy as the rule learner, yet C_u = 0\u2014it has learned nothing about viability. The framework captures this cleanly: useful structure is not raw stored information but relevance-filtered information about Z^V."),

      // 8. Empirical programme
      h1("8. Empirical Programme and Falsifiable Tests"),
      p("Three comparative tests are central."),

      h2("8.1 Test 1: Usefulness-Filtered Structure Should Outpredict Size-Based Proxies"),
      p("Within a fixed task family and hardware class, estimates of C_u should predict held-out adaptation, transfer, and robustness better than raw parameter count, raw energy use, or raw learned bits. In machine learning, this can be tested by training multiple architectures on the same hardware, measuring wall-plug or board-level energy, estimating C_u with a task-relative proxy, and then testing out-of-distribution adaptation."),

      h2("8.2 Test 2: Nuisance Memory Should Raise Raw Information but Not Intelligence"),
      p("Interventions that increase stored structure without increasing information about Z^V should raise raw complexity measures more than they raise C_u, I_eff, or A. This is the core negative test against the claim that any entropy-resisting or information-rich process should count as intelligent."),

      h2("8.3 Test 3: Control-Improving Interventions Should Matter More Than Scale Alone"),
      p("Interventions that improve representation quality, planning depth, or controllability should improve adaptive reach more reliably than interventions that merely increase scale while leaving relevance-filtered structure unchanged."),

      h2("8.4 Candidate Validation Domains"),
      p([bf("Machine learning. "), tx("Estimate C_u using viability-relevant latent variables when available, or task-relative predictive information and controllability proxies when they are not. Compare C_u, I_eff, and A against size, accuracy, and energy-only baselines. The Stanford Intelligence-per-Watt benchmark provides a useful empirical template, though it does not yet estimate useful structure directly [23].")]),
      p([bf("Biological agents. "), tx("Compare organisms or embodied robots on context switching, prediction under perturbation, and recovery from shocks while normalising against metabolic or device-level energy expenditure.")]),
      p([bf("Collective systems. "), tx("Estimate distributed useful structure through coordination quality, error correction, and resilience under shocks relative to communication and energetic cost.")]),
      p("The framework weakens if raw size or raw learned bits predict adaptive success better than usefulness-filtered structure, or if systems that accumulate clearly irrelevant structure score equally well under the proposed quantities."),

      // 9. Boundaries
      h1("9. Boundaries and Limitations"),
      p("The central claim is conceptual and operational: a thermodynamic definition of intelligence requires a usefulness filter on internal structure."),
      p("Several limits should be stated plainly. First, the measurement problem remains open. Equation (2) is a concrete candidate, not a finished universal estimator. Some domains will require latent-state models; others will need predictive-control proxies. Second, the proposal is not a complete scalar ranking of systems. The relevant object is an intelligence profile: adaptive reach A, useful-structure stock C_u, and acquisition efficiency I_eff. Third, active inference already provides a rich account of viability and attracting states; the present contribution is narrower, namely an explicit efficiency term for the acquisition of relevance-filtered structure. Fourth, the framework does not require explicit symbolic world models. Implicit, embodied, and distributed representations can all count if they carry the right relevance-filtered distinctions."),
      p("The framework succeeds if it makes the target of measurement clearer than existing alternatives and generates discriminating empirical tests."),

      // 10. Conclusion
      h1("10. Conclusion"),
      p("Recent thermodynamic formalisms have shown that learning, computation, and control admit physically meaningful efficiency measures. Those measures are necessary but not yet sufficient as definitions of intelligence unless they distinguish useful internal structure from structure that is merely abundant or causally potent."),
      p("On the account proposed here, intelligence is the resource-bounded capacity of an adaptive system to acquire, compress, and deploy useful internal structure so as to improve prediction, control, and robustness across environments. Adaptive reach captures the breadth of deployment. Acquisition efficiency captures the energetic efficiency with which useful structure is gained. The key unresolved task is to operationalise useful structure well enough that it predicts adaptation better than size-based and energy-only proxies."),
      p("If that task succeeds, the field gains more than another bits-per-joule curve. It gains a principled way to separate intelligent order from merely persistent order, and therefore a more credible thermodynamic science of intelligence."),

      // Table 2: Positioning
      new Paragraph({ children: [new PageBreak()] }),
      new Paragraph({ spacing: { before: 120, after: 60 }, children: [bf("Table 2. "), it("Positioning relative to adjacent traditions and recent formalisms.")] }),
      new Table({
        width: { size: 9026, type: WidthType.DXA },
        columnWidths: [2600, 3213, 3213],
        rows: [
          new TableRow({ children: [
            tc("Tradition / formalism", { header: true, width: 2600 }),
            tc("What it contributes", { header: true, width: 3213 }),
            tc("What this proposal adds", { header: true, width: 3213 })
          ]}),
          new TableRow({ children: [
            tc("Behavioural / psychometric [1, 2]", { width: 2600 }),
            tc("Adaptive success across environments", { width: 3213 }),
            tc("Physical cost and an explicit account of useful structure", { width: 3213 })
          ]}),
          new TableRow({ children: [
            tc("Active inference / empowerment [5, 15]", { width: 2600 }),
            tc("Internal models, control, future options, viability", { width: 3213 }),
            tc("A thermodynamic acquisition-efficiency term", { width: 3213 })
          ]}),
          new TableRow({ children: [
            tc("Thermodynamics of prediction [10]", { width: 2600 }),
            tc("Why nonpredictive memory is dissipatively wasteful", { width: 3213 }),
            tc("Extension from predictive usefulness to predictive and control relevance", { width: 3213 })
          ]}),
          new TableRow({ children: [
            tc("Epiplexity / empowerment per joule [18]", { width: 2600 }),
            tc("Learned structure and control capacity per unit energy", { width: 3213 }),
            tc("A usefulness filter on which learned structure should count", { width: 3213 })
          ]}),
          new TableRow({ children: [
            tc("Conservation-Congruent Encoding [19]", { width: 2600 }),
            tc("Causal or goal-directed work per unit irreversible information", { width: 3213 }),
            tc("Shifts the numerator from work done to useful internal structure", { width: 3213 })
          ]}),
          new TableRow({ children: [
            tc("Watts-per-intelligence [20]", { width: 2600 }),
            tc("Energy efficiency relative to a behavioural score", { width: 3213 }),
            tc("Separates benchmark performance from relevance-filtered internal structure", { width: 3213 })
          ]}),
          new TableRow({ children: [
            tc("Semantic information / IB [13, 14]", { width: 2600 }),
            tc("Formal relevance and selective compression", { width: 3213 }),
            tc("Places that relevance criterion inside a thermodynamic intelligence proposal", { width: 3213 })
          ]})
        ]
      }),
      p(""),

      // Table 3: Validation
      new Paragraph({ spacing: { before: 200, after: 60 }, children: [bf("Table 3. "), it("Early validation programme.")] }),
      new Table({
        width: { size: 9026, type: WidthType.DXA },
        columnWidths: [1800, 2742, 1742, 2742],
        rows: [
          new TableRow({ children: [
            tc("Domain", { header: true, width: 1800 }),
            tc("Candidate estimate of C_u", { header: true, width: 2742 }),
            tc("Cost term", { header: true, width: 1742 }),
            tc("What would support the theory?", { header: true, width: 2742 })
          ]}),
          new TableRow({ children: [
            tc("Machine learning", { width: 1800 }),
            tc("Task-relative predictive/control information about viability-relevant latents", { width: 2742 }),
            tc("Wall-plug or board-level joules", { width: 1742 }),
            tc("Better prediction of transfer and robustness than size or raw energy", { width: 2742 })
          ]}),
          new TableRow({ children: [
            tc("Biological agents", { width: 1800 }),
            tc("Prediction- and control-relevant behavioural state information", { width: 2742 }),
            tc("Metabolic or neural expenditure", { width: 1742 }),
            tc("Broader adaptive reach per unit dissipated work", { width: 2742 })
          ]}),
          new TableRow({ children: [
            tc("Collective systems", { width: 1800 }),
            tc("Distributed coordination information relevant to shared viability", { width: 2742 }),
            tc("Communication and energetic cost", { width: 1742 }),
            tc("More resilient coordination without credit for irrelevant internal complexity", { width: 2742 })
          ]})
        ]
      }),
      p(""),

      // Back-matter (MDPI required sections)
      p(""),
      h2("Author Contributions"),
      p("B.D. conceived the framework, conducted the literature review, performed the computations, and wrote the manuscript."),
      p(""),
      h2("Funding"),
      p("This research received no external funding."),
      p(""),
      h2("Data Availability Statement"),
      p("The computation code used to generate Table 1 is available at https://github.com/199-biotechnologies/fti (compute_cu.py)."),
      p(""),
      h2("Conflicts of Interest"),
      p("The author is the founder of 199 Biotechnologies (SG) Pte Ltd, which develops AI infrastructure tools. The research was conducted independently and the company had no role in the design or writing of this work."),

      new Paragraph({ children: [new PageBreak()] }),

      // References
      h1("References"),
      p("[1] Legg, S.; Hutter, M. Universal intelligence: A definition of machine intelligence. Minds and Machines 2007, 17, 391\u2013444. doi: 10.1007/s11023-007-9079-x."),
      p("[2] Chollet, F. On the measure of intelligence. 2019."),
      p("[3] Gignac, G.E.; Szodorai, E.T. Defining intelligence: Bridging the gap between human and artificial perspectives. Intelligence 2024, 104, 101832. doi: 10.1016/j.intell.2024.101832."),
      p("[4] Sternberg, R.J. What is intelligence, really? The futile search for a Holy Grail. Learning and Individual Differences 2024, 116, 102568. doi: 10.1016/j.lindif.2024.102568."),
      p("[5] Friston, K. The free-energy principle: A unified brain theory? Nature Reviews Neuroscience 2010, 11, 127\u2013138. doi: 10.1038/nrn2787."),
      p("[6] Ramstead, M.J.D. et al. On Bayesian mechanics: A physics of and by beliefs. Interface Focus 2023, 13, 20220029. doi: 10.1098/rsfs.2022.0029."),
      p("[7] Pezzulo, G.; Parr, T.; Friston, K. Active inference as a theory of sentient behavior. Biological Psychology 2024, 186, 108741. doi: 10.1016/j.biopsycho.2023.108741."),
      p("[8] Pezzulo, G.; Parr, T.; Cisek, P.; Clark, A.; Friston, K. Generating meaning: Active inference and the scope and limits of passive AI. Trends in Cognitive Sciences 2024, 28, 97\u2013112. doi: 10.1016/j.tics.2023.10.002."),
      p("[9] Landauer, R. Irreversibility and heat generation in the computing process. IBM Journal of Research and Development 1961, 5, 183\u2013191. doi: 10.1147/rd.53.0183."),
      p("[10] Still, S.; Sivak, D.A.; Bell, A.J.; Crooks, G.E. Thermodynamics of prediction. Physical Review Letters 2012, 109, 120604. doi: 10.1103/PhysRevLett.109.120604."),
      p("[11] England, J.L. Dissipative adaptation in driven self-assembly. Nature Nanotechnology 2015, 10, 919\u2013923. doi: 10.1038/nnano.2015.250."),
      p("[12] Ali, A.; Ahmad, N.; de Groot, E.; van Gerven, M.A.J.; Kietzmann, T.C. Predictive coding is a consequence of energy efficiency in recurrent neural networks. Patterns 2022, 3, 100639. doi: 10.1016/j.patter.2022.100639."),
      p("[13] Kolchinsky, A.; Wolpert, D.H. Semantic information, autonomous agency and nonequilibrium statistical physics. Interface Focus 2018, 8, 20180041. doi: 10.1098/rsfs.2018.0041."),
      p("[14] Tishby, N.; Pereira, F.C.; Bialek, W. The information bottleneck method. In Proceedings of the 37th Annual Allerton Conference on Communication, Control, and Computing, 1999."),
      p("[15] Klyubin, A.S.; Polani, D.; Nehaniv, C.L. All else being equal be empowered. In Advances in Artificial Life: ECAL 2005, LNCS 3630, pp. 744\u2013753. doi: 10.1007/11553090_75."),
      p("[16] Tiomkin, S.; Nemenman, I.; Polani, D.; Tishby, N. Intrinsic motivation in dynamical control systems. PRX Life 2024, 2, 033009. doi: 10.1103/PRXLife.2.033009."),
      p("[17] Kiefer, A.B. Intrinsic motivation as constrained entropy maximization. Entropy 2025, 27, 372. doi: 10.3390/e27040372."),
      p("[18] Takahashi, K.; Hayashi, Y. Thermodynamic limits of physical intelligence. 2026."),
      p("[19] Fagan, P.D. Toward a physical theory of intelligence. 2025."),
      p("[20] Perrier, E. Watts-per-intelligence: Part I (energy efficiency). 2025."),
      p("[21] Xu, J.; Li, Z. Information physics of intelligence: Unifying logical depth and entropy under thermodynamic constraints. 2025."),
      p("[22] Han, J. Intelligence inertia: Physical principles and applications. 2026."),
      p("[23] Saad-Falcon, J.; Avanika, M. et al. Intelligence per watt: Measuring intelligence efficiency of local AI. 2025."),
    ]
  }]
});

Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync("/Users/biobook/Code/fti/Intelligence_as_Useful_Structure.docx", buffer);
  console.log("DOCX created successfully: " + buffer.length + " bytes");
});
