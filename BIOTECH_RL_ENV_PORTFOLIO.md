# A Portfolio of Verifiable RL Environments for Biotechnology

**Research note — 24 September 2026**

## Executive answer

Beyond `RetroEnv`, there are **33 credible, public-data-backed biotechnology environment families** worth considering now, plus **3 useful but access-controlled or clinically sensitive environments** that should be kept out of an unrestricted public release.

The count is deliberately conservative. It excludes tasks that are merely question answering, have no defensible verifier, require an LLM judge, or cannot be split without obvious leakage. Of the 33 core candidates:

- **21 are Grade A candidates**: a strong public-data basis and an executable or hidden-record verifier, although some are computationally expensive.
- **12 are Grade B candidates**: valuable and trainable, but they need more scientific normalization, multimodal compute, or careful handling of proxy labels.
- **3 additional Grade C concepts** are worth pursuing only behind controlled-data and clinical-safety gates.

The best initial portfolio is not 33 unrelated benchmarks. It is one shared environment framework with four first verticals:

1. `ProteinMutationEnv` — experimental mutation-effect planning.
2. `TargetEvidenceEnv` — auditable target–disease evidence synthesis.
3. `CellAnnotationEnv` — iterative single-cell quality control and annotation.
4. `TrialProtocolAuditEnv` — structured clinical-trial protocol consistency checking.

Those four span protein, target, cell, and clinical work while exercising the same core machinery: hidden evidence, tools, state, backtracking, deterministic checks, and leakage-aware held-out evaluation.

## What MMAI Gym gets right—and where this project should go further

[MMAI Gym for Science](https://arxiv.org/abs/2603.03517) is the relevant reference. The paper describes a training and evaluation suite with more than 400 tasks across six high-level categories: 2D molecules, 3D molecules, 2D proteins, 3D proteins, drug–gene interactions, and cross-domain tasks. It combines SFT and reinforcement fine-tuning and emphasizes held-out and out-of-distribution evaluation. Its demonstrated workloads include molecular optimization, ADMET prediction, functional-group reasoning, retrosynthesis, and 3D generation.

The useful lessons are:

- train one model across related scientific formats rather than one model per benchmark;
- preserve domain-native representations such as SMILES, FASTA, structures, and numerical assay values;
- mix prediction, constrained generation, and cross-modal work;
- explicitly test distribution shift rather than reporting only random-split accuracy;
- treat post-training data, rewards, and evaluation as one system.

However, a classification, regression, or generation row is not automatically an RL environment. The stronger contribution here should be **interactive scientific episodes**:

```text
inspect hidden-case context
→ choose a tool or analysis
→ observe a result
→ test a hypothesis
→ revise or backtrack
→ submit a structured, auditable artifact
→ run deterministic and held-out verification
```

MMAI Gym also describes a generic reward that encourages longer reasoning. That should **not** be copied. Reasoning length is easy to game and is not scientific quality. Reward the correctness of submitted intermediate objects, evidence provenance, successful recovery, final validity, and tool efficiency. Private chain-of-thought should never be required for grading.

## The admission test for a good biotech RL environment

An environment enters the core portfolio only if it meets all six criteria:

1. **A real decision loop:** the agent has at least two consequential choices and can recover from a failed intermediate action.
2. **Useful tools:** sequence search, structure inspection, statistical analysis, database retrieval, image measurement, or another domain tool changes what the agent can know.
3. **A non-LLM verifier:** parsing, an executable scientific calculation, record replay, a hidden experimental measurement, or a curated evidence graph determines most of the score.
4. **Public raw material:** the source has downloadable records, stable identifiers, and enough provenance to produce versioned tasks.
5. **A defensible split unit:** related molecules, proteins, donors, patients, studies, or publications can be kept in one partition.
6. **A useful learned behavior:** success transfers to an actual R&D workflow rather than only reproducing a benchmark label.

### Readiness labels

| Label | Meaning |
|---|---|
| **A** | Strong public v0: mature data, strong verifier, manageable normalization, and useful agent behavior. |
| **B** | Strong second wave: public and valuable, but labels are noisier, computation is heavier, or validation needs several proxies. |
| **C** | Gated research environment: controlled access, high clinical stakes, or scientific truth is too indirect for a public reward. |

## Portfolio: 33 core environments

### Molecular biology and genetics

| # | Environment | Multi-step episode and tools | Raw public data | Verifier and leakage-safe split | Grade |
|---:|---|---|---|---|:---:|
| 1 | **PrimerForgeEnv** — PCR/qPCR primer design | Inspect gene/transcripts → choose target region → generate pairs → run specificity, dimer, hairpin, and amplicon checks → revise → submit a ranked assay panel. Tools: Primer3, BLAST/in-silico PCR, transcript browser, thermodynamics. | NCBI RefSeq/Genome plus the executable design behavior documented by [NCBI Primer-BLAST](https://www.ncbi.nlm.nih.gov/tools/primer-blast/). Public assay records can supply positive and failed examples where licensing permits. | Hard checks for orientation, product size, Tm spread, secondary structure, target coverage, and genome/transcriptome specificity. Hold out entire genes, paralog families, organisms, and assay publications—not individual primer pairs. | **A** |
| 2 | **CRISPRGuideEnv** — guide selection and off-target audit | Inspect locus and isoforms → choose nuclease/PAM → propose guides → enumerate off-targets → inspect functional consequences → revise → submit a small guide panel. Tools: genome browser, aligner, CRISPOR-compatible scores, variant overlap, exon/isoform inspector. | Reference genomes and transcript models from NCBI/Ensembl; public guide-efficiency and GUIDE-seq/CIRCLE-seq studies; [CRISPOR](https://github.com/maximilianh/crisporWebsite) provides an open scoring/tool reference. | Hard PAM, sequence, strand, target, uniqueness, and off-target checks; hidden experimental on-target values where available. Split by gene family, genomic locus, cell line/study, and publication. Restrict v0 to non-pathogenic, non-reproductive research targets. | **A** |
| 3 | **VariantEvidenceEnv** — research-grade variant interpretation | Normalize an HGVS/VCF variant → select transcript → run consequence prediction → retrieve assertions and functional evidence → resolve conflicts → submit an evidence table and uncertainty-aware classification. | [ClinVar downloads](https://www.ncbi.nlm.nih.gov/clinvar/docs/downloads/) provide complete XML plus VCF/TSV views; Ensembl VEP, UniProt, ProteinGym, and structure data add evidence. | Verify HGVS normalization, allele/transcript consistency, evidence citations, prohibited circular evidence, and agreement with a **future** expert-reviewed ClinVar snapshot. Split by gene family and publication; temporal evaluation is essential. This tests evidence assembly, not medical diagnosis. | **B** |
| 4 | **LocusToGeneEnv** — GWAS locus-to-gene prioritization | Inspect a locus → perform LD/credible-set filtering → query eQTL and regulatory context → connect variants to genes/pathways → compare hypotheses → submit ranked genes with evidence. | [GWAS Catalog summary statistics](https://www.ebi.ac.uk/gwas/downloads/summary-statistics), public GTEx summary resources, Open Targets Genetics/Platform, Ensembl, and functional annotations. GWAS Catalog notes that most summary statistics are CC0 but some studies have separate terms. | Recompute credible-set and overlap statistics; verify cited variant/gene/evidence edges; score against held-out colocalization, perturbation, or curated evidence. Group by LD block, trait family, cohort, and publication, then apply a time cutoff. | **B** |
| 5 | **RegulatoryElementEnv** — enhancer/promoter evidence planning | Inspect a genomic interval → choose relevant cell type → query accessibility, ChIP-seq, expression, and perturbation tracks → propose target gene and experiment → challenge alternatives → submit a causal evidence graph. | [ENCODE data](https://www.encodeproject.org/data/), GEO/SRA, Ensembl regulatory annotations, and validated enhancer/CRISPR perturbation records. | Validate coordinates, genome build, cell context, track provenance, motif calculations, and graph consistency; use hidden CRISPRi/reporter outcomes when available. Split by locus, gene neighborhood, biosample, experiment, and publication. | **B** |
| 6 | **RNAFunctionEnv** — non-coding RNA family and structure annotation | Inspect sequence → run similarity search → run covariance/folding tools → compare families/structures → identify conflicting hits → submit family, boundaries, structure, and confidence. | [RNAcentral downloads](https://rnacentral.org/downloads) provide versioned FASTA, mappings, Rfam annotations, APIs, and CC0 data; Rfam covariance models and experimentally resolved RNA structures add references. | Executable Infernal/cmscan and folding checks, boundary overlap, covariance score, base-pair validity, and hidden family labels. Split by Rfam clan and sequence-identity cluster—not random sequences. | **A** |

### Proteins and biologics

| # | Environment | Multi-step episode and tools | Raw public data | Verifier and leakage-safe split | Grade |
|---:|---|---|---|---|:---:|
| 7 | **ProteinMutationEnv** — mutation-panel planning | Inspect sequence, assay, and structure → propose informative mutations → query conservation/contacts → reject invalid or redundant choices → submit a constrained panel and predicted effects. | [ProteinGym](https://proteingym.org/) currently exposes DMS substitution and indel benchmarks covering millions of measured mutants across hundreds of assays, with clinical benchmarks as a separate regime. | Hard sequence and numbering checks; diversity/budget constraints; hidden DMS values for effect ranking and calibration. Split by protein sequence cluster/family, assay, organism, and publication. Never place mutants from one DMS assay across partitions. | **A** |
| 8 | **ProteinStabilityEnv** — stability engineering under constraints | Inspect sequence/structure → locate fragile regions → propose a small mutation set → run packing, exposure, and energy tools → revise after clashes or destabilization → submit ranked variants. | The [Tsuboyama megascale stability release](https://zenodo.org/records/7844779) includes processed stability values, raw NGS counts, structures, and analysis scripts; PDB and ProteinGym add external evaluation. | Hard residue/structure checks plus hidden experimental ΔG/ΔΔG. Separate homologous domains by stringent sequence/structure clustering and keep all single/double mutants of a parent in one split. | **A** |
| 9 | **PPIEngineeringEnv** — interface mutation planning | Inspect a complex → map interface residues → propose mutations for stronger/weaker binding → calculate contacts/clashes/electrostatics → revise → submit a mutation plan. | [SKEMPI 2.0](https://life.bsc.es/pid/skempi2/database/index) provides downloadable mutation energetics/kinetics and cleaned PDB files; the [wwPDB archive](https://www.wwpdb.org/ftp/pdb-ftp-sites) provides versioned structures. | Check chain/residue mapping, interface membership, structure geometry, and hidden ΔΔG. Split by complex, interacting protein-family pair, PDB lineage, and publication. Mutations from one complex remain together. | **A** |
| 10 | **ProteinFunctionEnv** — evidence-backed function annotation | Inspect a sequence → search homologs/domains → inspect structures and active-site residues → traverse ontology candidates → submit GO terms with evidence and calibrated specificity. | UniProt reference proteomes and reviewed records are downloadable through [UniProt](https://www.uniprot.org/help/proteome); [GO annotations](https://geneontology.org/docs/download-go-annotations/) provide evidence-coded, versioned GAF/GPAD files. | Validate identifiers, ontology consistency, taxon constraints, evidence types, and hierarchical precision/recall. Evaluate on future annotation releases and remote homology clusters; keep annotations published after the task cutoff out of visible tools. | **A** |
| 11 | **EnzymeFunctionEnv** — reaction and EC assignment | Inspect protein/domain → retrieve homologs and conserved residues → query balanced candidate reactions → check cofactors/direction → submit Rhea reaction/EC assignments and alternatives. | [Rhea downloads](https://www.rhea-db.org/help/download) provide reactions, participants, cross-references, and UniProt/EC links; reviewed UniProt enzymes provide sequence evidence. | Verify reaction balance, participant identifiers, directionality, EC hierarchy, catalytic-residue consistency, and hidden reviewed assignments. Split by reaction family, EC subtree, and protein sequence cluster with temporal release holdout. | **A** |
| 12 | **StructureQAEnv** — structure quality and residue mapping | Load an experimental or predicted model → map sequence/numbering → identify missing/uncertain regions → run geometry and clash checks → compare alternate structures → submit a quality report and usable-region mask. | [wwPDB versioned archives](https://www.wwpdb.org/ftp/pdb-ftp-sites) and [AlphaFold DB](https://alphafold.ebi.ac.uk/) structures, confidence values, and PAE. AlphaFold DB data are CC BY 4.0. | Executable chain mapping, bond/valence, Ramachandran, clash, missing-residue, pLDDT/PAE, and alignment checks. Split by sequence cluster, fold/superfamily, deposition date, and structural lineage. | **A** |
| 13 | **AntibodyInterfaceEnv** — antibody–antigen interface design | Inspect complex and CDRs → map contacts/epitope → propose a constrained mutation panel → run numbering, liability, clash, and interface checks → revise → submit candidates. | Public antibody structures from SAbDab/PDB and experimental affinity records where redistribution is permitted. [SAbDab](https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab/) standardizes public antibody structures. | Hard antibody numbering, CDR location, sequence validity, liability and geometry checks; hidden affinity labels only where measured. Split by antigen family, antibody sequence cluster, epitope, and study. Geometry is not proof of improved binding, so this remains Grade B. | **B** |

### Targets and pharmacology

| # | Environment | Multi-step episode and tools | Raw public data | Verifier and leakage-safe split | Grade |
|---:|---|---|---|---|:---:|
| 14 | **TargetEvidenceEnv** — target–disease prioritization | Parse a disease hypothesis → retrieve genetic, expression, pathway, tractability, safety, and drug evidence → identify contradictions → rank targets → submit an auditable evidence matrix. | [Open Targets data downloads](https://platform-docs.opentargets.org/data-access/datasets) expose versioned Parquet datasets for targets, diseases, drugs, variants, evidence, and associations; source-specific provenance must be retained. | Verify every evidence edge against the frozen graph, recompute component scores, penalize duplicated/dependent evidence, and hide future or selected evidence sources. Split by disease ontology branch, target family, evidence publication, and release date. Dataset-supported is not experimentally validated. | **A** |
| 15 | **BioactivityTriageEnv** — assay-aware drug–target activity reasoning | Normalize compound/target/assay → filter incomparable measurements → handle qualifiers and units → identify series and outliers → rank compounds or request the next assay. | [ChEMBL](https://www.ebi.ac.uk/chembl/) is a curated bioactivity database under CC BY-SA 3.0; [BindingDB downloads](https://ww.bindingdb.org/rwd/bind/chemsearch/marvin/Download.jsp) provide article-curated binding records and structures. | Hard unit, relation, target, assay, and structure normalization; replay hidden measurements; score calibration and experimental selection. Split jointly by Bemis–Murcko scaffold, target family, assay, document/patent, and time. | **A** |
| 16 | **CancerDependencyEnv** — context-specific target selection | Define cancer context → inspect expression/mutation/copy number → query CRISPR dependency → compare pan-essentiality and biomarkers → reject confounded targets → submit ranked dependencies and validation experiments. | [DepMap](https://depmap.org/portal/data_page/) publishes versioned CRISPR screens, expression, mutation, copy-number, fusion, and PRISM drug-screen data. | Verify dataset/entity joins, context filters, dependency statistics, selectivity, and hidden cell-line results. Split by gene family **and** lineage/cell-line cluster; use future DepMap releases for temporal evaluation. | **A** |
| 17 | **DrugSynergyEnv** — combination experiment planning | Inspect cell context and single-agent curves → propose drug pairs/doses → check mechanism redundancy and toxicity proxies → allocate a limited experiment budget → update after simulated results → submit a final panel. | [NCI-ALMANAC/CellMiner](https://www.discover.nci.nih.gov/cellminer/html/drug_almanac_combo_score.html) exposes public combination scores across NCI-60 cell lines and thousands of drug pairs. | Recompute synergy from dose-response records, enforce dose/budget constraints, and score hidden combinations. Split by both chemical scaffold pair and mechanism/target pair, with entire cell lines or tissue contexts held out for OOD. | **B** |
| 18 | **ADMETPortfolioEnv** — multi-objective candidate triage | Inspect candidate series → select property/assay tools → eliminate hard failures → reason over potency, permeability, metabolism, toxicity, and uncertainty → submit a diverse portfolio under a budget. | ChEMBL assay records, open TDC component datasets where their upstream licenses permit redistribution, MoleculeNet sources, and public ADMET studies. | Hard structure and constraint checks plus hidden experimental endpoints; Pareto-front, calibration, diversity, and cost rewards. Split by scaffold, chemical series/document, assay, target, and time—not random rows. | **A** |

### Cells, perturbations, and tissue

| # | Environment | Multi-step episode and tools | Raw public data | Verifier and leakage-safe split | Grade |
|---:|---|---|---|---|:---:|
| 19 | **CellAnnotationEnv** — iterative single-cell QC and annotation | Inspect count matrix/metadata → run QC and doublet checks → choose normalization/batch strategy → query marker/reference tools → annotate at a justified resolution → flag unknowns. | [CZ CELLxGENE Discover](https://cellxgene.cziscience.com/docs/03__Download%20Published%20Data) provides standardized downloadable h5ad data; the [Human Cell Atlas](https://data.humancellatlas.org/) supplies open multi-omic projects and donor metadata. | Execute QC/normalization steps, validate ontology labels and marker consistency, and compare hidden author/consensus labels while rewarding abstention. Split by study, donor, tissue, protocol, lab, and disease context; never split individual cells at random. | **A** |
| 20 | **PerturbationResponseEnv** — perturbation selection and response prediction | Define a cellular objective → inspect baseline state and target expression → choose perturbation and controls → predict responsive genes/cell states → update from a simulated partial readout → submit final prediction or next experiment. | [scPerturb](https://www.nature.com/articles/s41592-023-02144-y) publishes harmonized single-cell perturbation-response datasets in h5ad and related formats; public Perturb-seq studies can be snapshotted directly. | Check target/guide validity, controls, expressed genes, DE calculations, and hidden response profiles. Split by study, donor/cell line, perturbation target family, perturbation modality, and time. | **B** |
| 21 | **MorphologyMoAEnv** — Cell Painting mechanism-of-action investigation | Inspect image/profile QC → find replicates → correct batch effects → retrieve morphological neighbors → compare chemical/genetic perturbations → submit MoA and follow-up experiments. | The [Cell Painting Gallery](https://broadinstitute.github.io/cellpainting-gallery/overview.html) provides public images, extracted features, and metadata; [JUMP profiles](https://broadinstitute.github.io/jump_hub/howto/notebooks/11_retrieve_profiles.html) are versioned Parquet files. | Recompute replicate and retrieval metrics, verify batch-aware transformations, and compare hidden MoA/target labels. Split by compound scaffold, perturbation mechanism, plate/batch/site, and study. | **A** |
| 22 | **SpatialTissueEnv** — spatial cell typing and region mapping | Inspect image/spatial counts → run QC/segmentation → choose reference atlas → annotate cells/regions → test neighborhood hypotheses → revise → submit a tissue map and uncertainty mask. | [HuBMAP](https://hubmapconsortium.org/hubmap-data/) provides downloadable multimodal spatial and single-cell tissue data; HCA and CELLxGENE add references. | Geometry/coordinate checks, segmentation overlap, marker consistency, ontology validity, neighborhood statistics, and hidden annotations. Split at donor/specimen/block/study/site level, never patches from the same slide across splits. | **B** |
| 23 | **ExpressionPathwayEnv** — reproducible differential-expression investigation | Inspect study design → construct the correct contrast → run QC/normalization → detect covariates → compute DE → perform pathway analysis → challenge the result → submit a reproducible report. | [NCBI GEO](https://www.ncbi.nlm.nih.gov/geo/info/download.html) makes records and raw files available for bulk download; [Reactome downloads](https://reactome.org/download-data) provide versioned pathways and analysis APIs. | Re-execute the submitted analysis in a pinned container, validate sample/contrast logic, detect leakage/confounding, and compare to hidden author-supported signals. Split by accession/study, publication, cohort, lab, platform, and disease family. | **A** |
| 24 | **RegulatoryNetworkEnv** — gene-network reconstruction and intervention | Inspect perturbation/time-course data → select variables → infer candidate edges → test directionality/stability → choose an intervention → update from hidden response → submit a network and experiment. | ENCODE, GEO/SRA perturbation and time-course studies, and open DREAM challenge datasets where access and redistribution terms permit. | Executable graph constraints, held-out intervention prediction, edge precision/recall, causal direction tests, and stability across resamples. Split by organism, cell context, study, perturbation target family, and network module. | **B** |

### Clinical research, cancer, and biomedical omics

| # | Environment | Multi-step episode and tools | Raw public data | Verifier and leakage-safe split | Grade |
|---:|---|---|---|---|:---:|
| 25 | **TrialEligibilityEnv** — protocol-to-executable eligibility | Parse trial criteria → normalize diseases, interventions, labs, and time windows → resolve contradictions → compile criteria into rules → test against generated/de-identified cases → revise → submit executable criteria. | [ClinicalTrials.gov API v2](https://clinicaltrials.gov/data-about-studies/api-migration) supports per-study JSON and a full JSON archive. Cases should be procedurally generated from held-out protocol constraints unless a separately governed cohort is available. | Parser/schema checks plus execution against hidden positive, negative, boundary, and adversarial cases. Split by trial family, intervention, indication, sponsor, protocol text similarity, and registration time. Research matching only—no patient recommendation. | **B** |
| 26 | **TrialProtocolAuditEnv** — registry consistency and design audit | Inspect a protocol record → normalize arms, outcomes, time frames, enrollment, and criteria → run consistency checks → identify missing/contradictory fields → propose a machine-readable repair → submit audit. | ClinicalTrials.gov versioned records; historical snapshots or AACT-style relational snapshots can support change detection. | Almost entirely deterministic: schema, units, arm/intervention links, outcome time frames, age ranges, dates, enrollment arithmetic, duplicate criteria, and cross-version consistency. Split by trial family, sponsor, indication, intervention, and time. | **A** |
| 27 | **SafetySignalEnv** — pharmacovigilance signal investigation | Normalize drug/event names → deduplicate reports → choose a background → compute disproportionality and time trends → inspect confounding/indication → compare labels → submit a signal report with limitations. | [FDA FAERS quarterly files](https://www.fda.gov/drugs/drug-approvals-and-databases/fda-adverse-event-reporting-system-faers-database) and [openFDA downloads](https://open.fda.gov/data/). | Recompute counts, PRR/ROR/IC and confidence intervals, verify deduplication and mappings, and test on future quarters or label changes. Split temporally and by active ingredient/event hierarchy. Reward **signal detection, never causal claims**. | **B** |
| 28 | **DigitalPathologyEnv** — slide-region measurement and biomarker reasoning | Inspect a whole-slide image at low resolution → choose regions → run segmentation/detection tools → measure morphology → cross-check molecular/clinical metadata → revise → submit regions, measurements, and a research label. | [NCI Imaging Data Commons](https://datacommons.cancer.gov/repository/imaging-data-commons) includes public histopathology and annotations; TCGA data are available through [GDC](https://gdc.cancer.gov/access-data). | Patient/specimen identity checks, segmentation IoU/Dice, count and morphometry replay, hidden biomarker labels, and calibration. Split by patient, specimen, slide, acquisition site, and study; patch-level random splits are forbidden. | **B** |
| 29 | **CancerImagingEnv** — radiology segmentation and response measurement | Inspect a scan/series → select relevant phases → localize lesions → call segmentation and measurement tools → compare time points → correct implausible outputs → submit masks and measurements. | IDC exposes public radiology, segmentations, measurements, and clinical context with per-file licenses; its official catalog reports over 100 TB and mostly CC BY content. | DICOM/series consistency, anatomy and geometry checks, mask overlap, lesion measurements, longitudinal identity, and hidden expert annotations. Split by patient, institution, scanner/protocol, collection, and time. | **A** |
| 30 | **MultiOmicsCohortEnv** — cancer cohort hypothesis testing | Define cohort → inspect missingness/batches → join clinical, mutation, expression, copy-number, methylation, and proteomic layers → choose analysis → test robustness → submit a reproducible association and caveats. | [TCGA](https://www.cancer.gov/ccg/research/genome-sequencing/tcga) characterized more than 20,000 primary/matched samples across 33 cancer types; GDC provides public harmonized clinical and genomic data, with some controlled layers. | Re-run code, validate patient/sample joins and endpoint definitions, audit covariates and multiple testing, and evaluate on held-out cancer types or cohorts. Split by patient, study, cancer type, center, and time. Associations are not clinical recommendations. | **B** |
| 31 | **ProteomicsIDEnv** — peptide/protein identification and QC | Inspect spectra/metadata → choose search parameters and database → run search → control target-decoy FDR → diagnose contamination/modifications → revise → submit peptide/protein IDs and QC. | [PRIDE Archive](https://www.ebi.ac.uk/training/online/courses/pride-quick-tour/downloading-data-using-pride-archive/) provides raw files and standard processed formats such as mzIdentML/mzTab through FTP/Aspera. | Re-execute search on bounded spectra, validate mass tolerance/enzyme/modifications, target-decoy FDR, peptide uniqueness, and hidden IDs. Split by project, biological sample, instrument, lab, and peptide/protein homology. | **A** |
| 32 | **MetabolomicsIDEnv** — MS/MS identification and dereplication | Inspect precursor/adduct and spectrum → clean peaks → search libraries → compare formulas/fragments → inspect analog families → revise → submit candidates with confidence and evidence. | [GNPS](https://gnps.ucsd.edu/ProteoSAFe/static/gnps-splash.jsp) is an open-access ecosystem for raw, processed, and annotated MS/MS data with public datasets and spectral libraries. | Precursor/adduct/formula consistency, spectral cosine and matched peaks, decoy/FDR controls, library provenance, and hidden standard IDs. Split by molecular scaffold/InChIKey connectivity layer, dataset, instrument, lab, and acquisition mode. | **A** |
| 33 | **BioinformaticsRepairEnv** — scientific pipeline debugging | Receive a broken, containerized analysis over a small public dataset → inspect logs/files → choose diagnostics → patch parameters/config/code → rerun tests → submit a reproducible output and minimal repair. | Small pinned subsets of GEO/SRA, PRIDE, CELLxGENE, or synthetic fixtures; open nf-core or community workflow test profiles supply realistic pipeline structure. Faults are procedurally injected rather than copied from private incidents. | Strongest verifier in the portfolio: container exit status, unit/integration tests, checksums, schema/biological invariants, expected summary statistics, and patch scope. Split by workflow family, fault mechanism, tool/version, and source study. | **A** |

## Three gated extensions

These are real use cases, but they should not be bundled into an unrestricted Hugging Face release:

| # | Environment | Why gated | Suitable data/evaluation |
|---:|---|---|---|
| 34 | **EHRPhenotypeEnv** — cohort definition and phenotype extraction | Clinical notes and event timelines require credentialing, data-use agreements, PHI controls, and strong misuse safeguards. | MIMIC-IV or another governed de-identified EHR; verify executable SQL/FHIR queries and chart-derived labels. Group by patient and admission, then split temporally. Never redistribute source data. |
| 35 | **ClinicalTimelineEnv** — event reconstruction and deterioration research | High clinical stakes and severe shortcut risks; a predictive score can be mistaken for deployable medical guidance. | Credentialed EHR data; hidden future events and executable timeline checks. Evaluate calibration, abstention, subgroup behavior, and site/time transfer under an explicit research-only policy. |
| 36 | **PKSimulationEnv** — population-PK model selection and dose simulation | The computation is verifiable, but a public agent must not produce patient-specific dosing advice. | Public PK studies/PK-DB and synthetic virtual patients. Verify units, model equations, parameter recovery, and simulated concentrations. Keep all outputs explicitly non-clinical and scenario-based. |

## Recommended build order

### Wave 0 — shared substrate

Build these once before creating more verticals:

```text
source registry + immutable snapshots + checksums
→ canonical entity/ontology mapping
→ cross-source duplicate graph
→ group-aware and temporal splitter
→ common task schema
→ stateful tool runtime
→ hidden verifier service
→ evaluator and contamination report
```

Every environment should share a task envelope like:

```json
{
  "task_id": "protein_mutation:PG:assay:case",
  "environment": "ProteinMutationEnv",
  "observation": {},
  "objective": {},
  "constraints": {},
  "available_tools": [],
  "public_provenance": [],
  "split_tags": {},
  "hidden_reference_key": "evaluator-only"
}
```

The public row contains no answer, future evidence, reference route, hidden measurement, or feature derived from the test label.

### Wave 1 — four proof environments

| Order | Environment | Why first | v0 target |
|---:|---|---|---|
| 1 | `ProteinMutationEnv` | Large experimental labels, compact data, clear family-aware OOD splits, and a direct hidden-measurement reward. | 10–20 DMS assays for training, family-held-out dev/test, 3–5 tools, panel-selection tasks. |
| 2 | `TargetEvidenceEnv` | Strongly agentic: retrieval, source weighing, contradiction handling, and an auditable evidence graph. | One frozen Open Targets release, 3 disease branches, held-out sources and later-release evaluation. |
| 3 | `CellAnnotationEnv` | Tests numerical tools, ontology use, QC, abstention, and cross-study generalization. | A small standardized CELLxGENE atlas with donor/study/tissue OOD tasks. |
| 4 | `TrialProtocolAuditEnv` | Low compute, excellent deterministic verification, clinical-research relevance, and no patient records. | 5,000 versioned study records, synthetic faults plus naturally occurring inconsistencies, temporal test set. |

This first wave is intentionally heterogeneous. If a shared environment API works for all four, it is likely general enough for most of the portfolio.

### Wave 2 — expand action types

Add `PrimerForgeEnv`, `PPIEngineeringEnv`, `CancerDependencyEnv`, `MorphologyMoAEnv`, `ProteomicsIDEnv`, `MetabolomicsIDEnv`, and `BioinformaticsRepairEnv`. These introduce constrained sequence generation, 3D structure, causal screening, images, spectra, and executable code repair.

### Wave 3 — multimodal and proxy-heavy tasks

Add antibody design, locus-to-gene, spatial tissue, perturbation response, pharmacovigilance, pathology, and multi-omics only after the framework can clearly distinguish:

```text
valid computation
≠ dataset-supported conclusion
≠ causal biological result
≠ experimentally validated result
≠ clinically actionable result
```

## A common interaction design

The tools should expose small, typed operations, not unrestricted answer-bearing database dumps. A representative interface is:

```text
inspect_case()
get_sequence_or_structure(entity_id)
search_evidence(query, filters)
run_analysis(method, inputs, parameters)
validate_intermediate(object)
compare_hypotheses(candidate_ids)
discard_candidate(candidate_id, reason_code)
submit_artifact(schema_version, artifact)
```

Important environment behaviors:

- A tool result becomes part of episode state and can be cited by stable result ID.
- Expensive tools consume a budget; cheap validation tools do not need to be artificially scarce.
- Invalid actions return a diagnostic that enables recovery rather than immediately ending every episode.
- The agent can submit early, but hidden evaluation runs only on a schema-valid artifact.
- Evidence visible to the agent is separated from evidence used by the verifier.
- The evaluator records tool calls, revisions, invalid proposal rate, and final success without requiring private reasoning traces.

## Reward design

Use hard gates before soft reward. A generic template is:

```text
hard gates:
  parseable artifact
  identifiers and units valid
  no forbidden data access
  no train/test or hidden-evidence access
  core structural/graph/sequence constraints satisfied

soft score:
  0.35 hidden outcome or experimental agreement
  0.25 executable scientific validity
  0.20 evidence quality and provenance
  0.10 uncertainty calibration / appropriate abstention
  0.10 cost and tool efficiency
```

Weights should differ by environment. For example, `TrialProtocolAuditEnv` should emphasize executable consistency, while `ProteinMutationEnv` can emphasize hidden DMS ranking. No environment should reward verbosity, explanations, or the number of reasoning tokens.

### Metrics common across the portfolio

- `Success@1` and `Pass@k`
- full-artifact validity
- intermediate-action validity
- recovery rate after a failed tool result
- hidden-label or hidden-measurement performance
- calibration and selective accuracy when abstention is allowed
- tool calls and compute cost per successful episode
- citation/evidence precision and coverage
- ID, OOD, and temporal performance separately
- invalid proposal and hidden-data-access rates

## Leakage prevention: split biological entities, not rows

There is no honest universal claim of “zero leakage,” especially when evaluating pretrained foundation models. What can be guaranteed is that the released pipeline prevents known **dataset-internal leakage**, reports remaining contamination risks, and makes every split reproducible.

### Required split groups

| Domain | Primary grouping keys that must never cross splits | Recommended headline OOD split |
|---|---|---|
| Small molecules | canonical parent structure, stereoisomer policy, Bemis–Murcko scaffold, matched molecular series, patent/paper, assay | scaffold + document + target-family OOD |
| Reactions/routes | canonical reaction center, template, product scaffold, route graph, patent/publication | product-scaffold and reaction-class OOD |
| Proteins | sequence-identity cluster, domain/family, parent construct, assay, PDB lineage, publication | remote-family or fold OOD |
| Protein mutations | parent protein/construct and entire DMS assay | protein-family OOD; never mutant-row random split |
| Genetics | variant normalization, LD block/locus, gene family, trait, cohort, publication | locus/trait plus temporal OOD |
| Single-cell | study, donor, specimen, tissue, protocol, lab, batch | unseen study/donor/tissue |
| Perturbation | study, cell context, target/compound family, guide, donor | unseen perturbation family and cell context |
| Tissue/images | patient, specimen, slide/series, site, scanner/protocol, collection | unseen institution/collection |
| Clinical trials | NCT family, intervention, indication, sponsor, near-duplicate protocol, date | future registration/results period plus intervention OOD |
| EHR | patient, admission, site, time | future/site OOD; controlled only |
| Proteomics/metabolomics | project, sample, instrument, lab, peptide/protein family or chemical scaffold | unseen project/instrument and entity family |
| Literature/evidence graphs | DOI/PMID, preprint–paper family, supplementary dataset, cited record | publication-time OOD |

### Split procedure

1. Normalize identifiers and create canonical entities before splitting.
2. Build a graph connecting exact duplicates, near duplicates, shared parents, papers, patents, donors/patients, studies, assays, and derived records.
3. Collapse connected leakage components into indivisible groups.
4. Assign groups with stratified group optimization, not row shuffling.
5. Apply a temporal cutoff after grouping; no post-cutoff annotation can appear in train features or retrieval tools.
6. Re-run cross-split similarity audits: sequence identity, scaffold similarity, text MinHash, image/patient identifiers, study accessions, and citation relationships.
7. Freeze test sources and checksums before generating training traces.
8. Keep hidden evaluation artifacts on a separate service or private repository.
9. Publish a machine-readable leakage report with thresholds, exclusions, and known residual risks.

### Cross-dataset leakage is the difficult part

The same experiment may appear in GEO and a paper supplement; the same bioactivity can appear in ChEMBL and BindingDB; the same structure can appear in PDB-derived benchmarks; the same clinical trial can have registry updates and publications. Source names are not split boundaries. The source registry therefore needs an entity-resolution layer with:

```text
source accession
original publication/patent
canonical biological entities
assay or specimen identity
derived-from links
supersedes/version links
license and redistribution policy
snapshot date and checksum
```

## Data acquisition and release policy

Do not interpret “all public biotech data” as a single indiscriminate download. Publicly reachable is not the same as redistributable, compatible, current, or scientifically comparable. Use a manifest-driven source registry:

```yaml
source_id: proteingym
upstream_url: https://proteingym.org/
release_or_snapshot: pinned-version
retrieved_at: ISO-8601
license: reviewed-string
redistribution: allowed | derived-only | metadata-only | prohibited
files:
  - path: ...
    sha256: ...
provenance_fields: [...]
split_group_fields: [...]
parser_version: git-sha
```

For each source:

- preserve the untouched raw file;
- store a checksum and retrieval timestamp;
- version parsers separately from data;
- propagate row-level or study-level licensing where needed;
- record exclusions and failed parses;
- keep controlled data outside the repository and outside public HF artifacts;
- publish derived task rows only when the upstream terms allow it;
- provide download scripts instead of mirroring restricted or very large sources.

Notable licensing cautions include per-study exceptions in GWAS summary statistics, source-specific terms inherited by Open Targets, share-alike obligations for ChEMBL, collection-level licenses in IDC/TCIA, and submitter-specific constraints in archival repositories. A license review is part of dataset engineering, not a final documentation chore.

## Suggested repository layout

```text
FineEnvs/
├── 04-retroenv/
├── 05-protein-mutation-env/
├── 06-target-evidence-env/
├── 07-cell-annotation-env/
├── 08-trial-protocol-audit-env/
└── biotech-core/
    ├── source_registry/
    ├── entity_resolution/
    ├── splitters/
    ├── task_schema/
    ├── tool_runtime/
    ├── verifier_sdk/
    ├── evaluators/
    └── contamination_audit/
```

Avoid copying chemistry-specific code into every folder. The environment SDK should make state, tool budgets, hidden verifier calls, task versioning, and evaluator traces common; each scientific environment supplies only domain tools, schemas, hard gates, rewards, and split logic.

## What not to build first

- free-form “design a therapeutic protein/drug and explain why it works” tasks;
- patient diagnosis, treatment, or dose recommendation;
- exact-match grading against one scientific narrative;
- environments whose main reward is an LLM judge;
- random cell-, mutation-, spectrum-, image-patch-, or assay-row splits;
- target discovery scored only by an association score already shown to the agent;
- antibody or molecule generation scored only by docking or template similarity;
- trial-success prediction without a temporal split and explicit missing-result handling;
- literature QA whose answer is present verbatim in the retrieved abstract;
- any reward for long reasoning or persuasive prose.

## Bottom line

The opportunity is larger than retrosynthesis: **33 strong public-data environment families** cover molecular biology, genetics, proteins, targets, cells, tissue, clinical research, imaging, and omics. But the defensible contribution is not a directory of 33 static benchmarks. It is a reproducible system that converts raw public science into:

```text
versioned corpus
→ leakage-component graph
→ task generator
→ multi-step tool environment
→ deterministic/held-out verifier
→ OOD benchmark
→ post-training recipe
```

Build four diverse environments first, prove that their verifiers and splits survive adversarial audit, and then scale the shared framework across the portfolio.

## Primary source index

- [MMAI Gym for Science paper](https://arxiv.org/abs/2603.03517)
- [ProteinGym](https://proteingym.org/)
- [SKEMPI 2.0 downloads](https://life.bsc.es/pid/skempi2/database/index)
- [wwPDB archive downloads](https://www.wwpdb.org/ftp/pdb-ftp-sites)
- [AlphaFold Protein Structure Database](https://alphafold.ebi.ac.uk/)
- [UniProt proteome downloads](https://www.uniprot.org/help/proteome)
- [Gene Ontology annotation downloads](https://geneontology.org/docs/download-go-annotations/)
- [Rhea downloads](https://www.rhea-db.org/help/download)
- [RNAcentral downloads](https://rnacentral.org/downloads)
- [ClinVar downloads](https://www.ncbi.nlm.nih.gov/clinvar/docs/downloads/)
- [GWAS Catalog downloads](https://www.ebi.ac.uk/gwas/docs/downloads/)
- [ENCODE data portal](https://www.encodeproject.org/data/)
- [Open Targets downloads](https://platform-docs.opentargets.org/data-access/datasets)
- [ChEMBL](https://www.ebi.ac.uk/chembl/)
- [BindingDB downloads](https://ww.bindingdb.org/rwd/bind/chemsearch/marvin/Download.jsp)
- [DepMap data](https://depmap.org/portal/data_page/)
- [NCI-ALMANAC through CellMiner](https://www.discover.nci.nih.gov/cellminer/html/drug_almanac_combo_score.html)
- [CZ CELLxGENE data downloads](https://cellxgene.cziscience.com/docs/03__Download%20Published%20Data)
- [Human Cell Atlas data portal](https://data.humancellatlas.org/)
- [scPerturb](https://www.nature.com/articles/s41592-023-02144-y)
- [Cell Painting Gallery](https://broadinstitute.github.io/cellpainting-gallery/overview.html)
- [HuBMAP data](https://hubmapconsortium.org/hubmap-data/)
- [NCBI GEO downloads](https://www.ncbi.nlm.nih.gov/geo/info/download.html)
- [Reactome downloads](https://reactome.org/download-data)
- [ClinicalTrials.gov API v2 and bulk download](https://clinicaltrials.gov/data-about-studies/api-migration)
- [FDA FAERS](https://www.fda.gov/drugs/drug-approvals-and-databases/fda-adverse-event-reporting-system-faers-database)
- [NCI TCGA](https://www.cancer.gov/ccg/research/genome-sequencing/tcga)
- [NCI GDC](https://gdc.cancer.gov/access-data)
- [NCI Imaging Data Commons](https://datacommons.cancer.gov/repository/imaging-data-commons)
- [PRIDE Archive download guide](https://www.ebi.ac.uk/training/online/courses/pride-quick-tour/downloading-data-using-pride-archive/)
- [GNPS](https://gnps.ucsd.edu/ProteoSAFe/static/gnps-splash.jsp)
