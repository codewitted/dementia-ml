# CHAPTER 3: METHODOLOGY

## 3.1 Software Development Methodology

This project employed an iterative, agile-inspired development approach adapted for research software development. Unlike traditional waterfall methods with rigid sequential phases, the iterative approach enabled continuous refinement based on experimental results, literature insights, and technical discoveries.

### 3.1.1 Iterative Development Framework

The development process followed repeating cycles of:

**Planning**: Define objectives for current iteration, prioritize features, identify technical requirements

**Implementation**: Develop code modules, implement algorithms, create tests

**Experimentation**: Run models, analyze results, identify issues

**Evaluation**: Assess performance against objectives, gather insights

**Refinement**: Incorporate lessons learned, improve implementation

This cyclic approach proved essential for ML research where initial assumptions often require revision based on experimental outcomes.

### 3.1.2 Development Phases

**Phase 1: Foundation (Weeks 1-2)**
- Environment setup and dependency management
- Repository initialization with version control
- Basic project structure and documentation
- Data acquisition and initial exploration

**Phase 2: Data Pipeline (Weeks 3-4)**
- Data loading module implementation
- Preprocessing pipeline development
- Feature engineering and selection
- Train-test split methodology
- Data validation and quality checks

**Phase 3: Baseline Models (Weeks 5-6)**
- Logistic Regression implementation
- Random Forest development
- Gradient Boosting Machine training
- Initial performance evaluation
- Hyperparameter tuning

**Phase 4: Ensemble Methods (Weeks 7-8)**
- Stacking Ensemble implementation
- Voting Ensemble development
- Cross-validation framework
- Meta-learner optimization
- Comparative performance analysis

**Phase 5: Evaluation and Explainability (Weeks 9-10)**
- Comprehensive metric calculation
- Statistical significance testing
- Feature importance analysis
- SHAP value implementation
- Visualization generation

**Phase 6: Refinement and Documentation (Weeks 11-12)**
- Code refactoring and optimization
- Comprehensive testing
- Documentation completion
- Reproducibility verification
- Final evaluation and dissertation writing

## 3.2 Justification for Methodology Choice

### 3.2.1 Why Iterative Over Waterfall?

Machine learning research inherently involves uncertainty and discovery. Initial design decisions often require revision based on experimental results:

**Experimentation-Driven**: ML development requires frequent experiments to tune hyperparameters, select features, and compare algorithms. Waterfall's rigid phases would prevent necessary iteration.

**Knowledge Discovery**: Understanding which approaches work best emerges through experimentation, not upfront specification.

**Flexible Response**: Ability to quickly pivot based on results is critical. For example, discovering that ensemble methods outperform individual models led to increased focus on ensemble development.

**Risk Management**: Iterative approach enables early detection of technical issues or flawed assumptions, allowing course correction.

### 3.2.2 Why Agile Principles?

While not following formal Scrum or Kanban, this project adopted agile principles:

**Working Software**: Emphasis on functional code over comprehensive documentation upfront

**Embrace Change**: Welcome requirement changes based on experimental insights

**Regular Delivery**: Frequent commits to version control with incremental improvements

**Technical Excellence**: Continuous attention to code quality and testing

**Simplicity**: Focus on essential features, avoid premature optimization

**Reflection**: Regular assessment of process and technical approach

### 3.2.3 Adaptations for Research Context

Research software development differs from commercial software:

**Single Developer**: No team coordination overhead, but requires self-discipline and planning

**Uncertain Requirements**: Research objectives evolve based on findings

**Scientific Rigor**: Emphasis on reproducibility and documentation exceeds typical software projects

**Publication Focus**: Deliverables include academic outputs (dissertation, papers) alongside code

**Limited Scope**: Time-bound project (12 weeks) requires careful scope management

## 3.3 Development Lifecycle

### 3.3.1 Requirements Analysis

Requirements emerged through:

**Literature Review**: Understanding state-of-the-art and best practices

**Dataset Analysis**: Identifying data characteristics and limitations

**Technical Constraints**: Assessing computational resources and tool capabilities

**Academic Standards**: Aligning with dissertation requirements and grading criteria

**Stakeholder Input**: Incorporating supervisor feedback

### 3.3.2 Design Process

System design followed principles of:

**Modularity**: Clear separation between data loading, preprocessing, modeling, and evaluation components enables independent development and testing

**Configurability**: Externalized parameters in YAML configuration files enable easy experimentation without code changes

**Extensibility**: Interface-based design allows adding new models or features without modifying existing code

**Reproducibility**: Fixed random seeds, versioned dependencies, and comprehensive logging ensure result reproduction

**Usability**: Simple command-line interface enables non-expert users to run complete pipeline

### 3.3.3 Implementation Strategy

**Bottom-Up Approach**: Build foundational components (data loading, preprocessing) before complex features (ensemble methods, SHAP analysis)

**Test-Driven Mindset**: Write tests for core functionality to catch errors early

**Incremental Integration**: Integrate components gradually, validating at each step

**Documentation While Coding**: Document design decisions, algorithms, and usage concurrently with implementation

**Version Control Discipline**: Commit frequently with meaningful messages documenting changes

### 3.3.4 Testing and Validation

**Unit Testing**: Individual function testing for data loading, preprocessing, feature encoding

**Integration Testing**: Combined component testing to verify pipeline flow

**System Testing**: End-to-end pipeline execution with known datasets

**Model Validation**: Cross-validation, hold-out testing, comparison with literature

**Reproducibility Testing**: Re-running pipeline multiple times to verify identical results

### 3.3.5 Deployment and Maintenance

**Local Execution**: System designed for local execution on standard hardware

**Environment Management**: Conda environment ensures consistent dependencies

**Version Pinning**: Exact package versions specified to prevent breaking changes

**Documentation**: Comprehensive README enables new users to run system

**Public Repository**: GitHub hosting enables community access and contribution

## 3.4 Project Management Approach

### 3.4.1 Planning and Scheduling

**Milestone Definition**: Clear milestones defined for each development phase

**Time Allocation**: Weekly time budgets ensuring steady progress

**Gantt Chart**: Visual timeline tracking planned vs. actual progress (see Appendix E)

**Buffer Time**: Built-in slack for unexpected challenges and iterations

**Regular Review**: Weekly progress assessment and next-week planning

### 3.4.2 Risk Management

**Risk 1: Data Quality Issues**
- *Mitigation*: Early data exploration, validation checks, handling missing values
- *Outcome*: Successfully handled missing data through imputation

**Risk 2: Poor Model Performance**
- *Mitigation*: Multiple algorithms, hyperparameter tuning, ensemble methods
- *Outcome*: Achieved target performance (AUC-ROC > 0.90)

**Risk 3: Computational Resource Limitations**
- *Mitigation*: Efficient algorithms, batch processing, cloud computing option
- *Outcome*: Standard laptop sufficient for OASIS dataset size

**Risk 4: Reproducibility Challenges**
- *Mitigation*: Fixed seeds, version control, comprehensive documentation
- *Outcome*: Complete reproducibility achieved

**Risk 5: Time Management**
- *Mitigation*: Focused scope, iterative delivery, priority management
- *Outcome*: All objectives completed on schedule

### 3.4.3 Communication and Supervision

**Weekly Supervisor Meetings**: Progress updates, technical discussions, guidance

**Meeting Minutes**: Documented decisions and action items from each meeting

**Email Updates**: Interim progress reports between meetings

**Code Sharing**: GitHub repository providing supervisor access to work-in-progress

**Demonstration**: Regular demos of working features and results

### 3.4.4 Quality Assurance

**Code Review**: Self-review before commits, supervisor feedback on key modules

**Testing**: Comprehensive test suite ensuring correctness

**Documentation Review**: Peer review of README and documentation

**Result Validation**: Comparing outputs against published benchmarks

**Reproducibility Verification**: Running pipeline on different machines

### 3.4.5 Learning and Development

**Technical Skills**: Developing expertise in scikit-learn, SHAP, ensemble methods

**Domain Knowledge**: Deep understanding of dementia, ML in healthcare

**Software Engineering**: Applying best practices in testing, documentation, version control

**Academic Writing**: Developing dissertation and technical communication skills

**Self-Directed Learning**: Researching algorithms, reading papers, troubleshooting issues

### 3.4.6 Tools and Infrastructure

**Version Control**: Git and GitHub for code management and collaboration

**Development Environment**: VS Code with Python extensions

**Package Management**: Conda for environment management

**Testing**: pytest for automated testing

**Documentation**: Markdown for README, reStructuredText for code docs

**Visualization**: matplotlib and seaborn for publication-quality figures

**Notebook**: Jupyter for exploratory analysis and prototyping

### 3.4.7 Milestone Achievements

| Milestone | Planned Date | Actual Date | Status |
|-----------|--------------|-------------|--------|
| Environment Setup | Week 1 | Week 1 | ✓ Complete |
| Data Pipeline | Week 4 | Week 4 | ✓ Complete |
| Baseline Models | Week 6 | Week 6 | ✓ Complete |
| Ensemble Methods | Week 8 | Week 8 | ✓ Complete |
| Evaluation Complete | Week 10 | Week 10 | ✓ Complete |
| Documentation Done | Week 12 | Week 12 | ✓ Complete |

All milestones achieved on schedule, demonstrating effective project management.

---

*End of Chapter 3*
