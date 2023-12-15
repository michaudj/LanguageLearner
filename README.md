# LanguageLearner
This repository contains the code for the supplementary material of the article "Opening the black box of language acquisition".

It contains a number of python files used to perform the simulations in that paper.
- Raw_input.py contains the code for generating probabilistic context free grammars in the format used by the learning agent
- MyLearners.py contains the classes defining the different types of learners
  
Learning curves are produces using:
- LearningCurves.py (4 learning curves for a NVN language)
- LearningCurvesMD.py (learning curve for a language containing mono- and ditransitive verbs)
- LearningCurvesRel.py (learning curve for a language containing relative clauses)
- LearningCurvesComplNP.py (learning curve for a language with complex NPs)

Grammar snapshots are produced using:
- GrammarSnapshots.py producing 4 Excel files: QLearnerC_paper.xlsx, QLearnerN_paper.xlsx, RWQLearnerC_paper.xlsx, and RWQLearnerN_paper.xlsx
- GrammarSnapshotsMD.py producing RWQLearnerC_MD_paper.xlsx
- grammarSnapshots-rel.py producing RWQLearnerC_rel.xlsx
- grammarSnapshotsComplNP.py producing RWQLearnerC_MDComplNP.xlsx

Sentences extracted for the langauge with relative clauses are stored in the sentencesRelClauses.xlsx and sentencesRelClauses1.xlsx

Sentences extracted for the Complex NP language are stored in the sentencesCompNP_paper.xlsx

Other files are not relevant for the paper.
