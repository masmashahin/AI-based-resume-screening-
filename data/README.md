# Dataset setup

The reference paper (Section 3.2.1) describes a public Kaggle corpus with **962 resumes and 25 job categories**. It does not give the Kaggle slug, but its size and schema correspond to the widely used Kaggle **Resume Dataset** containing `UpdatedResumeDataSet.csv` with `Category` and `Resume` columns.

Download that genuine dataset from Kaggle while signed in, then place the CSV here as:

```text
data/UpdatedResumeDataSet.csv
```

It is intentionally not committed to this repository: it is third-party data and training must use the real source rather than fabricated examples. Before training, the script checks and prints a warning if the file is not 962 rows / 25 categories.

For this prepared workspace, the file has already been obtained from the public GitHub mirror `jjso/resume-dataset`, whose README identifies it as copied from the original Kaggle source above. It was validated as 962 rows, 25 categories, and the expected `Category`/`Resume` columns.
