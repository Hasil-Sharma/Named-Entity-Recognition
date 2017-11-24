# Named-Entity-Recognition

A learning based approach to named entity recognition to find all the references to genes in a set of biomedical journal
article abstracts. Different approaches tested out and corresponding result.

## HMM - Based

- [HMM with Part-of-Speech tags as feature][1]: `Precision: 0.23398, Recall: 0.07705, F-1 Score: 0.11593`
- [HMM (NLTK) with Part-of-Speech tags as feature][2]: `Precision: 0.24539, Recall: 0.07542, F-1 Score: 0.11886`

## MaxEnt - Based

- [MaxEnt with features as follow][3]: `Precision: 0.59823, Recall: 0.45590, F-1 Score:  	0.51746`
  - Part-of-Speech : Current, Previous and Next
  - Word: Current, Previous and Next
  - Word Shape: Current, Previous and Next
  - Lemma: Current, Previous and Next
  - Word Length: Current, Previous and Next
  - Suffix3: Current, Previous and Next
  - Prefix3: Current, Previous and Next
  - Current Word Index

[1]:https://github.com/Hasil-Sharma/Named-Entity-Recognition/blob/45c7d5497614e99391cec98e2ad198a43790cfc5/Named%20Entity%20Recognition.ipynb
[2]:https://github.com/Hasil-Sharma/Named-Entity-Recognition/blob/fc24d20f5b3fe4d7552d31a93bfc0eac62b63ce1/Named%20Entity%20Recognition.ipynb
[3]:https://github.com/Hasil-Sharma/Named-Entity-Recognition/blob/9f58a643c305b26f56b610451b6d35e2a4d67d88/Named%20Entity%20Recognition.ipynb
