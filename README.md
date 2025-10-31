## HomoSplice

A small library for inferring exon and intron homologies of paralogous genes. Currenlty, the following core functions are implemented:
1. Extraction of paralogous sequences from provided genome sequences and annotations (one may create hybrid proteins including alternatively spliced exons)
2. Multiple sequence alignment
3. Plotting MSA on transcript structures (introns are resized for clarity)
4. Inferring exon and intron orthogroups
5. Searching for exons missing from orhtogroups in corresponding introns with blast
6. Pretty and flexible visualization of results based on matplotlib
7. Zoom to any desired MSA fragment for sequence inspection
