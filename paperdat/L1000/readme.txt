L1000.pmids:
	pmids for the set of 1000 testing documents

L1000.TiAbMe: 
	1000 documents, each document has a title, abstract, and mesh term list

L1000_MTI.out:	
	MTI results for those documents in L1000.TiAbMe; 
    the file format is available at: http://ii.nlm.nih.gov/

L1000_50neighbors.pmids:
    the pmid list of neighboring documents for those 1000 documents in L1000.TiAbMe.

L1000_50neighbors.score:
    neighborsity score for the input document and neighboring documents (top 50 neighbors)
    file format:
      input_doc <tab> neighbor <tab> neighborsity_score

L1000_50neighbors.TiAbMe:
    the title, abstract, and mesh term list for those documents in L1000_50neighbors.pmids;
