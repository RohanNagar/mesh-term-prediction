S200.pmids
	The pmids of the 200 training documents
	
S200.TiAbMe: 
    200 documents, each document has a title, abstract, and mesh term list

S200_MTI.out:
    MTI results for those documents in S200.pmids; 
    the file format is available at: http://ii.nlm.nih.gov/

S200_50neighbors.pmids:
    the pmid list of neighboring documents for those 200 documents in S200.pmids

S200_50neighbors.score:
    neighborsity score for the input document and neighboring documents (top 50 neighbors)
    file format:
      input_doc <tab> neighbor <tab> neighborsity_score

S200_50neighbors.TiAbMe:
    the title, abstract, and mesh term list for those documents in S200_50neighbors.pmid;
