NLM2007.pmids
	the pmids of the 200 testing documents
	
NLM2007.TiAbMe: 
    200 documents, each document has a title, abstract, and mesh term list

NLM2007_MTI.out:
     MTI results for those documents in NLM2007.TiAbMe; 
    the file format is available at: http://ii.nlm.nih.gov/

NLM2007_50neighbors.pmids:
    the pmid list of neighboring documents for those 200 documents in NLM2007.TiAbMe.

NLM2007_50neighbors.score:
    neighborsity score for the input document and neighboring documents (top 50 neighbors)
    file format:
      input_doc <tab> neighbor <tab> neighborsity_score

NLM2007_50neighbors.TiAbMe:
    the title, abstract, and mesh term list for those documents in NLM2007_50neighbors.pmids
