import sys

from citation_grabber import write_citations

with open('citations.TiAbMe', 'a+') as citations_file:
    with open('paperdat/L1000/L1000.pmids') as pubmed_ids_file:
        for num, pubmed_id in enumerate(pubmed_ids_file):
            to_fetch = pubmed_id.strip()
            print(to_fetch)
            sys.stdout.write("%03d %s\r" % (num, to_fetch))
            sys.stdout.flush()
            write_citations(citations_file, to_fetch)
