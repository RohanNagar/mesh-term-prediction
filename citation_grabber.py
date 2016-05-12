import requests
import requests_cache

requests_cache.install_cache('citations_cache')

from bs4 import BeautifulSoup

from Bio import Entrez

def get_citations(pmcid):
    articles_url = 'http://www.ncbi.nlm.nih.gov/pmc/articles/%s' % pmcid
    headers = {
        'User-Agent': (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/39.0.2171.95 Safari/537.36"
        )
    }
    page = requests.get(articles_url, headers=headers)
    soup = BeautifulSoup(page.content, "lxml")
    pubmed_article_urls = [
        span.a['href'] for span in soup.findAll("span", {"class": "nowrap ref pubmed"})
    ]
    return [url.replace(r'/pubmed/', '') for url in pubmed_article_urls]


# If we access the DB too much they will send an email before cutting us off.
# Try to access in batches (not sure if it actually helps or not).
Entrez.email = "rohan.nagar@utexas.edu"

def get_pmc_id(pubmed_id):
    pubmed_record_url = (
        'http://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?'
        'db=pubmed&id=%s&retmode=xml'
    ) % pubmed_id
    headers = {
        'User-Agent': (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/39.0.2171.95 Safari/537.36"
        )
    }
    page = requests.get(pubmed_record_url, headers=headers)
    soup = BeautifulSoup(page.content, "lxml")
    return soup.find("articleid", {"idtype": "pmc"}).get_text()

def get_article_info(pubmed_ids):
    handle = Entrez.efetch(db="pubmed", id=pubmed_ids, rettype="medline", retmode="xml")
    records = Entrez.read(handle)

    articles = []
    for record in records:
        article_info = dict()
        
        if 'MedlineCitation' not in record:
            continue

        article_info['id'] = record['MedlineCitation']['PMID']
        article_info['title'] = record['MedlineCitation']['Article']['ArticleTitle']

        # Get the abstract. It's formatted weird in the dict
        if 'Abstract' in record['MedlineCitation']['Article']:
            abstract_list = record['MedlineCitation']['Article']['Abstract']['AbstractText']
            article_info['abstract'] = ''.join(abstract_list)
        else:
            article_info['abstract'] = 'n/a'

        headings = []
        # Make sure it has MeSH terms before trying to get them
        if 'MeshHeadingList' in record['MedlineCitation']:
            for heading in record['MedlineCitation']['MeshHeadingList']:
                term = heading['DescriptorName']
                # If the heading is a main heading
                if heading['DescriptorName'].attributes['MajorTopicYN'] == 'Y':
                    term += '*'
                headings.append(term)
        article_info['terms'] = headings
        articles.append(article_info)

    handle.close()
    return articles

def write_to_file(f, original_id, citation_articles):
    f.write('{}|c|{}\n'.format(original_id, len(citation_articles)))
    for article in citation_articles:
        f.write('{}|orig|{}\n'.format(article['id'], original_id))
        f.write('{}|a|{}\n'.format(article['id'], article['abstract']))
        f.write('{}|t|{}\n'.format(article['id'], article['title']))
        terms = ""
        for term in article['terms']:
            terms += '{}|'.format(term)
        f.write('{}|m|{}\n'.format(article['id'], terms))

def write_citations(f, pubmed_id):
    try:
        pmc_id = get_pmc_id(pubmed_id)
        citations = get_citations(pmc_id)
        citation_articles = get_article_info(citations)
        write_to_file(f, pubmed_id, citation_articles) 
    except AttributeError as e:
        print(pubmed_id, e)
    except RuntimeError as e:
        print(str(e))
        if(str(e) == 'Supplied id parameter is empty.'): pass
        else: raise e
    

if __name__ == '__main__':
    import sys
    write_citations(sys.stdout, 15064396)
    #get_pmc_id(15064396)
