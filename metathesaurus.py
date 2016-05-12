import requests
import requests_cache

from bs4 import BeautifulSoup

URL = 'https://uts-ws.nlm.nih.gov/rest'

class UMLS():
    def __init__(self, username, password):
        params = {'username': username, 'password': password}
        headers = {
            'Content-type':'application/x-www-form-urlencoded',
            'Accept': 'text/plain',
            'User-Agent':'python',
        }
        with requests_cache.disabled():
            auth_response = requests.post(
                'https://utslogin.nlm.nih.gov/cas/v1/tickets/',
                data=params,
                headers=headers,
            )
            parser = BeautifulSoup(auth_response.content, 'lxml')
            self.ticket_granter = parser.form['action']

    def get_ticket(self):
        with requests_cache.disabled():
            params = {'service': 'http://umlsks.nlm.nih.gov'}
            headers = {
                'Content-type':'application/x-www-form-urlencoded',
                'Accept': 'text/plain',
                'User-Agent':'python',
            }
            ticket_response = requests.post(
                self.ticket_granter, 
                data=params, 
                headers=headers
            )
            return ticket_response.text

    def run_query(self, endpoint, params):
        auth_params = { 'ticket' : self.get_ticket() }
        auth_params.update(params)
        return requests.get(
            URL+endpoint,
            params=auth_params
        )


    def search(self, **kwargs):
        return self.run_query('/search/current', kwargs).json()

    def content(self, cui, **kwargs):
        return self.run_query('/content/current/CUI/%s' % cui, kwargs).json()

    def atoms(self, cui, **kwargs):
        return self.run_query('/content/current/CUI/%s/atoms' % cui, kwargs).json()


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('term')
    args = parser.parse_args()

    umls = UMLS('zuhairp', 'C^Do7DZiVG1#QZ0F')

    term_result = umls.search(string=args.term, sabs='MSH')
    cui = term_result['result']['results'][0]['ui']
    num_atoms = umls.content(cui)['result']['atomCount']
    synonyms_result = umls.atoms(cui, pageSize=num_atoms, ttys='SY')['result']

    print(len(synonyms_result))
    for synonym_record in synonyms_result:
        print(synonym_record['name'], synonym_record['rootSource'])
