
def load_valid_pmids():
    pmids = []
    with open('ourdat/citations_TEST.TiAbMe', 'r') as f:
        try:
            while(True):
                line = next(f).strip()
                pubmed_id, _, num_cited = line.split('|')
                pmids.append(pubmed_id)
                
                num_cited = int(num_cited)
                for _ in range(num_cited):
                    _ = next(f)
                    _ = next(f)
                    _ = next(f)
                    _ = next(f)

        except(StopIteration):
            pass

    return pmids

def make_test_documents(pmids, outfile):
    with open('paperdat/NLM2007/NLM2007.TiAbMe', 'r') as f:
        for line in f:
            record = line.split('|')
            if record[0] not in pmids: continue

            outfile.write('{}'.format(line))

    with open('paperdat/L1000/L1000.TiAbMe', 'r') as f:
        for line in f:
            record = line.split('|')
            if record[0] not in pmids: continue

            outfile.write('{}'.format(line))

def make_test_neighbor_score(pmids, outfile):
    neighbors = []

    with open('paperdat/NLM2007/NLM2007_50neighbors.score', 'r') as f:
        for line in f:
            record = line.split()
            if record[0] not in pmids: continue

            neighbors.append(record[1])
            outfile.write('{}'.format(line))

    with open('paperdat/L1000/L1000_50neighbors.score', 'r') as f:
        for line in f:
            record = line.split()
            if record[0] not in pmids: continue

            neighbors.append(record[1])
            outfile.write('{}'.format(line))

    return neighbors

def test_neighbor_score(f):
    cur = None
    count = 0
    for line in f:
        record = line.split()
        if record[0] == cur: continue

        count += 1
        cur = record[0]

    return count

def make_test_neighbors(pmids, outfile):
    with open('paperdat/NLM2007/NLM2007_50neighbors.TiAbMe', 'r') as f:
        for line in f:
            record = line.split('|')
            if record[0] not in pmids: continue

            outfile.write('{}'.format(line))

    with open('paperdat/L1000/L1000_50neighbors.TiAbMe', 'r') as f:
        for line in f:
            record = line.split('|')
            if record[0] not in pmids: continue

            outfile.write('{}'.format(line))

def test_neighbors(f):
    cur = None
    count = 0
    curCount = 0
    types = []
    for line in f:
        record = line.split('|')
        if record[0] != cur:
            count += 1
            cur = record[0]
            print('CUR COUNT: {} TYPES: {}'.format(curCount, types))
            types.clear()
            curCount = 0

        types.append(record[1])
        curCount += 1

    return count


pmids = load_valid_pmids()

with open('paperdat/TEST/TEST.TiAbMe', 'w') as f:
    make_test_documents(pmids, f)

with open('paperdat/TEST/TEST_50neighbors.score', 'w') as f:
    neighbors = make_test_neighbor_score(pmids, f)

with open('paperdat/TEST/TEST_50neighbors.score', 'r') as f:
    print(test_neighbor_score(f))

print(neighbors)
print(len(neighbors))

with open('paperdat/TEST/TEST_50neighbors.TiAbMe', 'w') as f:
    make_test_neighbors(neighbors, f)

with open('paperdat/TEST/TEST_50neighbors.TiAbMe', 'r') as f:
    print(test_neighbors(f))
