from collections import defaultdict
import networkx as nx

def find_cliques(G):
    if len(G) == 0:
        return

    adj = {u: {v for v in G[u] if v != u} for u in G}
    Q = [None]

    subg = set(G)
    cand = set(G)
    u = max(subg, key=lambda u: len(cand & adj[u]))
    ext_u = cand - adj[u]
    stack = []

    try:
        while True:
            if ext_u:
                q = ext_u.pop()
                cand.remove(q)
                Q[-1] = q
                adj_q = adj[q]
                subg_q = subg & adj_q
                if not subg_q:
                    yield Q[:]
                else:
                    cand_q = cand & adj_q
                    if cand_q:
                        stack.append((subg, cand, ext_u))
                        Q.append(None)
                        subg = subg_q
                        cand = cand_q
                        u = max(subg, key=lambda u: len(cand & adj[u]))
                        ext_u = cand - adj[u]
            else:
                Q.pop()
                subg, cand, ext_u = stack.pop()
    except IndexError:
        pass


def get_adjacent_cliques(clique, membership_dict):
    adjacent_cliques = set()
    for n in clique:
        for adj_clique in membership_dict[n]:
            if clique != adj_clique:
                adjacent_cliques.add(adj_clique)
    return adjacent_cliques


def get_fast_percolated_cliques(G, k):
    print("\n\n Starting Clique Finding for size " + str(k))
    cliques = [frozenset(c) for c in find_cliques(G.list) if len(c) >= k]

    print("Cliques found.")
    nodesToCliquesDict = defaultdict(list)
    for clique in cliques:
        for node in clique:
            nodesToCliquesDict[node].append(clique)

    print("NodesToCliques Map built. ")
    cliquesToComponents = dict()
    currentComponent = 0

    cliquesProcessed = 0
    for clique in cliques:
        cliquesProcessed += 1
        if cliquesProcessed % 1000 == 0:
            print("Total cliques processed: ", str(cliquesProcessed))

        if not clique in cliquesToComponents:
            currentComponent += 1
            cliquesToComponents[clique] = currentComponent
            frontier = set()
            frontier.add(clique)
            componentCliquesProcessed = 0

            while len(frontier) > 0:
                currentClique = frontier.pop()
                componentCliquesProcessed += 1
                if componentCliquesProcessed % 1000 == 0:
                    print("Component cliques processed: ", str(componentCliquesProcessed))
                    print("Size of frontier: ", len(frontier))

                for neighbour in get_adjacent_cliques(currentClique, nodesToCliquesDict):
                    if len(currentClique.intersection(neighbour)) >= (k - 1):
                        cliquesToComponents[neighbour] = currentComponent
                        frontier.add(neighbour)
                        for node in neighbour:
                            nodesToCliquesDict[node].remove(neighbour)

    print("CliqueGraphComponent Built")
    componentToNodes = defaultdict(set)
    for clique in cliquesToComponents:
        componentCliqueIn = cliquesToComponents[clique]
        componentToNodes[componentCliqueIn].update(clique)

    print("Node Components Assigned. ")
    return componentToNodes.values()

def CPM(G, k):
    k = 5
    trainComm = {}
    commNum = 0
    for c in get_fast_percolated_cliques(G, k):
        list_c = list(c)
        trainComm[commNum] = list_c
        commNum = commNum + 1
    return trainComm
