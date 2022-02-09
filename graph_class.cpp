#include <bits/stdc++.h>
using namespace std;
#define INF INT_MAX
#define NIL 0

// o constanta pe care o folosesc in anumite probleme pentru a da dimensiuni array-urilor / matricilor
const int nmax = INT_MAX / 2;

class Edge{
public:
    int i, j, cost;
    Edge(int _i, int _j, int _cost) : i(_i), j(_j), cost(_cost){}

    friend bool operator<(const Edge& e1, const Edge& e2) {
          return e1.cost < e2.cost;
    }
    // util in ordonarea dupa cost
};
// o structura de tip muchie, utila in problema APM cand, aplicand algoritmul lui Kruskall
// avem nevoie sa sortam muchiile crescator dupa cost

// structura care tine minte parintele si rangul in contextul padurilor de multimi disjuncte
struct parentRank {
    int parent;
    int rank;
};

// clasa de paduri de multimi disjuncte, contine un vector cu informatii (parinte si rang) despre elemente
// si metodele pentru reuniune si gasirea reprezentantului
class Disjoint {
    vector<parentRank> info;
public:
    Disjoint(int);
    int findRep(int);
    void reunion(int, int);
};

Disjoint::Disjoint(int n) {
    info.resize(n + 1);
    for(int i = 1; i <= n; ++i) {
        info[i].parent = i;
        info[i].rank = 0;
    }

}

int Disjoint::findRep(int x) {
    if(x == info[x].parent)
        return x;
    return info[x].parent = findRep(info[x].parent);
}

void Disjoint::reunion(int x, int y) {
    int repX = this->findRep(x);
    int repY = this->findRep(y);
    if(info[repX].rank > info[repY].rank)
        info[repX].parent = repY;
    else
        if(info[repX].rank < info[repY].rank)
            info[repY].parent = repX;
        else {
            info[repX].rank++;
            info[repY].parent = repX;
        }
}



// in principiu util doar pt HavelHakimi
void countSort(vector<int>& input)
{
    map<int, int> freq;
    for (int x: input) {
        freq[x]++;
    }
    int i = 0;
    for (auto p: freq)
    {
        while (p.second--) {
            input[i++] = p.first;
        }
    }
}

bool HavelHakimi(vector<int> d){
        int n = d.size();
        int sum = 0;
        for(auto degree: d) {
            if(degree > n - 1)
                return false;
            sum += degree;
        }

        while(d.size()){
            countSort(d);
            int biggest = d[0];
            d.erase(d.begin());
            for(int i = 0; i < biggest; ++i)
                {
                    --d[i];
                    if(d[i] < 0){
                        return false;
                    }
                }
            }
        return true;
    }

// gaseste si elimina minimul dintr-un min heap, eu il folosesc la Dijkstra
int extractMin(priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>>& pq){
    int temp = pq.top().second;
    pq.pop();
    return temp;
}

class Graph {
    // numarul de noduri
    int V;
    // numarul de muchii
    int E;
    vector<vector<pair<int, int>>> adj;
    // lista de adiacenta a nodului 'n' e formata din perechi de tipul
    // {vecin, cost}, unde cost = costul muchiei {n, vecin}.
    // daca graful n-are costuri pe muchii cost = 0, deoarece un graf
    // fara costuri pe muchii este un caz particular de graf cu costuri pe muhcii, unde toate costurile sunt = 0
    void DFSUtil(int, vector<bool>&, vector<int>&);
    // metoda ajutatoare pentru DFS, primeste ca parametrii nodul de unde incepe parcurgerea, vectorul care tine
    // minte nodurile vizitate / nevizitate (pe care il modifica) si insula curenta (pe care o formeaza, fiind de asemenea
    // transmisa prin referinta)
    void dfBCC(int, vector<bool>&, vector<int>&, vector<int>&, vector<vector<int>>&, stack<pair<int, int>>&);
    // df pentru determinarea componentelor biconexe primeste ca argumente nodul de start, vectorul
    // care spune daca noduri sunt vizitate / nevizitate, cei doi vectori care contin nivelul si nivelul minim
    // accesibil pentru fiecare nod, lista componentelor biconexe transmisa prin referinta pe care o actualizeaza
    // si stiva de muchii utila pentru determinarea componentei biconexe curente
    bool bfsForMaxFlow(vector<int>&, vector<vector<int>>&, vector<vector<int>>&);
    // bfs ul pentru augmenting path, primeste ca parametri vectorul de tati, matricea cu capacitatile si functia
    bool bfsMatching(int N, int M, vector<int>&, vector<int>&, vector<int>&);
    // bfs care returneaza true daca exista augmenting path; primeste ca argumente pe N si M,
    // vectorul de distante, si vectorii care realizeaza cuplajele (R si L)
    bool dfsMatching(int crt, int N, int M, vector<int>&, vector<int>&, vector<int>&);
    // dfs care vede daca exista augmenting path incepand din crt si reconstruieste, primeste ca argumente
    // nodul de start, N si M, vectorul de distante, si vectorii care realizeaza cuplajele (R si L)
    void dfEuler(int, vector<int>&, vector<vector<int>>&, vector<bool>&, vector<int>&, vector<int>&);
    // df - ul utilitar pentru gasirea ciclului eulerian, primeste ca argumente nodul de start
    // ciclul pe care il si modifica, matricea de adiacenta cu muchiile, vectorul care ne spune ce
    // muchii au fost folosite, si cei doi vectori care ne ajuta sa tinem minte capetele muchiilor

public:
    Graph(int, int, vector<vector<pair<int, int>>>);
    // constructorul care initializeaza numarul de noduri si muchii
    // si lista de adiacenta
    vector<int> DFS(int, vector<bool>&);
    // metoda care returneaza (intr-un vector) ordinea parcurgerii dfs incepand din nodul primit ca prim parametru.
    // de asemenea primeste si un parametru de tip vector de bool (true / false) care modifica nodurile vizitate
    // la parcurgerea curenta
    int connectedComponents();
    // metoda care - mi returneaza numarul de componente conexe dintr - un graf neorientat
    pair<vector<int>, vector<int>> bfs(int src);
    // returnez ordinea parcurgerii bfs, dar si
    // un vector de distante minime. Amandoua sunt relevante
    // si specifice parcurgerii in latime
    void dfForTopoSort(int, vector<bool>&, stack<int>&);
    // metoda df pentru sortare topologica, primeste ca argumente nodul de unde incepe aceasta parcurgere,
    // vectorul de care tine minte pentru noduri daca sunt vizitate / nevizitate pe care il si actualizeaza
    // si o stiva (pe care o construieste, fiind transmisa prin referinta) ce va contine nodurile in ordinea
    // inversa a timpilor de finalizare
    vector<int> topoSort();
    // metoda care - mi returneaza un vector continand nodurile intr-o ordine de sortare topologica
    void DFKosaraju(vector<vector<pair<int, int>>>, vector<vector<int>>&, int, vector<bool>&);
    // DF util pentru Kosaraju, primeste lista de adiacenta a grafului transpus, lista componentelor conexe pe care o actualizeaza, fiind
    // transmisa prin referinta, nodul de start al DF - ului si vectorul care contine informatii de tip vizitat / nevizitat despre noduri
    // de asemenea transmis prin referinta pentru ca - l modifica
    vector<vector<int>> Kosaraju();
    // algoritmul lui Kosaraju, imi returneaza lista componentelor conexe, adica o lista de liste de noduri
    vector<vector<int>> biconnectedComponents();
    // metoda ce returneaza componentele biconexe, adica o lista de liste de noduri
    static vector<vector<int>> RoyFloyd(vector<vector<int>>&);
    // am facut-o statica deoarece nu construiec efectiv graful, ci lucrez direct pe matricea ponderilor
    vector<int> Dijkstra();
    // algoritmul lui Dijkstra, returnez un vector al distantelor minime
    vector<int> BellmanFord();
    // algoritmul lui Bellman-Ford, returnez un vector al distantelor minime
    pair<vector<Edge>, int> mstKruskall();
    // algoritmul lui kruskall, returnez o lista de muchii ce formeaza arborele partial de cost minim
    // cat si costul acestuia
    int maxFlow();
    // returneaza fluxul maxim
    int diameter();
    // imi returneaza diametrul grafului (DOAR in cazul in care acesta e arbore)
    vector<int> EulerCircuit();
    // returneaza ciclul eulerian, ne folosim de rezultatul care spune ca un graf eulerian
    // poate fi partitionat in cicluri disjuncte si apoi aplicam algoritmul lui Hierholzer
    int minCostHamiltonianCircuit();
    // returneaza costul minim al unui ciclu hamiltonian
    vector<pair<int, int>> hopcroftKarp(int N, int M);
    // algoritmul lui hopcroftKarp, imi returneaza o lista de perechi
    // reprezentnd muchiile din cuplaj. Ca parametri primeste pe N si M,
    // cardinalele multimilor din enunt
};

Graph::Graph(int _V, int _E, vector<vector<pair<int, int>>> _adj) : V (_V), E(_E), adj(_adj) {
}

void Graph::DFSUtil(int v, vector<bool>& vis, vector<int>& island){
    for(auto i : adj[v]){
        int ngb = i.first;
        if(!(vis[ngb])) {
            island.push_back(ngb);
            vis[ngb] = true;
            DFSUtil(ngb, vis, island);
        }
    }
}
vector<int> Graph::DFS(int src, vector<bool>& vis) {
    // prin "island" returnez 'insula' obtinuta din parcurgerea dfs din nodul curent, adica in principiu
    // nodurile din componenta sa conexa (in grafuri neorientate e mai intuitiv) si in ordinea dfs
    vector<int> island;
    DFSUtil(src, vis, island);
    return island;
}

int Graph::connectedComponents() {
    int nrIslands = 0;
    vector<bool> vis;
    vis.resize(V + 1, false);
    for(int i = 1; i <= V; ++i) {
        if(!vis[i]){
            ++nrIslands;
            DFS(i, vis);
        }
    }
    return nrIslands;
}

pair<vector<int>, vector<int>> Graph::bfs(int src) {
     pair<vector<int>, vector<int>> toReturn;
     queue<int> q;
     q.push(src);
     vector<int> bfsOrder;
     vector<int> dist;
     dist.resize(V + 1, -1);
     q.push(src);
     dist[src] = 0;
     while(!(q.empty())){
         int dad = q.front();
         bfsOrder.push_back(dad);
         q.pop();
         for(auto i : adj[dad]) {
             int ngb = i.first;
             if(dist[ngb] == - 1){
                  dist[ngb] = dist[dad] + 1;
                  q.push(ngb);
            }
        }
    }
    toReturn.first = dist;
    toReturn.second = bfsOrder;
    return toReturn;
}

void Graph::dfForTopoSort(int src, vector<bool>& vis, stack<int>& st) {
    for(auto i: adj[src]) {
        int ngb = i.first;
        if(vis[ngb] == false) {
            vis[ngb] = true;
            dfForTopoSort(ngb, vis, st);
        }
    }
    st.push(src);
}

vector<int> Graph::topoSort(){
    vector<bool> vis;
    stack<int> st;
    vis.resize(V + 1, false);
    for(int i = 1; i <= V; ++i)
         if(!vis[i]) {
            vis[i] = true;
            dfForTopoSort(i, vis, st);
        }
    vector<int> topoSorted;
    while(st.size()) {
        topoSorted.push_back(st.top());
        st.pop();
    }
    return topoSorted;
}


void Graph::DFKosaraju(vector<vector<pair<int, int>>> adjT, vector<vector<int>>& sol, int node, vector<bool>& visT){
    sol[sol.size() - 1].push_back(node);
    visT[node] = true;
    for(auto i: adjT[node]) {
            int ngb = i.first;
            if(visT[ngb] == false)
            {
                DFKosaraju(adjT, sol, ngb, visT);
            }
        }
}

vector<vector<int>> Graph::Kosaraju() {
    vector<vector<pair<int, int>>> adjT;
    adjT.resize(V + 1);
    for(int i = 1; i <= V; ++i)
        for(auto ngb : adj[i])
            adjT[ngb.first].push_back(make_pair(i, 0));

    vector<bool> visT;
    vector<bool> vis;

    visT.resize(V + 1, false);
    vis.resize(V + 1, false);

    stack<int> st;
    vector<vector<int>> stronglyCC;
    for(int i = 1; i <= V; ++i)
        if(!vis[i])
            {
                vis[i] = true;
                dfForTopoSort(i, vis, st);
                // construim stiva specifica sortarii topologice
                // adica cu nodurile in ordinea inversa a timpilor de finalizare
                // e un pic impropriu sa vorbim despre sortare topologica si componente tare
                // conexe in aceeasi problema deoarece exista unei componente tare conexe
                // implica existenta unui ciclu, si stim ca grafurile care contin cicluri
                // nu admit sortare topologica, dar alegerea nodurilor in ordinea inversa a
                // timpilor de finalizare este valabila pentru ambele probleme

            }

     while(st.size())
        {
            stronglyCC.push_back(vector<int>());
            while(st.size() && visT[st.top()] == true)
                st.pop();
            if(st.size())
            {
                int crt = st.top();
                DFKosaraju(adjT, stronglyCC, crt, visT);
            }
        }
    return stronglyCC;
}


vector<vector<int>> Graph::biconnectedComponents() {
    vector<bool> vis;
    vis.resize(V + 1, false);
    vector<int> level;
    level.resize(V + 1);
    vector<int> minLevel;
    minLevel.resize(V + 1);
    vector<vector<int>> BCC;
    stack<pair<int, int>> edges;
    level[1] = 0;
    dfBCC(1, vis, level, minLevel, BCC, edges);

    return BCC;
}

void Graph::dfBCC(int crt, vector<bool>& vis, vector<int>& level, vector<int>& minLevel, vector<vector<int>>& BCC, stack<pair<int, int>>& edges) {
    vis[crt] = true;
    minLevel[crt] = level[crt];
    for(auto i: adj[crt]) {
        int ngb = i.first;
        if(vis[ngb] == false) {
            level[ngb] = level[crt] + 1;
            edges.push(make_pair(crt, ngb));
            dfBCC(ngb, vis, level, minLevel, BCC, edges);
            if(minLevel[ngb] >= level[crt]) {
                BCC.push_back(vector<int>());
                BCC[BCC.size() - 1].push_back(crt);
                while((edges.top().first == crt && edges.top().second == ngb) == false) {
                    BCC[BCC.size() - 1].push_back(edges.top().second);
                    edges.pop();
                }
                BCC[BCC.size() - 1].push_back(ngb);
                edges.pop();
            }
                minLevel[crt] = min(minLevel[crt], minLevel[ngb]);
        }
        else if (level[crt] - level[ngb] >= 2)
            minLevel[crt] = min(minLevel[crt], level[ngb]);
    }
}

vector<int> Graph::Dijkstra(){
    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq;
    // min heap ce tine perechi de tipul {cost, nod}
    vector<int> minDistances;
    minDistances.resize(V + 1);
    minDistances[1] = 0;
    for(int i = 2; i <= V; ++i)
        minDistances[i] = INF;
    pq.push(make_pair(0, 1));
    vector<bool> extractedBefore;
    extractedBefore.resize(V + 1, false);

    while(!(pq.empty())){
        int crt = extractMin(pq);
        if(extractedBefore[crt] == true)
            continue;
        extractedBefore[crt] = true;
        for(auto i: adj[crt]){
            int ngb = i.first;
            int cost = i.second;
            if(cost + minDistances[crt] < minDistances[ngb])
                {
                    minDistances[ngb] = cost + minDistances[crt];
                    pq.push(make_pair(minDistances[ngb], ngb));
                }
        }
    }
    for(int i = 2; i <= V; ++i)
        if(minDistances[i] == INF)
            minDistances[i] = 0;

    return minDistances;
}

vector<int> Graph::BellmanFord() {
    queue<int> q;
    vector<bool> inQ;
    inQ.resize(V + 1, false);
    vector<int> minDistances;
    minDistances.resize(V + 1, INF);
    vector<int> cnt;
    cnt.resize(V + 1, 0);
    minDistances[1] = 0;
    q.push(1);
    inQ[1] = true;
    while(!(q.empty())) {
        int crt = q.front();
        q.pop();
        inQ[crt] = false;
        for(auto i: adj[crt]){
            int ngb = i.first;
            int cost = i.second;
            if(minDistances[crt] + cost < minDistances[ngb]){
                minDistances[ngb] = minDistances[crt] + cost;
                if(inQ[ngb] == false)
                    {   q.push(ngb);
                        inQ[ngb] = true;
                        ++cnt[ngb];
                        if(cnt[ngb] > V)
                            {
                                // dau o dimensiune imposibila ca sa fie clar ca e situatie speciala (CICLU NEGATIV)
                                minDistances.resize(V + 2, -1);
                                return minDistances;
                            }
                    }
            }
        }
    }
    return minDistances;
}

pair<vector<Edge>, int> Graph::mstKruskall() {
    vector<Edge> mst;
    vector<Edge> edges;
    for(int i = 1; i <= V; ++i) {
        int src = i;
        for(int j = 0; j < adj[i].size(); ++j) {
            int dest = adj[i][j].first;
            int cost = adj[i][j].second;
            if(src > dest) {
                Edge tempEdge = Edge(src, dest, cost);
                edges.push_back(tempEdge);
            }
        }
    }
     int total = 0;
     sort(edges.begin(), edges.end());
     Disjoint d(V);

     int cnt = 0;
     for(int i = 0; i < edges.size() && cnt <= E - 1; ++i) {
            Edge tempEdge = edges[i];
            if(d.findRep(edges[i].i) != d.findRep(edges[i].j)) {
                ++cnt;
                mst.push_back(tempEdge);
                d.reunion(tempEdge.i, tempEdge.j);
                total += tempEdge.cost;
            }
        }
    return make_pair(mst, total);
}

bool Graph::bfsForMaxFlow(vector<int>& dad, vector<vector<int>>& capacity, vector<vector<int>>& f) {
    vector<bool> vis;
    vis.resize(V + 1, false);
    dad.resize(V + 1, 0);
    queue<int> q;
    q.push(1);
    vis[1] = true;
    while(q.empty() == false) {
        int crt = q.front();
        q.pop();
        for(auto i: adj[crt]) {
            int ngb = i.first;
            if(vis[ngb] == false && capacity[crt][ngb] > f[crt][ngb]) {
                q.push(ngb);
                vis[ngb] = true;
                dad[ngb] = crt;
                if(ngb == V)
                    return true;
            }
        }
    }
    return false;
}

int Graph::maxFlow() {
    vector<vector<int>> capacity;
    vector<vector<int>> f;

    capacity.resize(V + 1);
    for(int i = 0; i <= V; ++i)
        capacity[i].resize(V + 1, 0);

    f.resize(V + 1);
    for(int i = 0; i <= V; ++i)
        f[i].resize(V + 1, 0);

    vector<int> dad;
    dad.resize(V + 1, 0);

    int flow = 0;

    for(int i = 1; i <= V; ++i)
        for(auto j: adj[i]) {
            int ngb = j.first;
            int cap = j.second;
            capacity[i][ngb] = cap;
        }

    while(bfsForMaxFlow(dad, capacity, f)) {
        int i = INT_MAX;
        int crt = V;

        while(crt != 1) {
            i = min(i, capacity[dad[crt]][crt] - f[dad[crt]][crt]);
            crt = dad[crt];
        }

        crt = V;
        while(crt != 1) {
            f[dad[crt]][crt] += i;
            f[crt][dad[crt]] -= i;
            crt = dad[crt];
        }
        flow += i;
    }
    return flow;
}

vector<int> Graph::EulerCircuit() {
    vector<int> L;
    vector<int> R;
    vector<bool> used;
    used.resize(E + 1, false);
    vector<vector<int>> adjEdges;
    adjEdges.resize(V + 1);
    L.resize(E + 1);
    R.resize(E + 1);
    int cnt = 0;
    for(int i = 1; i <= V; ++i)
        for(auto j : adj[i]) {
            int ngb = j.first;
            adjEdges[i].push_back(++cnt);
            adjEdges[ngb].push_back(cnt);
            L[cnt] = i;
            R[cnt] = ngb;
        }

    for(int i = 1; i <= V; ++i) {
        if(adjEdges[i].size() % 2) {
            vector<int> err;
            err.resize(V + 2, -1);
            return err;
        }
    }
    /*for(int i = 1; i <= V; ++i) {
        cout << i << ": ";
        for(int j = 0; j < adjEdges[i].size(); ++j)
            cout << adjEdges[i][j] << ' ';
        cout << '\n';
    }*/

    vector<int> circuit;
    dfEuler(1, circuit, adjEdges, used, L, R);
    circuit.pop_back();
    return circuit;
}


void Graph::dfEuler(int crt, vector<int>& circuit, vector<vector<int>>& adjEdges, vector<bool>& used, vector<int>& L, vector<int>& R) {
    while(adjEdges[crt].size()) {
        int e = adjEdges[crt].back();
        adjEdges[crt].pop_back();
        if(used[e] == false) {
            used[e] = true;
            int other = crt ^ L[e] ^ R[e];
            dfEuler(other, circuit, adjEdges, used, L, R);
        }
    }
    circuit.push_back(crt);
}

int Graph::diameter() {
    vector<int> dst = (this->bfs(1)).first;
    int maxDist = -1;
    int maxDistVertex = 0;
    for(int i = 1; i <= V; ++i)
        if(dst[i] > maxDist) {
            maxDist = dst[i];
            maxDistVertex = i;
        }
    // caut cel mai departat nod de nodul 1, iar apoi cel mai departat nod de acesta din urma.
    // cele doua formeaza un diametru al arborelui
    dst = (this->bfs(maxDistVertex)).first;
    for(int i = 1; i <= V; ++i)
        if(dst[i] > maxDist) {
            maxDist = dst[i];
            maxDistVertex = i;
        }
    return maxDist + 1;
}


vector<vector<int>> Graph::RoyFloyd(vector<vector<int>>& costMatrix) {
        int N = costMatrix.size() - 1;
        for(int k = 0; k < N; ++k)
        // pana la pasul asta, am calculat distanta minima la i la j folosindu-ma de nodurile 0..k - 1 (DACA EXISTA!)
            for(int i = 0; i < N; ++i)
                for(int j = 0; j < N; ++j)
                    if((costMatrix[i][j] > costMatrix[i][k] + costMatrix[k][j] || (costMatrix[i][j] == 0 && i != j)) && (costMatrix[i][k] * costMatrix[k][j] != 0))
                        costMatrix[i][j] = costMatrix[i][k] + costMatrix[k][j];
        return costMatrix;
}


int Graph::minCostHamiltonianCircuit() {
    // acestea doua trebuie puse globale la problema asta,
    // pe compilatorul meu si cel de pe infoarena e un overflow
    // daca le las locale
    int minCost[1 << nmax + 5][nmax], costMatrix[nmax][nmax];
    for (int i = 0; i < V; ++i)
        for (int j = 0; j < V; ++j)
            costMatrix[i][j] = INF / 2;

    for(int i = 0; i < V; ++i) {
        for(auto j: adj[i]){
            int ngb = j.first;
            int cost = j.second;
            costMatrix[i][ngb] = cost;
        }
    }

    int pow = 1 << V;
    for (int i = 0; i < pow ;++i)
        for (int j = 0; j < V; ++j)
            minCost[i][j]= INF / 2;

    minCost[1][0]=0;

    for (int i = 0; i < pow; ++i)
          for (int j = 0; j < V; ++j)
              if ((i & (1 << j)))
                  for (int intermediar = 0; intermediar < V; ++intermediar)
                       if (intermediar != j && (i & (1 << intermediar)))
                              minCost[i][j] = min(minCost[i ^ (1 << j)][intermediar] + costMatrix[intermediar][j], minCost[i][j]);

    int minimum = minCost[pow - 1][0];

    for (int i = 1; i < V; ++i) {
          if(minCost[pow - 1][i] + costMatrix[i][0] < minimum)
              minimum = minCost[pow - 1][i] + costMatrix[i][0];
    }


   return minimum;
}



bool Graph::dfsMatching(int crt, int N, int M, vector<int>& dist, vector<int>& L, vector<int>& R) {
    if(crt != NIL) {
        for(auto i: adj[crt]) {
            int ngb = i.first;
            if(dist[R[ngb]] == dist[crt] + 1) {
                if(dfsMatching(R[ngb], N, M, dist, L, R)) {
                    R[ngb] = crt;
                    L[crt] = ngb;
                    return true;
                }
            }
        }
        dist[crt] = INF;
        return false;
    }
    return true;
}

bool Graph::bfsMatching(int N, int M, vector<int>& dist, vector<int>& L, vector<int>& R) {
    queue<int> q;
    for(int i = 1; i <= N; ++i) {
        if(L[i] == NIL) {
            dist[i] = 0;
            q.push(i);
        }
        else
            dist[i] = INF;
    }

    dist[NIL] = INF;

    while(q.empty() == false) {
        int crt = q.front();
        q.pop();
        if(dist[crt] < dist[NIL]) {
            for(auto i: adj[crt]) {
                int ngb = i.first;

                if(dist[R[ngb]] == INF) {
                    dist[R[ngb]] = dist[crt] + 1;
                    q.push(R[ngb]);
                }
            }
        }
    }
    return (dist[NIL] != INF);
}


vector<pair<int, int>> Graph::hopcroftKarp(int N, int M) {
    vector<int> L, R, dist;
    L.resize(N + 1);
    R.resize(M + 1);
    dist.resize(N + 1);

    for(int i = 0; i <= N; ++i)
        L[i] = NIL;
    for(int i = 0; i <= M; ++i)
        R[i] = NIL;

    int matching = 0;
    while(bfsMatching(N, M, dist, L, R)) {
        for(int i = 1; i <= N; ++i)
            if(L[i] == NIL && dfsMatching(i, N, M, dist, L, R))
                ++matching;
    }
    vector<pair<int, int>> matches;
    for(int i = 1; i <= N; ++i) {
        if(L[i] != NIL)
            matches.push_back(make_pair(i, L[i]));
    }
    return matches;
}
int main()
{

    // problema dfs pe pe infoarena
    // https://www.infoarena.ro/problema/dfs
    ifstream fin("dfs.in");
    ofstream fout("dfs.out");
    int v, e;
    fin >> v >> e;
    vector<vector<pair<int, int>>> adj;
    adj.resize(v + 1);
    for(int i = 1; i <= e; ++i) {
        int src, dst;
        fin >> src >> dst;
        adj[src].push_back(make_pair(dst, 0));
        adj[dst].push_back(make_pair(src, 0));
    }
    Graph g(v, e, adj);
    fout << g.connectedComponents();


    return 0;
}
