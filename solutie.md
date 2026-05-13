# Cum am gandit si implementat solutia

Cand am citit cerinta, mi-am dat seama ca problema nu este doar una de search dupa cuvinte cheie. Trebuie sa decid daca o companie chiar respecta intentia utilizatorului.

De exemplu, pentru:

```text
Logistic companies in Romania
```

nu este suficient ca o companie sa mentioneze "logistics". O firma care face software pentru logistica poate fi apropiata semantic, dar nu este neaparat o companie de logistica. O companie relevanta ar trebui sa faca efectiv transport, depozitare, freight forwarding, livrare sau servicii similare.

De aici am tratat problema ca intent qualification, nu doar ca semantic search.

## Ideea de baza

Am vrut sa evit doua extreme.

Prima ar fi fost sa trimit fiecare companie la un LLM si sa intreb daca este match. Aceasta varianta ar fi buna ca acuratete, dar scumpa si lenta. Daca sistemul ar avea zeci de mii de companii, nu ar scala bine.

A doua ar fi fost sa folosesc doar embeddings. Aceasta varianta este rapida si ieftina, dar similaritatea nu inseamna mereu relevanta reala. Pentru un query despre furnizori de ambalaje pentru cosmetice, embeddings pot aduce branduri de cosmetice, desi userul cauta furnizori de packaging.

De aceea am ales un pipeline hibrid:

```text
1. LLM pentru intelegerea query-ului
2. Filtre structurate in Python
3. Embeddings pentru shortlist semantic
4. LLM pentru decizia finala
```

## Pasul 1: intelegerea query-ului

Primul pas este functia:

```python
analyze_query(query)
```

Aici folosesc Claude Haiku ca sa transform query-ul intr-un JSON cu constrangeri structurate.

De exemplu:

```text
Public software companies with more than 1,000 employees
```

poate deveni:

```json
{
  "is_public": true,
  "min_employees": 1000,
  "semantic_description": "Public software companies with more than 1,000 employees."
}
```

Aceasta etapa extrage tara, regiunea, daca firma este publica, revenue, numar de angajati, an de fondare si o descriere semantica a intentiei.

Mi s-a parut important sa separ intelegerea query-ului de evaluarea companiilor. Dupa ce query-ul este structurat, pot aplica reguli simple si controlabile.

## Pasul 2: filtre structurate

Dupa analiza query-ului, aplic filtre in:

```python
passes_filters(company, constraints)
```

Aici verific lucruri clare:

- tara companiei;
- daca este publica;
- revenue;
- numarul de angajati;
- anul fondarii.

Pentru un query ca:

```text
Construction companies in the United States with revenue over $50 million
```

pot elimina direct firmele care nu sunt in SUA sau care au revenue sub 50M, fara sa consum apeluri LLM.

Am ales sa nu elimin automat companiile cu date numerice lipsa. Daca lipseste `employee_count` sau `revenue`, compania poate merge mai departe, pentru ca dataseturile reale sunt incomplete.

## Pasul 3: transformarea companiilor in text

Pentru embeddings si pentru LLM, transform fiecare companie intr-un text compact cu:

```python
company_to_text(company)
```

Textul include numele, locatia, industria, descrierea, business model, target markets, core offerings, employees, revenue, anul fondarii si daca firma este publica.

Aceasta forma este mai usor de folosit decat JSON-ul brut, pentru ca atat modelul de embeddings, cat si LLM-ul lucreaza mai bine cu text natural.

## Pasul 4: embeddings pentru shortlist

Dupa filtre, pot ramane inca multe companii. Pentru ranking semantic folosesc:

```python
SentenceTransformer("all-MiniLM-L6-v2")
```

La inceput calculez embeddings pentru toate companiile:

```python
precompute_embeddings(companies)
```

Pentru fiecare query, compar embedding-ul query-ului cu embeddings companiilor ramase dupa filtrare:

```python
scores = embs @ q_emb
```

Pentru ca embeddings sunt normalizate, produsul scalar este echivalent cu cosine similarity.

Embeddings nu dau verdictul final. Ele doar aleg top 60 candidati care par cei mai apropiati semantic de query. Am ales top 60 ca un compromis intre recall si cost.

## Pasul 5: calificarea finala cu LLM

Ultima etapa este cea in care LLM-ul decide match sau non-match:

```python
llm_qualify(query, ranked, companies, batch_size=10)
```

Trimit candidatii in batch-uri de cate 10, iar LLM-ul returneaza pentru fiecare companie:

```json
{
  "match": true,
  "score": 5,
  "reason": "..."
}
```

In prompt am pus regula ca firma trebuie sa faca efectiv lucrul cerut, nu doar sa fie apropiata de industrie.

De exemplu:

- logistics software nu ar trebui acceptat ca firma de logistica;
- un brand de cosmetice nu ar trebui acceptat ca furnizor de ambalaje;
- o companie care doar opereaza proiecte de energie nu este neaparat producator de echipamente.

## Batch-uri si modele diferite

Batch-urile reduc costul. Pentru top 60 companii, fac aproximativ 6 apeluri LLM in loc de 60.

Le rulez in paralel cu:

```python
ThreadPoolExecutor(max_workers=4)
```

Folosesc Haiku pentru query-uri mai simple, pentru ca este mai ieftin. Pentru query-uri mai grele, cum ar fi packaging suppliers, fintech sau EV battery components, folosesc Sonnet, deoarece necesita mai mult reasoning.

## Rezultatul final

Scriptul afiseaza in terminal modelul folosit, filtrele extrase, cate companii au ramas dupa filtrare, cate au fost evaluate si care sunt match-urile principale.

La final salveaza si:

```text
results.json
```

Acest fisier este mai complet decat output-ul din terminal, pentru ca include si companiile respinse, nu doar match-urile afisate.

## Tradeoff-uri

Solutia este mai ieftina si mai rapida decat LLM per company, dar are limite.

Daca o companie buna nu intra in top 60 dupa embeddings, LLM-ul nu o mai vede. Deci pot exista false negatives.

LLM-ul poate da raspunsuri usor diferite intre rulari, mai ales pentru cazurile borderline.

Datele lipsa pot duce la aproximari. De exemplu, pentru query-uri precum "using Shopify", datasetul nu are mereu informatia necesara.

## Error analysis

Sistemul merge cel mai bine pe query-uri structurate, unde conditiile pot fi verificate direct. Exemple bune sunt query-urile despre companii publice cu peste 1.000 de angajati sau companii farmaceutice in Elvetia. In aceste cazuri, filtrele structurate fac mare parte din munca, iar LLM-ul doar confirma daca industria si intentia sunt corecte.

Merge destul de bine si pe query-uri de supply chain atunci cand datele contin semnale clare. De exemplu, pentru packaging sau componente de baterii EV, daca descrierea mentioneaza explicit ambalaje, cathode materials, electrolytes sau battery components, LLM-ul poate distinge destul de bine intre furnizori si companii care doar sunt apropiate de acea industrie.

Unde sistemul se chinuie mai mult este la query-uri unde informatia necesara nu exista in dataset. Un exemplu este `E-commerce companies using Shopify or similar platforms`. Datasetul nu are un camp de technology stack, deci sistemul poate identifica firme de e-commerce, dar nu poate verifica sigur daca folosesc Shopify.

Un alt exemplu dificil este `Fast-growing fintech companies competing with traditional banks in Europe`. "Fast-growing" nu poate fi derivat corect din date statice daca nu am revenue growth, user growth sau alte semnale temporale. In acest caz, sistemul poate doar aproxima pe baza descrierii.

## Scaling

Pentru datasetul actual, cu cateva sute de companii, solutia ruleaza simplu in memorie. Embeddings sunt precompute la pornire, iar filtrarea se face direct in Python.

Daca sistemul ar trebui sa ruleze pe 100.000 de companii, as schimba cateva lucruri. Filtrele structurate le-as muta intr-o baza de date, de exemplu PostgreSQL sau Elasticsearch, cu indexuri pe tara, revenue, employee count si alte campuri importante.

Embeddings nu le-as mai recalcula la fiecare rulare. Le-as calcula offline si le-as salva intr-un vector database sau intr-un index ANN, cum ar fi Faiss, Qdrant sau Weaviate. Astfel, cautarea vectoriala ar ramane rapida si la volum mare.

Etapa LLM ar ramane limitata la top candidati. Ideea importanta este ca partea scumpa nu creste proportional cu numarul total de companii, ci ramane controlata prin filtrare si ranking.

## Failure modes

Un failure mode important apare cand descrierea companiei este vaga sau prea mult marketing. De exemplu, o companie poate spune ca "transforma industria logisticii", dar sa fie de fapt o companie SaaS, nu o firma de logistica.

Alt caz dificil este geografia. Eu folosesc `country_code` din adresa, dar o companie poate fi inregistrata intr-o tara si sa opereze in alta. Asta poate produce false negatives sau false positives.

Mai exista si riscul ca etapa de embeddings sa rateze candidati buni. Daca o companie relevanta nu ajunge in top 60, LLM-ul nu o mai evalueaza.

In productie, as monitoriza rata de match per query, distributia scorurilor LLM, erorile de parsare JSON, latenta fiecarei etape si as face verificari manuale pe companii cunoscute ca relevante.

## Concluzie

Am incercat sa folosesc fiecare tehnica acolo unde are cel mai mult sens:

```text
filtrele reduc spatiul de cautare,
embeddings aleg candidatii relevanti,
LLM-ul decide intent qualification.
```

Astfel, solutia pastreaza capacitatea de reasoning a LLM-ului, dar reduce costul si timpul prin filtrare, embeddings si batching.
