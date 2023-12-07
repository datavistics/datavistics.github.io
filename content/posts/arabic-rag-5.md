+++
title = 'Arabic RAG 5: VectorDB'
date = 2023-12-06T13:56:26+04:00
author = "Derek Thomas"
draft = false
ShowReadingTime = true
tags = ["Arabic NLP", "Arabic RAG", "Tutorial"]
cover.image = "cover_images/arabic-rag-5.png"
cover.alt = "Photo Credits to DALL·E 3"
+++

# Goal
This is part 5 of 6 in our tutorial on Arabic RAG. I'll be using this blog as a guide, but to actually run this
tutorial, its best that you run
[this notebook](https://huggingface.co/spaces/derek-thomas/arabic-RAG/blob/main/notebooks/05_vector_db.ipynb)
as described in [part 1]({{< ref "arabic-rag-1" >}}).

{{< notice info >}}
In this blog you will learn:
- What are good practices for preparing your data for VectorDBs
- How to use LanceDB for RAG
- How well does LanceDB work for RAG
{{< /notice >}}
 
# Approach
## VectorDB
Why do we even need a VectorDB? When we get our query we need to be able to find relevant documents. This happens through
a distance comparison between the documents and your query. We want to find the nearest documents to the query. This is 
computationally expensive (~2M comparisons), so we use a family of algorithms called Aproximate Nearest Neighbors (ANN) 
to accelerate this. We make a tradeoff to make fewer comparisons but lose performance.

There are a number of aspects of choosing a vector db that might be unique to your situation. Before choosing you should
think through your requirements (this is not exhaustive): 
- HW 
  - Do you have GPUs available?
  - How many replicas do you need?
- Utilization/Scale
  - How many simultaneous requests do you expect?
- Latency requirements
- Ease of design
- On Prem/Cloud solutions

This will drive your decisions. 

## LanceDB
I've been hearing a lot about LanceDB and wanted to check it out. It's newer and may or may not be good for **your** 
use-case. I'm attracted by its fast ingestion, cuda assisted indexing, and portability. It has some drawbacks, it 
doesn't support graph-based indexing like [hnsw](https://towardsdatascience.com/similarity-search-part-4-hierarchical-navigable-small-world-hnsw-2aad4fe87d37)
or [vamana](https://thedataquarry.com/posts/vector-db-3/#vamana) yet and it seems quite volatile!

Learn more about it in this awesome blog: https://thedataquarry.com/posts/embedded-db-3/

You will be blown away on how fast ingestion + indexing is with LanceDB. If this is your first experience with VectorDB 
ingestion, I'm so sorry. Your expectations are probably going to be too high when working with other options.

## Ingestion Strategy
I used the ~100k document `.ndjson` files in sequence to upload. After uploading I index.

## Indexing
The algorithm used is `IVF_PQ`. I ignore the `PQ` part because I want better recall. Recall is important since 
[jais](https://huggingface.co/core42/jais-13b-chat) only has a 2k context window, I can't put my top 10 documents for 
RAG in my prompt. It will be my top 3 (512\*3 + query + instructions ~ 2k). For many use-cases its worth the trade-off 
as you get much faster retrieval with not much performance loss. 

More partitions means faster retrieval but slower indexing. How do I "ignore" `PQ`? I chose 384 sub_vectors to be equal
to my embedding dimension size. 

```tbl.create_index(num_partitions=1024, num_sub_vectors=384, accelerator="cuda")```

Read more about it [here](https://lancedb.github.io/lancedb/ann_indexes/).

# Implementation

```python
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'
```

```python
from pathlib import Path
import json

from tqdm.notebook import tqdm
import lancedb
```

```python
proj_dir = Path.cwd().parent
print(proj_dir)
```
    /home/ec2-user/arabic-wiki

```python
files_in = list((proj_dir / 'data/embedded/').glob('*.ndjson'))
```

## LanceDB
To work with LanceDB we want to create the table before ingesting the first batch. To create a table we need at least 1 
doc.

```python
with open(files_in[0], 'r') as f:
    first_line = f.readline().strip()  # read only the first line
    document = json.loads(first_line)
    document['vector'] = document.pop('embedding')
```

```python
doc = document.copy()
doc['vector'] = doc['vector'][:5] + ['...']
doc
```

    {'content': 'الماء مادةٌ شفافةٌ عديمة اللون والرائحة، وهو المكوّن الأساسي للجداول والبحيرات والبحار والمحيطات وكذلك للسوائل في جميع الكائنات الحيّة، وهو أكثر المركّبات الكيميائيّة انتشاراً على سطح الأرض. يتألّف جزيء الماء من ذرّة أكسجين مركزية ترتبط بها ذرّتا هيدروجين على طرفيها برابطة تساهميّة بحيث تكون صيغته الكيميائية H2O. عند الظروف القياسية من الضغط ودرجة الحرارة يكون الماء سائلاً؛ أمّا الحالة الصلبة فتتشكّل عند نقطة التجمّد، وتدعى بالجليد؛ أمّا الحالة الغازية فتتشكّل عند نقطة الغليان، وتسمّى بخار الماء.\nإنّ الماء هو أساس وجود الحياة على كوكب الأرض، وهو يغطّي 71% من سطحها، وتمثّل مياه البحار والمحيطات أكبر نسبة للماء على الأرض، حيث تبلغ حوالي 96.5%. وتتوزّع النسب الباقية بين المياه الجوفيّة وبين جليد المناطق القطبيّة (1.7% لكليهما)، مع وجود نسبة صغيرة على شكل بخار ماء معلّق في الهواء على هيئة سحاب (غيوم)، وأحياناً أخرى على هيئة ضباب أو ندى، بالإضافة إلى الزخات المطريّة أو الثلجيّة. تبلغ نسبة الماء العذب حوالي 2.5% فقط من الماء الموجود على الأرض، وأغلب هذه الكمّيّة (حوالي 99%) موجودة في الكتل الجليديّة في المناطق القطبيّة، في حين تتواجد 0.3% من الماء العذب في الأنهار والبحيرات وفي الغلاف الجوّي.\nأما في الطبيعة، فتتغيّر حالة الماء بين الحالات الثلاثة للمادة على سطح الأرض باستمرار من خلال ما يعرف باسم الدورة المائيّة (أو دورة الماء)، والتي تتضمّن حدوث تبخّر ونتح (نتح تبخّري) ثم تكثيف فهطول ثم جريان لتصل إلى المصبّ في المسطّحات المائيّة.\n',
     'content_type': 'text',
     'score': None,
     'meta': {'id': '7',
      'revid': '2080427',
      'url': 'https://ar.wikipedia.org/wiki?curid=7',
      'title': 'ماء',
      '_split_id': 0,
      '_split_overlap': [{'doc_id': '725ec671057ef790ad582509a8653584',
        'range': [887, 1347]}]},
     'id_hash_keys': ['content'],
     'id': '109a29bb227b1aaa5b784e972d8e1e3e',
     'vector': [-0.07318115,
      0.087646484,
      0.03274536,
      0.034942627,
      0.097961426,
      '...']}

Here we create the db and the table.



```python
from lancedb.embeddings.registry import EmbeddingFunctionRegistry
from lancedb.embeddings.sentence_transformers import SentenceTransformerEmbeddings

db = lancedb.connect(proj_dir/".lancedb")
tbl = db.create_table('arabic-wiki', [document])
```

For each file we:
- Read the `ndjson` into a list of documents
- Replace 'embedding' with 'vector' to be compatible with LanceDB
- Write the docs to the table

After that we index with a cuda accelerator.

```python
%%time
for file_in in tqdm(files_in, desc='Wiki Files: '):

    tqdm.write(f"Reading documents {str(file_in)}")
    with open(file_in, 'r') as f:
        documents = [json.loads(line) for line in f]
    tqdm.write(f"Read documents")

    for doc in tqdm(documents):
        if 'embedding' in doc:
            doc['vector'] = doc.pop('embedding')
    
    tqdm.write(f"Adding documents {str(file_in)}")
    tbl.add(documents)
    tqdm.write(f"Added documents")
tbl.create_index(
     num_partitions=1024,
     num_sub_vectors=384,
     accelerator="cuda"
)
    
```


    Wiki Files:   0%|          | 0/23 [00:00<?, ?it/s]
    Reading documents /home/ec2-user/arabic-wiki/data/embedded/ar_wiki_1.ndjson
    Read documents
      0%|          | 0/243068 [00:00<?, ?it/s]
      ...
      0%|          | 0/70322 [00:00<?, ?it/s]


    Adding documents /home/ec2-user/arabic-wiki/data/embedded/ar_wiki_23.ndjson
    Added documents

    [2023-10-31T18:47:43Z WARN  lance_linalg::kmeans] KMeans: cluster 108 is empty
    [2023-10-31T18:52:24Z WARN  lance_linalg::kmeans] KMeans: cluster 227 is empty
    [2023-10-31T18:57:24Z WARN  lance_linalg::kmeans] KMeans: cluster 167 is empty
    [2023-10-31T19:14:19Z WARN  lance_linalg::kmeans] KMeans: cluster 160 is empty


    CPU times: user 2h 2min 51s, sys: 1min 38s, total: 2h 4min 30s
    Wall time: 42min 56s


It's crazy how fast it was. 42 minutes to ingest and index >2M documents. When I used the default settings that did 
quantization it only took 11 min. 


## Verification
Let's run a test to make sure it worked!

```python
from sentence_transformers import SentenceTransformer

name="sentence-transformers/paraphrase-multilingual-minilm-l12-v2"
model = SentenceTransformer(name)

# used for both training and querying
def embed_func(batch):
    return [model.encode(sentence) for sentence in batch]
```

```python
query = "What is the capital of China? I think it's Singapore."
query_vector = embed_func([query])[0]
[doc['meta']['title'] for doc in tbl.search(query_vector).limit(10).to_list()]
```
    ['بكين',
     'كونمينغ',
     'نينغشيا',
     'تاي يوان',
     'تشنغتشو',
     'شانغهاي',
     'سنغافورة',
     'دلتا نهر يانغتسي',
     'تشانغتشون',
     'بكين']

## Analysis
Great! Even with the misdirection `"I think it's Singapore"` we still got `'بكين'`! It's useful to do a full analysis 
of your system where you evaluate both your algorithm and retriever implementation. This would involve creating a 
dataset of Queries and corresponding documents and assess the results. Recall at top_k and 
[MRR](https://en.wikipedia.org/wiki/Mean_reciprocal_rank) are useful metrics here, especially with k= the number of 
documents you can put in your context window.

{{< notice tip >}}
Assessing and improving your retrieval capabilities is the most important step in improving your RAG system. This starts
at pre-processing, but the next step is to fine-tune your retriever or improve your ANN.
{{< /notice >}}

# Next Steps
So now we have processed documents in a VectorDB! In the next blogpost I'll show you how to actually turn this into a 
functional RAG system with a custom version of [jais](https://huggingface.co/core42/jais-13b-chat) and a nice looking 
gradio app. 