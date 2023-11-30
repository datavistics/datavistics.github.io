+++
title = 'Arabic RAG 3: Pre-Processing'
date = 2023-11-30T17:48:34+04:00
author = "Derek Thomas"
draft = false
ShowReadingTime = true
tags = ["Arabic NLP", "Arabic RAG", "Tutorial"]
cover.image = "cover_images/arabic-rag-3.png"
cover.alt = "Photo Credits to DALL·E 3"
+++

# Goal
This is part 3 of 6 in our tutorial on Arabic RAG. I'll be using this blog as a guide, but to actually run this
tutorial, its best that you run
[this notebook](https://huggingface.co/spaces/derek-thomas/arabic-RAG/blob/main/notebooks/03_preprocessing.ipynb)
as described in [part 1]({{< ref "arabic-rag-1" >}}).

In this blog you will learn:
- Chunking Considerations
- How to leverage the very useful [Haystack](https://haystack.deepset.ai) library from [Deepset](https://www.deepset.ai) for
preprocessing your data for RAG
- How to structure your data and code for parallel pre-processing

# Preprocessing
```python
import json
from pathlib import Path
import pickle
from tqdm.auto import tqdm

from haystack.nodes.preprocessor import PreProcessor
```

```python
proj_dir = Path.cwd().parent
print(proj_dir)
```

    /home/ec2-user/arabic-wiki


It's really useful to pull this configuration out early so that we can easily control our notebook. Note that in 
[part 1]({{< ref "arabic-rag-1" >}}) we converted the downloaded wiki format into a list of ndjson files with `100_000` 
articles each. Here we are going to process each of these files.
```python
files_in = list((proj_dir / 'data/consolidated').glob('*.ndjson'))
folder_out = proj_dir / 'data/processed'
folder_out_str = str(folder_out)
```

```python
with open(files_in[0], 'r') as f:
    articles = [json.loads(line) for line in f]
```

Let's print out what an article looks like. 
```python
from pprint import pprint
article = articles[0].copy()
article['content'] = article['content'][:50] + '...'
pprint(article)
```

    {'content': 'الماء مادةٌ شفافةٌ عديمة اللون والرائحة، وهو المكو...',
     'meta': {'id': '7',
              'revid': '2080427',
              'title': 'ماء',
              'url': 'https://ar.wikipedia.org/wiki?curid=7'}}

This format is common for haystack. While we arent required to keep it as Im only using haystack for pre-processing, I 
think its clean.

## Considerations
It's really important to choose good pre-processing options. As this will heavily impact your RAG potential.
### Cleaning 
- Excess whitespace causes issues each stage of RAG. 
  - It adds noise to the embeddings
  - It wastes precious space in the prompt
  - Unless it has semantic meaning its best to clean it up

### Chunking
- In an ideal world we would split by tokens, alas thats not an easy option here, and it adds a fair amount of 
computational cost up front
- But based on [part 2]({{< ref "arabic-rag-2" >}}) we know that `225 words` is a pretty good estimate for ~512 tokens.
  - This way we dont throwaway information since our embedding model can only handle `512 tokens`
  - We confirmed this by analyzing the z-scores
- We will preserve semantic meanings with `split_respect_sentence_boundary = True`, otherwise the embedding model will
get inputs that its not used to
- We will also use an overlap of 50 words. This way each chunk wont be overloaded to be the only representation for its content
  - This was more important with [ExtractiveQA](https://huggingface.co/tasks/question-answering#task-variants), but can still be important
- There is a nice sanity check with `max_chars_check = 10_000`, this prevents malformed inputs from casuing issues.

### Other Use-cases
Chunking wikipedia is typically easier than most real-life use-cases. 
- Remember that a model might only get your chunk and no "extra" information
  - If this is in a pdf, consider adding "page i of n"
  - You might want to add the document title (We do this later on for subsequent chunks)
- You need to make your chunk retrievable.
  - You might want to add a summary if you can't naively chunk
- Know your domain and ask if you could answer questions based on the chunks

{{< notice tip >}}
Chunking is one of the most important parts of RAG. The LLM always gets the attention, but thats usually the least 
impactful part since LLMs are already quite good
{{< /notice >}}


```python
pp = PreProcessor(clean_whitespace = True,
             clean_header_footer = False,
             clean_empty_lines = True,
             remove_substrings = None,
             split_by='word',
             split_length = 225,
             split_overlap = 50,
             split_respect_sentence_boundary = True,
             tokenizer_model_folder = None,
             id_hash_keys = None,
             progress_bar = False,
             add_page_number = False,
             max_chars_check = 10_000)
```

## Data Augmentation
When we break a wikipedia article up, we lose some of the context. The local context is somewhat preserved by the 
`split_overlap`. Im trying to preserve the global context by adding a prefix that has the article's title.

You could enhance this with the summary as well (which can be a resource intesive task). This is mostly to help the 
retrieval step of RAG. Note that the way Im doing it alters some of `haystack`'s features like the hash and the lengths, but those arent too necessary for our usage.

A more advanced way for many business applications would be to summarize the document and add that as a prefix for sub-documents.

One last thing to note, is that it would be prudent (in some use-cases) to preserve the original document without the 
summary to give to the reader (retrieve with the summary but prompt without), but since this is a demo use-case I wont 
be doing that.


## Parallel Processing
{{< notice tip >}}
We will use multiple cores to process each document in parallel. This should make our processing much quicker
than just using one process.
{{< /notice >}}

We can easily prove this out:

```python
with open(files_in[0], 'r', encoding='utf-8') as f:
    articles = [json.loads(line) for line in f]
```
```python
%%time
documents = pp.process(articles)
```
    ...
    CPU times: user 3min 31s, sys: 95.1 ms, total: 3min 31s
    Wall time: 3min 31s

Ok, processing one group of articles took **3min 31s** of wall time. Lets see what happens with parallel processing. Do
you have any predictions? Comment below with how long you think it will take.


```python
%%time
import os
import concurrent.futures

def parallel_preprocessing(articles):
    # Utility function to divide the articles into smaller chunks
    def chunkify(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    # Size of each chunk. Adjust based on your needs.
    CHUNK_SIZE = 10_000  
    article_chunks = list(chunkify(articles, CHUNK_SIZE))

    # Number of processes to run in parallel.
    # Use all available CPUs, but you can reduce the number if you wish to leave some CPUs free.
    NUM_PROCESSES = os.cpu_count()  

    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_PROCESSES) as executor:
        documents_list = list(executor.map(pp.process, article_chunks))

    # Flatten the documents_list to get a single list of documents
    documents = [doc for sublist in documents_list for doc in sublist]
    return documents

documents = parallel_preprocessing(articles)

```
    ...
    CPU times: user 6.86 s, sys: 1.31 s, total: 8.16 s
    Wall time: 1min 33s

{{< notice note >}}
Wow! A huge improvement, we went from **3min 31s** to **1min 33s**!! Thats ~60% faster!
{{< /notice >}}

This is our pre-processing loop where we apply our `parallel_preprocessing`. Note that we prepend the title for 
documents of a later `_split_id` as discussed above.
```python
%%time
for file_in in tqdm(files_in):
    # Load articles
    with open(file_in, 'r', encoding='utf-8') as f:
        articles = [json.loads(line) for line in f]
        
    # Preprocess articles
    documents = parallel_preprocessing(articles)
    
    # Prefix each document's content
    for document in tqdm(documents):
        if document.meta['_split_id'] != 0:
            document.content = f'عنوان: {document.meta["title"]}. ' + document.content
            
    processed_articles = [document.to_dict() for document in documents]
    with open(folder_out/file_in.name, 'w', encoding='utf-8') as f:
        for article in processed_articles:
            json_str = json.dumps(article, ensure_ascii=False)
            f.write(json_str + '\n')
        
```


      0%|          | 0/23 [00:00<?, ?it/s]
      0%|          | 0/243068 [00:00<?, ?it/s]
      ...
      0%|          | 0/70322 [00:00<?, ?it/s]


    CPU times: user 2min 21s, sys: 20.1 s, total: 2min 41s
    Wall time: 13min 36s


## Pre-processing Examples
This is a good place to check out our chunks and make sure the look like what we expect. Notice in this first one we have
a `'_split_id': 0` so we didnt add the title here.
```python
documents[0]
```
    <Document: {'content': 'عشاء كارين هو من سلسلة مطاعم استرالية يهدف عمداً عن تجربة تناول طعام غير سارَة ويتم توجيه الموظفين لإهانة العملاء طوال وجباتهم.\nاقتبس اسم المطعم من المصطلح العامي على الإنترنت (كارين) والذي يستخدم لوصف امرأة بيضاء مسنة وقحة بشكل نمطي.\nتاريخ المطعم.\nتم إنشاء السلسلة في أستراليا (سيدني) في عام 2021 من قبل إيدين ليفن وجيمس فاريل. المطعم ذو طابع خاص يعتمد على خدمة تجربة طعام غير سارة حيث يدفع العملاء للموظفين لإهانتهم وكان من المفترض ان يكون المطعم مطعماً منبثقاً لمدة ستة أشهر في وورلد سكوير.\nاثارت فكرة المطعم في البداية ردات فعل متغايرة مما أثار الخوف بشأن ما إذا كانت الإهانات المتبادلة من الممكن ان تعرض الموظفين لسوء المعاملة من قبل العملاء.\nاسم (كارين) هو إشارة إلى الإسم المستخدم في الميمات (النكت التي تنشهر بسرعة في مواقع التواصل) لوصف امرأة بيضاء في منتصف العمر ووقحة بشكل نمطي.\nيطلب من الموظفين ارتداء شخصية وقحة والسخرية من العملاء بشكل هزلي اثناء تناول وجباتهم ومن المتوقع ان يعيد العملاء هذا السلوك من خلال التصرف بوقاحة تجاه الموظفين ومع ذلك يُحظر على العملاء والموظفين استخدام الإهانات العنصرية أو التحيز الجنسي أو رهاب المثلية الجنسية.\nتتضمن العديد من هذه التبادلات لغة نابية ويجب ان يكون برفقة الاشخاص اللذين يقلون عن 16 عاماََ بالغين.\nكما يمكن لمالكي بطاقة هوية تظهر ان اسمهم كارين الحصول على مشروب مجاني.\n', 'content_type': 'text', 'score': None, 'meta': {'id': '8974231', 'revid': '593870', 'url': 'https://ar.wikipedia.org/wiki?curid=8974231', 'title': 'مطعم عشاء كارين', '_split_id': 0, '_split_overlap': [{'doc_id': '288196225044b53e6ff86f2485257a0a', 'range': (790, 1225)}]}, 'id_hash_keys': ['content'], 'embedding': None, 'id': '1af84f3b4cc6a9f1018f2f80b4fd3ba7'}>

Here we have a `'_split_id': 1` and we can verify that `'content'` does in fact start with 'عنوان: مطعم عشاء كارين. ' 
which means our pre-processing worked! 
```python
documents[1]
```
    <Document: {'content': 'عنوان: مطعم عشاء كارين. يطلب من الموظفين ارتداء شخصية وقحة والسخرية من العملاء بشكل هزلي اثناء تناول وجباتهم ومن المتوقع ان يعيد العملاء هذا السلوك من خلال التصرف بوقاحة تجاه الموظفين ومع ذلك يُحظر على العملاء والموظفين استخدام الإهانات العنصرية أو التحيز الجنسي أو رهاب المثلية الجنسية.\nتتضمن العديد من هذه التبادلات لغة نابية ويجب ان يكون برفقة الاشخاص اللذين يقلون عن 16 عاماََ بالغين.\nكما يمكن لمالكي بطاقة هوية تظهر ان اسمهم كارين الحصول على مشروب مجاني.\nيرتكز المطعم علو وجبات العشاء الأمريكية في خمسينات القرن الماضي وتتميز القائمة بالهامبرغر وأجنحة الدجاج.\nأصبح محتوى شائع لوسائل التواصل الإجتماعي خصوصاً على منصة (تيك توك) حيث نشر العملاء مقاطع فيديو لتفاعلاتهم مع الموظفين.\nفتحت السلسلة في مواقع في المملكة المتحدة والولايات المتحدة ونيوزيلندا.\nفي شهر أغسطس سنة 2022 اثار المطعم جدلاً بعد ان انتشر مقطع يُظهر فيه أحد موظفي فريق العمل في منطقة بريزبين يتصرف بشكل غير لائق على منصة تيك توك حيث القى تعليقات غير لائقة موجهة إلى زبونة قاصر ووالدها الذي كان يشاركها الطعام بإتهامه انه يمارس الرذيلة مع الأطفال، فقام المتحدث باسم السلسلة بالرد بإنهم اصيبو بخيبة أمل بسبب السلوك وأن الحادث يتعارض مع إرشاداتهم.', 'content_type': 'text', 'score': None, 'meta': {'id': '8974231', 'revid': '593870', 'url': 'https://ar.wikipedia.org/wiki?curid=8974231', 'title': 'مطعم عشاء كارين', '_split_id': 1, '_split_overlap': [{'doc_id': '1af84f3b4cc6a9f1018f2f80b4fd3ba7', 'range': (0, 435)}]}, 'id_hash_keys': ['content'], 'embedding': None, 'id': '288196225044b53e6ff86f2485257a0a'}>

Lets check one more for the sake of being prudent.
```python
documents[10102]
```
    <Document: {'content': 'نيكولاي فليفا والمعروف أيضًا باسم نيكو فليفا (1840 - 4 أغسطس عام 1920) هو سياسي وصحفي سياسي ومحامي أفلاقي ومن ثم روماني. اشتهر بانخراطه في المجريات السياسية ونزعته الوطنية الصريحة التي كادت تصل إلى حد الديماغوجية. اختبر كافة الصيغ السياسية التي يسمح بها نظام الحزبين في رومانيا. دام حضوره على الساحة العامة عقودًا من الزمن، شغل خلالها مقعدًا في جمعية النواب وتولى منصب عمدة مدينة بوخارست خلال الفترة الممتدة من عام 1884 حتى عام 1886.\nباشر فليفا مسيرته السياسية مع الحزب الليبرالي الوطني الذي ساعد على تأسيسه وتمثيله أمام القضاء، ولكنه اتجه في ما بعد إلى معارضة احتكار الحزب للسلطة. حاول إنشاء حزب ثالث ودخل في مفاوضات من أجل اعتماد برامج سياسية مشتركة خاصة بقوى المعارضة المختلفة ومن بينها حزب المحافظين وجمعية جونيما في ظل الإدارات الليبرالية الوطنية المتعاقبة. ذاع صيته عندما تورط في فضيحتين كبيرتين خلال ثمانينيات القرن التاسع عشر حين أدى استهزائه بسلطة الحزب الليبرالي الوطني إلى اندلاع معارك في الشوارع ووقوع حادثتي إطلاق النار منفصلتين. اعتُبرت الجماعات الموالية لفليفا الصوت الرائد المعبر عن سخط الطبقة الوسطى وقتذاك، وشكلت إحدى التيارات التي دفعت باتجاه تبني حق الاقتراع العام للذكور.\nعاد فليفا إلى المعسكر الليبرالي الوطني بعد منعه من تولي حقائب وزارية ريادية في الحكومات المحافظة، وأصبح وزيرًا للشؤون الداخلية خلال الفترة من عام 1895 حتى عام 1896. ', 'content_type': 'text', 'score': None, 'meta': {'id': '9044009', 'revid': '1673186', 'url': 'https://ar.wikipedia.org/wiki?curid=9044009', 'title': 'نيكولاي فليفا', '_split_id': 0, '_split_overlap': [{'doc_id': '188181b1026773d720383c7e7307b241', 'range': (943, 1257)}]}, 'id_hash_keys': ['content'], 'embedding': None, 'id': 'af5cda4722fa2a961bef66de8a6b3e17'}>

```python
!cat "$folder_out_str"/*.ndjson | wc -l
```
    2094596
Here we can see the total amount of articles thanks to our handy `.ndjson` format. Note that some of the articles have
been filtered out. It's worth diving deeper into that, for production use-cases.

# Next steps
In the next blog post we will talk about an efficient way to get embeddings for our chunks! You can see in my recent 
[poll](https://www.linkedin.com/posts/dthomas_embedding-your-chunks-is-an-important-step-activity-7130532352178462722-2xoo)
that there is not much of a concensus on how expensive this is for ~2M articles. I'll show you a quick, easy, and very 
cheap way to go about this!
    
